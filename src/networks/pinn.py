# =============================================================================
# PINN policy: π_φ(ε, θ) → δ  (physics-informed; drop-in for the CNN/MLP)
# =============================================================================
"""
Why a PINN should beat the CNN here (in theory)
===============================================
The CNN ([networks/cnn.py]) is a strong but *generic* image→image map: it has
to learn the ε→δ relationship purely from Evolution-Strategies reward. ES is
sample-expensive — every population rollout is up to T FDFD solves — so the
inductive bias of the policy directly sets how fast it learns and how well it
extrapolates to unseen (state, goal) pairs. This PINN replaces "generic spatial
prior" with the *actual governing physics* of the device:

  2-D TM Helmholtz equation for the out-of-plane field E_z:

        ∇²E_z(x, y) + k₀² ε_r(x, y) E_z(x, y) = -S(x, y)

  where k₀ = ω/c is the free-space wavenumber, ε_r is the relative permittivity
  the policy controls (rods), and S is the line source on the left wall.

We inject that structure three ways:

  1. PDE COEFFICIENTS AS INPUT FEATURES.
     At every rod the trunk sees the local coefficients of the Helmholtz
     operator it is steering: ε_r, ∇ε_r (index gradient — the thing that
     scatters the wave), ∇²ε_r, and the potential term k₀²ε_r. A CNN must
     *discover* these from raw ε through many conv layers; the PINN is handed
     them, computed with the same finite-difference stencils FDFD itself uses.

  2. PLANE-WAVE TARGET PRIOR (physics-informed goal conditioning).
     The goal k is a beam DIRECTION. Instead of a bare (sin, cos) angle token,
     we feed the *phase ramp of the desired far field* — a plane wave toward
     the receiver azimuth β(k):
         target_phase(x, y) = k₀ (x cos β + y sin β),
     supplied as (cos, sin). This tells the network the WAVE STRUCTURE of the
     objective, so "steer toward k" is grounded in the field it implies, not an
     abstract index. This is the single biggest reason the PINN should
     generalize across angles from fewer rollouts than the CNN.

  3. SELF-SUPERVISED HELMHOLTZ RESIDUAL (the actual PINN loss).
     A small auxiliary field head predicts Ê_z(x, y) = (Re, Im) from the SHARED
     trunk. During the AWR warm-start we add a physics-consistency penalty

         L_phys = mean|∇²Ê_z + k₀²ε_r Ê_z + S|² / mean(term magnitudes)

     i.e. a RELATIVE residual: the squared residual is divided by the typical
     magnitude of its own terms (detached), so L_phys ≈ O(1) regardless of k₀,
     grid spacing, or field amplitude. The source term S is what makes this
     non-trivial: Ê_z ≡ 0 would satisfy a source-free residual, but with S≠0 the
     network is forced to predict a real propagating field (Ê_z ≡ 0 → L_phys=1).
     Because the field head and the δ head share the trunk, this gradient shapes
     the *policy's* features to be wave-consistent — the mechanism by which
     physics knowledge transfers to the control output.
     (Wire-in: ESPolicyConfig.physics_loss_weight; see es_policy.awr_init. It is
     applied ONLY to policies that expose `helmholtz_residual_loss`, so the
     CNN/MLP are untouched. The relative form keeps physics_loss_weight an
     interpretable O(0.1) knob that survives recalibrating k₀.)

Honest caveats
==============
- k₀ IS calibrated to the pm_setup device (f = 6 GHz, rod pitch 0.022 m, 10×10
  → k0_norm ≈ 12.45, ≈ 3.96 λ across the array); see PINNPolicy.calibrated_k0
  to recompute for other devices. So the target-beam wavelength and the
  residual's potential term k₀²ε_r match the simulation's spatial frequency.
- The ε→ε_r map (eps_r_mid ± eps_r_span) is still a PRIOR: the normalized state
  ε∈[-1,1] is mapped to a generic dielectric range, NOT the simulation's true
  rod permittivity (which converter.py sets in absolute units). It only fixes
  the relative feature/potential scale, which the trunk can rescale.
- The residual uses replicate padding at the domain border (a Neumann-ish
  stand-in for the true wall/PML BCs), so L_phys is a soft consistency
  regularizer over the rod region, not a calibrated full-domain PDE solve.
- The field head only affects forward() indirectly (through the shared trunk);
  it is dead weight under ES (its params do not change δ). Kept small for that
  reason, and trained only by L_phys during AWR.

Drop-in contract
================
Same (state_shape, n_goals, hidden_dim, n_hidden_layers, tanh_output,
tanh_output_scale) constructor and forward(eps, goal) -> delta signature as the
CNN/MLP, so ESPolicy builds and rolls it out unchanged. PINN-specific knobs
carry their own defaults. hidden_dim → trunk width; n_hidden_layers → trunk
depth (number of pointwise layers).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Discrete differential operators on a (B, C, H, W) grid (H = x-axis = N_x,
# W = y-axis = N_y). Replicate padding => Neumann-ish border; the same 5-point
# Laplacian / central-difference stencils FDFD discretizes with.
# -----------------------------------------------------------------------------
def _grad_xy(t, hx, hy):
    """Central-difference ∂/∂x and ∂/∂y; returns (gx, gy), each (B, C, H, W)."""
    tp = F.pad(t, (1, 1, 1, 1), mode="replicate")
    gx = (tp[..., 2:, 1:-1] - tp[..., :-2, 1:-1]) / (2.0 * hx)   # along H (x)
    gy = (tp[..., 1:-1, 2:] - tp[..., 1:-1, :-2]) / (2.0 * hy)   # along W (y)
    return gx, gy


def _laplacian(t, hx, hy):
    """5-point Laplacian ∇²t; returns (B, C, H, W)."""
    tp = F.pad(t, (1, 1, 1, 1), mode="replicate")
    c = tp[..., 1:-1, 1:-1]
    d2x = (tp[..., 2:, 1:-1] - 2.0 * c + tp[..., :-2, 1:-1]) / (hx * hx)
    d2y = (tp[..., 1:-1, 2:] - 2.0 * c + tp[..., 1:-1, :-2]) / (hy * hy)
    return d2x + d2y


class PINNPolicy(nn.Module):
    """
    Physics-informed, coordinate/pointwise goal-conditioned retargeting policy.
    See module docstring for the physics rationale.

    Inputs (batched):
        eps:  (B, N_x, N_y)  — current normalized permittivity ∈ [-1, 1]
        goal: (B,) int       — receiver index k; beam azimuth β = π·k/(n_goals-1)
                               (SEMICIRCLE, matching the ~180° receiver fan).
    Output:
        delta: (B, N_x, N_y) — perturbation in ε-space, optionally tanh-scaled.
    """

    def __init__(
        self,
        state_shape,
        n_goals,
        hidden_dim=256,          # -> trunk width
        n_hidden_layers=3,       # -> trunk depth (pointwise layers)
        tanh_output=True,
        tanh_output_scale=1.0,
        # --- PINN-specific priors (own defaults; tunable, not calibrated) ---
        trunk_width=None,        # default: hidden_dim
        n_fourier=4,             # coordinate Fourier features (combat spectral bias)
        # Wavenumber on the normalized [-1,1]² grid, CALIBRATED to the pm_setup:
        #   f = 6 GHz, c → λ ≈ 49.97 mm; rod pitch = 2·radius + distance
        #   = 2·0.01 + 0.002 = 0.022 m; the rod-center span (N-1)·pitch = 0.198 m
        #   (≈ 3.96 λ) maps to the normalized length 2, so
        #       k0_norm = (2π f / c) · (N-1)·pitch / 2 ≈ 12.45.
        # Use PINNPolicy.calibrated_k0(...) to recompute for a different device.
        k0=12.4493,
        eps_r_mid=6.0,           # ε_r = eps_r_mid + eps_r_span·eps   (eps∈[-1,1])
        eps_r_span=5.0,          #   default range [1, 11] (high-index dielectric)
        source_amp=1.0,          # left-wall line-source amplitude (residual only)
        source_halfwidth=None,   # source vertical half-extent in cells (default N_y//5)
    ):
        super().__init__()
        N_x, N_y = state_shape
        self.state_shape = (N_x, N_y)
        self.n_goals = n_goals
        self.tanh_output = tanh_output
        self.tanh_output_scale = float(tanh_output_scale)

        width = int(hidden_dim if trunk_width is None else trunk_width)
        depth = max(int(n_hidden_layers), 1)
        self.n_fourier = int(n_fourier)
        self.k0 = float(k0)
        self.eps_r_mid = float(eps_r_mid)
        self.eps_r_span = float(eps_r_span)

        # Grid spacing on normalized coords x, y ∈ [-1, 1].
        self.hx = 2.0 / max(N_x - 1, 1)
        self.hy = 2.0 / max(N_y - 1, 1)

        # --- Static buffers (NOT parameters: don't inflate the ES dim) ------
        xs = torch.linspace(-1.0, 1.0, N_x).view(1, 1, N_x, 1).expand(1, 1, N_x, N_y)
        ys = torch.linspace(-1.0, 1.0, N_y).view(1, 1, 1, N_y).expand(1, 1, N_x, N_y)
        self.register_buffer("xs", xs.contiguous())
        self.register_buffer("ys", ys.contiguous())

        # Coordinate Fourier features (sin/cos of 2^l·π·{x,y}); fixed → buffer.
        feats = []
        for l in range(self.n_fourier):
            f = (2.0 ** l) * math.pi
            feats += [torch.sin(f * xs), torch.cos(f * xs),
                      torch.sin(f * ys), torch.cos(f * ys)]
        fourier = torch.cat(feats, dim=1) if feats else xs[:, :0]
        self.register_buffer("fourier", fourier.contiguous())   # (1, 4F, H, W)

        # Left-wall line source for the Helmholtz residual: amplitude on the
        # x=0 edge (dim H, index 0), centered vertically (dim W). Real-valued.
        hw = int(source_halfwidth if source_halfwidth is not None else max(N_y // 5, 1))
        src = torch.zeros(1, 1, N_x, N_y)
        yc = N_y // 2
        src[:, :, 0, max(yc - hw, 0): yc + hw + 1] = float(source_amp)
        self.register_buffer("source", src.contiguous())

        # Input channel count (see _assemble for the exact order):
        #   eps(1) + ε_r(1) + ∇ε_r(2) + ∇²ε_r(1) + k0²ε_r(1)
        #   + coords(2) + fourier(4F) + target plane wave(2) + (cosβ,sinβ)(2)
        in_ch = 1 + 1 + 2 + 1 + 1 + 2 + 4 * self.n_fourier + 2 + 2

        # --- Pointwise (1×1) trunk: a shared per-rod MLP. GELU activations. ---
        layers = [nn.Conv2d(in_ch, width, 1), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 1), nn.GELU()]
        self.trunk = nn.Sequential(*layers)

        # δ head (policy output). Small init → starts near "do nothing".
        self.delta_head = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.delta_head.weight, std=1e-2)
        nn.init.zeros_(self.delta_head.bias)

        # Field head (auxiliary): predicts Ê_z = (Re, Im) for the PINN residual.
        self.field_head = nn.Conv2d(width, 2, 1)

    # ----- calibrate k0 to a physical device -------------------------------
    @staticmethod
    def calibrated_k0(frequency_hz, n_rods, pitch_m=None, radius_m=None,
                      distance_m=None, c=2.99792458e8):
        """
        Wavenumber in the PINN's normalized-grid units (where the rod-center
        span maps to length 2), from the device's physical parameters:

            k0_norm = (2π · frequency / c) · (n_rods - 1) · pitch / 2

        Pass `pitch_m` directly, or `radius_m` + `distance_m` (pitch = wall gap
        + rod diameter = distance + 2·radius, matching geometry.create_grid).
        For pm_setup: calibrated_k0(6e9, 10, radius_m=0.01, distance_m=0.002)
        ≈ 12.45. Assumes near-isotropic pitch (N_x ≈ N_y, equal x/y spacing).
        """
        if pitch_m is None:
            if radius_m is None or distance_m is None:
                raise ValueError("provide pitch_m, or both radius_m and distance_m.")
            pitch_m = distance_m + 2.0 * radius_m
        k0_phys = 2.0 * math.pi * frequency_hz / c
        span = (n_rods - 1) * pitch_m
        return k0_phys * (span / 2.0)

    # ----- goal → beam azimuth β (SEMICIRCLE, same convention as CNN/MLP) ---
    def _beta(self, goal):
        denom = max(self.n_goals - 1, 1)
        return (math.pi / denom) * goal.float()          # (B,)

    # ----- assemble the physics-informed per-rod feature stack -------------
    def _assemble(self, eps, goal):
        B, N_x, N_y = eps.shape[0], *self.state_shape
        e = eps.unsqueeze(1)                              # (B, 1, H, W)

        # Helmholtz coefficients.
        eps_r = self.eps_r_mid + self.eps_r_span * e      # (B, 1, H, W)
        gx, gy = _grad_xy(eps_r, self.hx, self.hy)        # ∇ε_r
        lap = _laplacian(eps_r, self.hx, self.hy)         # ∇²ε_r
        potential = (self.k0 ** 2) * eps_r                # k₀²ε_r

        # Coordinates + Fourier features (static, broadcast over batch).
        xs = self.xs.expand(B, -1, -1, -1)
        ys = self.ys.expand(B, -1, -1, -1)
        fourier = self.fourier.expand(B, -1, -1, -1)

        # Plane-wave target prior toward azimuth β(goal).
        beta = self._beta(goal).view(B, 1, 1, 1)
        cb, sb = torch.cos(beta), torch.sin(beta)
        phase = self.k0 * (self.xs * cb + self.ys * sb)   # (B, 1, H, W)
        tgt_cos, tgt_sin = torch.cos(phase), torch.sin(phase)
        cb_map = cb.expand(B, 1, N_x, N_y)
        sb_map = sb.expand(B, 1, N_x, N_y)

        feats = torch.cat(
            [e, eps_r, gx, gy, lap, potential,
             xs, ys, fourier, tgt_cos, tgt_sin, cb_map, sb_map],
            dim=1,
        )
        return feats, eps_r

    def forward(self, eps, goal):
        feats, _ = self._assemble(eps, goal)
        h = self.trunk(feats)
        delta = self.delta_head(h).squeeze(1)             # (B, H, W)
        if self.tanh_output:
            delta = self.tanh_output_scale * torch.tanh(delta)
        return delta

    # ----- PINN auxiliary loss: relative Helmholtz residual ----------------
    def helmholtz_residual_loss(self, eps, goal):
        """
        RELATIVE (self-normalizing) discrete Helmholtz residual of the field
        head's prediction:

            R = ∇²Ê_z + k₀² ε_r Ê_z + S ,   Ê_z = (Re, Im)

            L_phys = mean(|R|²) / mean(|∇²Ê_z|² + |k₀²ε_r Ê_z|² + S²)

        The denominator is the typical magnitude of the terms that COMPOSE the
        residual, detached so the optimizer minimizes R rather than inflating
        the normalizer. This keeps L_phys ≈ O(1) regardless of k₀, grid
        spacing, or the (initially random) field-head amplitude — so
        physics_loss_weight stays an interpretable O(0.1) knob and is robust to
        recalibrating k₀ via calibrated_k0(). A field that solves the PDE →
        L_phys ≈ 0; the trivial Ê_z ≡ 0 → R = S → L_phys = 1 (penalized, so the
        source term still forces a real propagating field).
        """
        feats, eps_r = self._assemble(eps, goal)
        h = self.trunk(feats)
        E = self.field_head(h)                            # (B, 2, H, W)
        E_re, E_im = E[:, 0:1], E[:, 1:2]

        lap_re = _laplacian(E_re, self.hx, self.hy)
        lap_im = _laplacian(E_im, self.hx, self.hy)
        pot = (self.k0 ** 2) * eps_r                      # (B, 1, H, W)
        pot_re, pot_im = pot * E_re, pot * E_im

        R_re = lap_re + pot_re + self.source              # source on the real part
        R_im = lap_im + pot_im

        # Characteristic term scale (detached → not gameable by inflating Ê_z).
        scale = (lap_re ** 2 + lap_im ** 2
                 + pot_re ** 2 + pot_im ** 2
                 + self.source ** 2).mean().detach() + 1e-12
        return (R_re ** 2 + R_im ** 2).mean() / scale


# Drop-in alias, matching networks/cnn.py and networks/mlp.py.
PolicyNetwork = PINNPolicy


if __name__ == "__main__":
    # Shape / drop-in smoke test against the 10×10 pm_setup geometry.
    torch.manual_seed(0)
    net = PINNPolicy(state_shape=(10, 10), n_goals=30,
                     hidden_dim=256, n_hidden_layers=3,
                     tanh_output=True, tanh_output_scale=0.25)
    eps = torch.empty(4, 10, 10).uniform_(-1, 1)
    goal = torch.randint(0, 30, (4,))
    delta = net(eps, goal)
    res = net.helmholtz_residual_loss(eps, goal)
    n_params = sum(p.numel() for p in net.parameters())
    assert delta.shape == (4, 10, 10), delta.shape
    assert torch.all(delta.abs() <= 0.25 + 1e-5), delta.abs().max().item()
    # Relative residual is O(1): ~1 for an untrained (near-zero-ish) field head.
    assert 0.0 <= res.item() <= 100.0, res.item()
    print(f"ok: out={tuple(delta.shape)}  |delta|max={delta.abs().max():.4f}  "
          f"helmholtz_residual={res.item():.4g}  params={n_params}  (ES dim d)")
