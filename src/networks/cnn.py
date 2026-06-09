# =============================================================================
# CNN policy: π_φ(ε, θ) → δ  (fully-convolutional, physics-aware drop-in for
# the flatten-and-MLP PolicyNetwork in algorithms/policies/es_policy.py)
# =============================================================================
"""
Why a CNN here (vs. the flatten-MLP baseline)
=============================================
The state ε is NOT an unstructured length-100 vector — it is the permittivity
of a 10×10 lattice of plasma rods (state_shape = (N_x, N_y)). The flatten-MLP
destroys that 2D structure on the very first layer: rod (i, j) and its
neighbour (i, j+1) become two arbitrary coordinates in a 100-vector with no
built-in notion of adjacency. A convolutional policy instead bakes in the
physics:

  1. LOCAL ROD COUPLING (3×3 convolutions).
     A plasma rod's contribution to the field is governed by the *local*
     refractive-index landscape: scattering is driven by index gradients
     between a rod and its immediate neighbours. A 3×3 kernel is exactly the
     near-neighbour stencil that sets up the effective waveguide / GRIN-lens
     the beam follows. Stacking blocks grows the receptive field until it
     spans the whole 10×10 region (the lens is a global object).

  2. ABSOLUTE POSITION MATTERS — so this is NOT translation invariant.
     Vanilla convolution is translation-EQUIVARIANT, but the physical problem
     is not translation-invariant: the source is pinned to the LEFT wall
     (centered vertically) and the receivers sit on fixed RIGHT/TOP/BOTTOM
     boundaries. We break the symmetry with two constant CoordConv channels
     (normalized x, y meshgrid), so the network can learn "rods near the
     source wall behave differently from rods near the receiver boundary"
     without us hard-coding it.

  3. BEAM DIRECTION IS A GLOBAL, EMERGENT PROPERTY (global-context branch).
     The steering angle is not decidable from any local patch — it is set by
     the whole lens at once. A purely local CNN cannot see it. We add a
     global-average-pool → MLP → broadcast term so every rod is told the
     aggregate field summary before it decides its own δ.

  4. GOAL CONDITIONING MATCHES THE SEMICIRCLE GEOMETRY (FiLM + tiled channels).
     The goal is a target receiver index k, encoded EXACTLY as in the MLP
     baseline: angle = π·k/(n_goals−1), feature = (sin, cos), so the 30
     receivers map onto a SEMICIRCLE (k=0 → angle 0; k=N−1 → angle π),
     matching the ~180° physical receiver fan around the left-wall source.
     The goal is a *global* objective (it sets the whole target field), so it
     is injected globally via FiLM (per-channel affine modulation of every
     feature map) and additionally tiled as two input channels so early
     layers see it locally too.

Drop-in contract
================
Same constructor kwargs as the MLP PolicyNetwork
    (state_shape, n_goals, hidden_dim, n_hidden_layers, tanh_output,
     tanh_output_scale)
and the same forward(eps, goal) -> delta signature
    eps  : (B, N_x, N_y)  float
    goal : (B,)           long   receiver index k ∈ [0, n_goals-1]
    delta: (B, N_x, N_y)  float, optionally tanh-scaled
so ESPolicy can construct and roll it out unchanged. CNN-specific knobs
(channels, n_blocks, ...) carry their own defaults; the MLP's hidden_dim /
n_hidden_layers are re-purposed (see __init__) so an existing ESPolicyConfig
still wires up sensibly.

NOTE on normalization & ES. Phase 2 trains φ with Evolution Strategies and
rolls the policy out one state at a time (batch = 1 at inference). BatchNorm is
therefore unusable (batch-1 statistics + running-stat drift under perturbed
parameters). We use GroupNorm, which is batch-independent and deterministic
per sample, so there is no train/eval mismatch and ES sees a stationary
objective. CoordConv / goal maps are registered as buffers (not parameters),
so they do NOT inflate the ES search dimension d.
"""

import math

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# FiLM: feature-wise linear modulation from the goal encoding.
#   feat <- gamma(goal) * feat + beta(goal),  broadcast over the spatial dims.
# This is how the (global) target angle reshapes every feature map.
# -----------------------------------------------------------------------------
class _FiLM(nn.Module):
    def __init__(self, goal_dim, channels, hidden):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(goal_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )
        self.channels = channels

    def forward(self, feat, goal_feat):
        gb = self.to_gamma_beta(goal_feat)              # (B, 2C)
        gamma, beta = gb[:, : self.channels], gb[:, self.channels:]
        # (B, C) -> (B, C, 1, 1) for broadcast over (H, W).
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        # 1 + gamma so the layer starts near identity (gamma≈0 at init).
        return (1.0 + gamma) * feat + beta


# -----------------------------------------------------------------------------
# Residual conv block: Conv-Norm-FiLM-GELU -> Conv-Norm-GELU (+ skip).
# 3×3 kernels = near-neighbour rod stencil; residual keeps deep stacks
# trainable and starts each block near identity (good for ES warm-starts).
# -----------------------------------------------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, channels, goal_dim, film_hidden, n_groups, padding_mode):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1,
                               padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1,
                               padding_mode=padding_mode)
        self.norm1 = nn.GroupNorm(n_groups, channels)
        self.norm2 = nn.GroupNorm(n_groups, channels)
        self.film = _FiLM(goal_dim, channels, film_hidden)
        self.act = nn.GELU()

    def forward(self, x, goal_feat):
        h = self.act(self.film(self.norm1(self.conv1(x)), goal_feat))
        h = self.act(self.norm2(self.conv2(h)))
        return x + h


class CNNPolicyNetwork(nn.Module):
    """
    Fully-convolutional, goal-conditioned retargeting policy for the plasma
    rod array. See module docstring for the physics rationale.

    Inputs (batched):
        eps:  (B, N_x, N_y)  — current normalized permittivity ∈ [-1, 1]
        goal: (B,) int       — receiver index k, encoded as the SEMICIRCLE
                               (sin, cos) feature angle = π·k/(n_goals-1).
    Output:
        delta: (B, N_x, N_y) — perturbation in ε-space, optionally tanh-scaled
                               to (-tanh_output_scale, +tanh_output_scale).
    """

    def __init__(
        self,
        state_shape,
        n_goals,
        hidden_dim=256,          # MLP-compat: re-used as the FiLM/global MLP width
        n_hidden_layers=3,       # MLP-compat: re-used as the number of conv ResBlocks
        tanh_output=True,
        tanh_output_scale=1.0,
        # --- CNN-specific knobs (own defaults; safe to ignore) -----------
        channels=32,             # conv feature width (per-rod hidden dim)
        n_blocks=None,           # overrides n_hidden_layers if given
        n_groups=8,              # GroupNorm groups (batch-independent norm)
        use_global_context=True, # GAP→MLP→broadcast (sees the global beam)
        padding_mode="zeros",    # physical walls bound the domain; "replicate"
                                 # is a reasonable alt to "zeros"
    ):
        super().__init__()
        N_x, N_y = state_shape
        self.state_shape = (N_x, N_y)
        self.n_goals = n_goals
        self.tanh_output = tanh_output
        self.tanh_output_scale = float(tanh_output_scale)

        n_blocks = int(n_hidden_layers if n_blocks is None else n_blocks)
        n_blocks = max(n_blocks, 1)
        film_hidden = int(hidden_dim)
        # GroupNorm needs channels % n_groups == 0; fall back gracefully.
        n_groups = math.gcd(n_groups, channels) or 1

        goal_dim = 2   # (sin, cos)

        # --- Constant input maps (buffers, NOT ES-perturbed parameters) ---
        # CoordConv: normalized x, y in [-1, 1] so conv can use absolute
        # position (fixed source wall / receiver boundaries).
        xs = torch.linspace(-1.0, 1.0, N_x).view(N_x, 1).expand(N_x, N_y)
        ys = torch.linspace(-1.0, 1.0, N_y).view(1, N_y).expand(N_x, N_y)
        coords = torch.stack([xs, ys], dim=0).unsqueeze(0)   # (1, 2, N_x, N_y)
        self.register_buffer("coords", coords)

        # Input channels: eps(1) + coord(2) + tiled goal(2).
        in_ch = 1 + 2 + goal_dim
        self.stem = nn.Conv2d(in_ch, channels, 3, padding=1,
                              padding_mode=padding_mode)
        self.stem_act = nn.GELU()

        self.blocks = nn.ModuleList([
            _ResBlock(channels, goal_dim, film_hidden, n_groups, padding_mode)
            for _ in range(n_blocks)
        ])

        # Global-context branch: GAP over (H, W) -> MLP -> per-channel bias
        # broadcast back to every rod. Lets each rod see the aggregate field.
        self.use_global_context = use_global_context
        if use_global_context:
            self.global_mlp = nn.Sequential(
                nn.Linear(channels, film_hidden),
                nn.GELU(),
                nn.Linear(film_hidden, channels),
            )

        # Head: 1×1 conv to a single δ channel (per-rod perturbation).
        self.head = nn.Conv2d(channels, 1, 1)
        # Small (not zero) head init: the policy starts near "do nothing"
        # (|δ|≈0.05, far from tanh saturation) so the warm-started / ES
        # rollout begins gentle rather than as a saturated jump — but the
        # output still depends on (ε, goal) at init, so goal conditioning is
        # exercised from the first forward pass.
        nn.init.normal_(self.head.weight, std=1e-2)
        nn.init.zeros_(self.head.bias)

    # ----- goal encoding: identical SEMICIRCLE map to the MLP baseline -----
    def _goal_feat(self, goal):
        # k=0 → angle 0 → (0, 1) ; k=n_goals-1 → angle π → (0, -1).
        denom = max(self.n_goals - 1, 1)
        angle = (math.pi / denom) * goal.float()
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # (B,2)

    def forward(self, eps, goal):
        B, N_x, N_y = eps.shape[0], *self.state_shape

        goal_feat = self._goal_feat(goal)                          # (B, 2)

        x = eps.unsqueeze(1)                                       # (B, 1, H, W)
        coords = self.coords.expand(B, -1, -1, -1)                # (B, 2, H, W)
        goal_map = goal_feat.view(B, 2, 1, 1).expand(B, 2, N_x, N_y)
        x = torch.cat([x, coords, goal_map], dim=1)               # (B, 5, H, W)

        h = self.stem_act(self.stem(x))
        for block in self.blocks:
            h = block(h, goal_feat)

        if self.use_global_context:
            pooled = h.mean(dim=(2, 3))                           # (B, C)
            ctx = self.global_mlp(pooled)                        # (B, C)
            h = h + ctx.unsqueeze(-1).unsqueeze(-1)              # broadcast

        delta = self.head(h).squeeze(1)                          # (B, H, W)
        if self.tanh_output:
            delta = self.tanh_output_scale * torch.tanh(delta)
        return delta


# Drop-in alias: `from networks.cnn import PolicyNetwork` swaps the MLP for the
# CNN with no other call-site change (same constructor kwargs + forward).
PolicyNetwork = CNNPolicyNetwork


if __name__ == "__main__":
    # Shape / drop-in smoke test against the 10×10 pm_setup geometry.
    torch.manual_seed(0)
    net = CNNPolicyNetwork(state_shape=(10, 10), n_goals=30,
                           hidden_dim=256, n_hidden_layers=3,
                           tanh_output=True, tanh_output_scale=0.25)
    eps = torch.empty(4, 10, 10).uniform_(-1, 1)
    goal = torch.randint(0, 30, (4,))
    delta = net(eps, goal)
    n_params = sum(p.numel() for p in net.parameters())
    assert delta.shape == (4, 10, 10), delta.shape
    assert torch.all(delta.abs() <= 0.25 + 1e-5), delta.abs().max().item()
    print(f"ok: out={tuple(delta.shape)}  |delta|max={delta.abs().max():.4f}  "
          f"params={n_params}  (ES search dim d)")
