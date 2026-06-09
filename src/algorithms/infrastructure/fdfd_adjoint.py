# =============================================================================
# Differentiable FDFD: ceviche adjoint ↔ PyTorch bridge
# =============================================================================
"""
A `torch.autograd.Function` (and an `nn.Module` wrapper) that wraps the true
ceviche FDFD solve so it can be dropped into ESPolicy.train_phase2_grad as the
`power_model`, giving UNBIASED analytic policy gradients (vs. the surrogate M,
which is biased by its approximation error).

The differentiable chain is

    ε_rod (N_x, N_y)                      # the policy's permittivity state
      │  linear scatter onto the FDFD canvas (rod disks; static masks)
      ▼
    eps_r canvas (H_c, W_c)
      │  ceviche.fdfd_ez(...).solve(source)   # Maxwell solve — adjoint-differentiable
      ▼
    E_z field
      │  P_j = Σ |E_z|² · receiver_mask_j     # receiver powers
      ▼
    P (n_recv,)

ceviche is built on HIPS autograd, so the whole chain `ε → P` is expressed in
`autograd.numpy` and differentiated by reverse-mode autograd. `make_vjp` runs
ONE forward solve and returns a closure that, given the upstream cotangent
∂L/∂P, runs ONE adjoint solve to produce ∂L/∂ε. We stash that closure on the
autograd context so PyTorch's backward is exactly one adjoint solve per sample.

Cost: ~1 forward + ~1 adjoint solve per (sample, BPTT step). With B tasks and
horizon T that's ~2·B·T solves per gradient step — far more than the surrogate,
but the gradient is the true ∂P/∂ε with no surrogate bias. Use small B/T (and
ESPolicy.train_phase2_grad's `bptt_truncate`) or reserve this for fine-tuning a
surrogate-pretrained policy.

Correctness guarantee: the forward here is built from the SAME stamping rule as
algorithms/agents/es_agent.apply_eps_to_canvas (canvas[disk(rod)] = ε_rod, rod
disks disjoint from walls/PML), so `FDFDPowerModel(env)(ε)` reproduces
`get_receiver_powers` up to float tolerance — verified in __main__.
"""

import numpy as np
import torch
import torch.nn as nn

import autograd.numpy as npa
from autograd import make_vjp
from skimage.draw import disk


class FDFDAdjointSolver:
    """Static FDFD pieces extracted from an initialized env + the autograd
    `ε_flat → P` map. ε_flat is the (N_x·N_y,) row-major (x-major) rod vector,
    i.e. ε_2d.reshape(-1) with ε_2d[x-1, y-1] = rod (x, y)."""

    def __init__(self, env):
        dr = env.design_region
        canvas = dr._canvas
        H_c, W_c = canvas.shape
        self.dL = dr.resolution
        self.npml = int(dr.num_pml_cells)
        self.canvas_shape = (H_c, W_c)

        N_x, N_y = env.grid.num_rods_x, env.grid.num_rods_y
        self.state_shape = (N_x, N_y)
        self.n_rods = N_x * N_y
        radius_cells = env.grid.radius / dr.resolution

        # Per-rod disk masks, ordered by the row-major flat index of ε_2d
        # (k = (x-1)·N_y + (y-1)), and a `base` canvas with the rod cells
        # zeroed so that  canvas = base + Σ_k ε_k · mask_k  reproduces
        # apply_eps_to_canvas exactly (rod cell → ε_k; everything else → base).
        base = np.asarray(canvas, dtype=np.float64).copy()
        mask_stack = np.zeros((self.n_rods, H_c, W_c), dtype=np.float64)
        for (x, y), rod in env.grid.rods.items():
            rr, cc = disk(center=rod._center, radius=radius_cells, shape=canvas.shape)
            k = (x - 1) * N_y + (y - 1)
            mask_stack[k, rr, cc] = 1.0
            base[rr, cc] = 0.0
        self.base = base
        self.mask_stack = mask_stack

        # Sources: (ω, source_vector) per source — matches simulation.py
        # (amplitude 1e3 on the source mask), summed COHERENTLY (Σ E_z) before
        # taking intensity, exactly like get_receiver_powers.
        self.sources = []
        for s in env.sources:
            omega = 2.0 * np.pi * s.frequency
            svec = np.zeros((H_c, W_c), dtype=np.complex128)
            svec[s._mask == 1] = 1e3
            self.sources.append((omega, svec))

        # Receiver masks stacked: (n_recv, H_c, W_c), ordered as env.receivers.
        self.recv_stack = np.stack(
            [np.asarray(r._mask, dtype=np.float64) for r in env.receivers], axis=0
        )
        self.n_recv = self.recv_stack.shape[0]

    # autograd.numpy: ε_flat (n_rods,) → P (n_recv,)
    def eps_to_P(self, eps_flat):
        import ceviche  # imported lazily so the module loads without a solve
        canvas = self.base + npa.tensordot(eps_flat, self.mask_stack, axes=([0], [0]))
        ez_total = 0.0
        for omega, svec in self.sources:
            sim = ceviche.fdfd_ez(omega, self.dL, canvas, [self.npml, self.npml])
            _, _, ez = sim.solve(svec)
            ez_total = ez_total + ez
        intensity = npa.abs(ez_total) ** 2
        # P_j = Σ intensity · recv_mask_j
        return npa.sum(intensity[None, :, :] * self.recv_stack, axis=(1, 2))


class _FDFDPowerFn(torch.autograd.Function):
    """Autograd bridge: forward solves FDFD for P; backward runs one adjoint
    solve via the stashed `make_vjp` closure to return ∂L/∂ε_flat."""

    @staticmethod
    def forward(ctx, eps_flat, solver):
        eps_np = eps_flat.detach().cpu().numpy().astype(np.float64)
        # One forward solve; vjp_fun closes over it for a single adjoint pass.
        vjp_fun, P = make_vjp(solver.eps_to_P)(eps_np)
        ctx.vjp_fun = vjp_fun
        ctx.in_meta = (eps_flat.dtype, eps_flat.device)
        return torch.as_tensor(np.asarray(P), dtype=eps_flat.dtype, device=eps_flat.device)

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.detach().cpu().numpy().astype(np.float64)
        grad_eps = ctx.vjp_fun(g)                       # one adjoint solve
        dtype, device = ctx.in_meta
        return torch.as_tensor(np.asarray(grad_eps), dtype=dtype, device=device), None


class FDFDPowerModel(nn.Module):
    """Differentiable `ε → P` power model backed by the true ceviche FDFD,
    drop-in for ESPolicy.train_phase2_grad's `power_model`.

        eps:  (B, N_x, N_y)  permittivity states
        P:    (B, n_recv)    receiver powers, differentiable w.r.t. eps

    FDFD has no batch dimension, so samples are solved one at a time (the B·T
    solves per gradient step are the cost of unbiased gradients). Carries no
    trainable parameters — it is pure physics.
    """

    def __init__(self, env):
        super().__init__()
        self.solver = FDFDAdjointSolver(env)
        self.state_shape = self.solver.state_shape
        self.n_recv = self.solver.n_recv

    def forward(self, eps):
        single = (eps.dim() == 2)
        if single:
            eps = eps.unsqueeze(0)
        outs = [_FDFDPowerFn.apply(eps[b].reshape(-1), self.solver)
                for b in range(eps.shape[0])]
        P = torch.stack(outs, dim=0)
        return P[0] if single else P


# =============================================================================
# Single-sample solve primitives + a map-parallel power model
# =============================================================================
# The two primitives below are the unit of work fanned across Modal workers:
# one FDFD forward solve, and one forward+adjoint (VJP) solve. Both the local
# test maps and the Modal workers (train_phase2_grad_modal.py) call these, so
# the per-sample physics lives in exactly one place.

def fdfd_solve_forward(solver, eps_np):
    """One FDFD forward solve: ε (N_x, N_y) → P (n_recv,)."""
    from autograd import make_vjp
    _, P = make_vjp(solver.eps_to_P)(np.asarray(eps_np, np.float64).reshape(-1))
    return np.asarray(P)


def fdfd_solve_vjp(solver, eps_np, g_np):
    """One adjoint (VJP) solve: given cotangent g = ∂L/∂P, return ∂L/∂ε
    (N_x, N_y). Re-runs the forward internally (make_vjp), so ~2 solves."""
    from autograd import make_vjp
    eps_np = np.asarray(eps_np, np.float64)
    vjp_fun, _ = make_vjp(solver.eps_to_P)(eps_np.reshape(-1))
    grad_flat = vjp_fun(np.asarray(g_np, np.float64))
    return np.asarray(grad_flat).reshape(eps_np.shape)


class _RemoteFDFDFn(torch.autograd.Function):
    """Batched FDFD power via injected forward/VJP MAPS. Splits the autograd
    primitive so the solves run wherever the map sends them (local list, or a
    Modal .map() across workers) while the autograd graph stays on the driver."""

    @staticmethod
    def forward(ctx, eps_batch, model):
        eps_np = eps_batch.detach().cpu().numpy().astype(np.float64)
        P_list = model._forward_map([eps_np[b] for b in range(eps_np.shape[0])])
        ctx.model = model
        ctx.save_for_backward(eps_batch)
        return torch.as_tensor(np.stack(P_list), dtype=eps_batch.dtype, device=eps_batch.device)

    @staticmethod
    def backward(ctx, grad_P):
        (eps_batch,) = ctx.saved_tensors
        eps_np = eps_batch.detach().cpu().numpy().astype(np.float64)
        g_np = grad_P.detach().cpu().numpy().astype(np.float64)
        grad_list = ctx.model._vjp_map(
            [(eps_np[b], g_np[b]) for b in range(eps_np.shape[0])])
        grad_eps = torch.as_tensor(
            np.stack(grad_list), dtype=eps_batch.dtype, device=eps_batch.device)
        return grad_eps, None


class RemoteFDFDPowerModel(nn.Module):
    """Differentiable ε→P backed by PLUGGABLE solve maps, so the per-sample FDFD
    forward + adjoint solves can be fanned across Modal workers (B-way parallel
    per BPTT step — the maximum parallelization a sequential BPTT allows).
    Drop-in for train_phase2_grad's `power_model`; identical math to
    FDFDPowerModel — only WHERE the solves run differs.

        forward_map: callable  list[eps_np (N_x,N_y)]    -> list[P_np (n_recv,)]
        vjp_map:     callable  list[(eps_np, g_np)]       -> list[grad_np (N_x,N_y)]

    Back the maps with `local_solve_maps(env)` (single process, for testing /
    one machine) or a Modal function's `.map()` (production).
    """

    def __init__(self, forward_map, vjp_map):
        super().__init__()
        self._forward_map = forward_map
        self._vjp_map = vjp_map

    def forward(self, eps):
        single = (eps.dim() == 2)
        if single:
            eps = eps.unsqueeze(0)
        P = _RemoteFDFDFn.apply(eps, self)
        return P[0] if single else P


def local_solve_maps(env):
    """(forward_map, vjp_map) backed by one in-process FDFDAdjointSolver — for
    testing RemoteFDFDPowerModel without Modal, or single-machine runs."""
    solver = FDFDAdjointSolver(env)
    fwd = lambda eps_list: [fdfd_solve_forward(solver, e) for e in eps_list]
    vjp = lambda pairs: [fdfd_solve_vjp(solver, e, g) for (e, g) in pairs]
    return fwd, vjp


if __name__ == "__main__":
    # Validate: (1) forward matches get_receiver_powers; (2) adjoint matches
    # finite differences — on the tiny 3×3 env from the test suite.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from test_phase2_tiny import _build_tiny_env
    from algorithms.agents.es_agent import get_receiver_powers, apply_eps_to_canvas

    env = _build_tiny_env()
    N_x, N_y = env.grid.num_rods_x, env.grid.num_rods_y
    rng = np.random.default_rng(0)
    eps0 = rng.uniform(-1.0, 1.0, (N_x, N_y)).astype(np.float64)

    model = FDFDPowerModel(env)

    # (1) forward agreement
    apply_eps_to_canvas(env, eps0)
    P_ref = get_receiver_powers(env)
    P_bridge = model(torch.as_tensor(eps0)).detach().numpy()
    rel = np.max(np.abs(P_bridge - P_ref) / (np.abs(P_ref) + 1e-12))
    print(f"forward max rel err vs get_receiver_powers: {rel:.2e}")

    # (2) adjoint vs finite difference of Σ_j w_j P_j
    w = rng.uniform(size=model.n_recv)
    eps_t = torch.tensor(eps0, requires_grad=True)
    P = model(eps_t)
    (torch.as_tensor(w) @ P).backward()
    g_adjoint = eps_t.grad.numpy().copy()

    h = 1e-4
    g_fd = np.zeros_like(eps0)
    for i in range(N_x):
        for j in range(N_y):
            ep = eps0.copy(); ep[i, j] += h
            em = eps0.copy(); em[i, j] -= h
            Pp = model(torch.as_tensor(ep)).detach().numpy()
            Pm = model(torch.as_tensor(em)).detach().numpy()
            g_fd[i, j] = (w @ (Pp - Pm)) / (2 * h)
    gerr = np.max(np.abs(g_adjoint - g_fd)) / (np.max(np.abs(g_fd)) + 1e-12)
    print(f"adjoint vs finite-difference max rel err: {gerr:.2e}")

    # (3) RemoteFDFDPowerModel (local maps) must match FDFDPowerModel exactly —
    # this validates the map-split autograd.Function; only the Modal .map()
    # transport is then untested.
    fwd_map, vjp_map = local_solve_maps(env)
    rmodel = RemoteFDFDPowerModel(fwd_map, vjp_map)
    epsb = torch.tensor(np.stack([eps0, -eps0]), requires_grad=True)   # batch of 2
    epsb_ref = torch.tensor(np.stack([eps0, -eps0]), requires_grad=True)
    Pr = rmodel(epsb)
    Pf = model(epsb_ref)
    fwd_match = float((Pr - Pf).abs().max())
    (torch.as_tensor(w) * Pr).sum().backward()
    (torch.as_tensor(w) * Pf).sum().backward()
    grad_match = float((epsb.grad - epsb_ref.grad).abs().max())
    print(f"Remote vs FDFDPowerModel  forward max|Δ|={fwd_match:.2e}  grad max|Δ|={grad_match:.2e}")

    ok = rel < 1e-6 and gerr < 1e-3 and fwd_match < 1e-9 and grad_match < 1e-9
    print("OK" if ok else "MISMATCH — investigate")
