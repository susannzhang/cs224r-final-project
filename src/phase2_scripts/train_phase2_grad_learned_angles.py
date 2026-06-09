"""Gradient (BPTT) refinement of a Phase 2 policy on the Phase-1 LEARNED ANGLES.

Curriculum step 2 (after distillation): instead of generalizing to all 30
receivers in one shot, restrict the goal set to the Phase-1 training angles —
`goal_indices = sorted(memory_bank.keys())` — and grad-refine the policy to
retarget among them. Rollouts start from one converged Phase-1 config and steer
toward another (the memory bank seeds both `goal_indices` and the warm-start
`converged_indices`), so the policy reproduces-and-improves what Phase 1 found
before any extrapolation to novel angles.

Differentiable env model (`power_model`):
  --power-model fdfd       true ceviche FDFD adjoint (unbiased, ~2 solves/step)
  --power-model surrogate  learned M(ε)→P PNetwork (fast, biased by M's error)

Typical pipeline:
  1. python pretrain_phase2_from_buffer.py --policy-arch pinn ... \\
         --out pretrain/policy_buffer_traj_pinn.pt
  2. python train_phase2_grad_learned_angles.py \\
         --policy pretrain/policy_buffer_traj_pinn.pt \\
         --memory-bank phase1-uniform-init-output \\
         --reward-mode source_norm --n-iter 300 --T 20 --bptt-truncate 4 \\
         --out pretrain/policy_grad_learned.pt

Then widen `goal_indices` to novel receivers for the generalization phase.
"""

import argparse
import sys as _sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_DBS = Path(__file__).resolve().parent           # dynamic_beam_steering/
_PROJ = _DBS.parent                               # cs153 repo root
for _p in (_PROJ, _DBS):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from geometry import (create_design_region, create_environment, create_grid,
                      create_receiver, create_source)
from simulation import initialize_environment
from algorithms.policies.es_policy import ESPolicy, ESPolicyConfig, Phase2GradConfig
from algorithms.infrastructure.fdfd_adjoint import FDFDPowerModel
from algorithms.agents.es_agent import compute_source_power


# --- canonical pm_setup env (matches train_phase1.build_env) ----------------
def build_env():
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0,
                                   margin_cells=20)
    grid = create_grid(num_rods_x=10, num_rods_y=10,
                       radius=0.01, distance=0.002, rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
    receivers = []
    for i in range(1, 11):
        receivers.append(create_receiver(index=i, length=0.02, side='bottom', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=10 + i, length=0.02, side='right', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=20 + i, length=0.02, side='top', rod_index=11 - i))
    env = create_environment(design_region=region, grid=grid,
                             sources=[source], receivers=receivers)
    initialize_environment(env)
    return env


def load_memory_bank(d: Path) -> dict:
    bank = {}
    for td in sorted(d.glob("target_*")):
        if not td.is_dir():
            continue
        try:
            idx = int(td.name.split("_")[-1])
        except ValueError:
            continue
        p = td / "eps_star.npy"
        if p.exists():
            bank[idx] = np.load(p).astype(np.float32)
    if not bank:
        raise FileNotFoundError(f"No target_*/eps_star.npy under {d}")
    return bank


class SurrogatePowerModel(nn.Module):
    """Differentiable ε→P from a trained M(ε) PNetwork checkpoint. The net
    predicts normalized log1p(P); we de-normalize (expm1) to raw P, keeping the
    graph intact so gradients flow for BPTT."""
    def __init__(self, ckpt_path, device="cpu"):
        super().__init__()
        from train_V_network import PNetwork
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.net = PNetwork(ckpt["state_shape"], ckpt["n_receivers"],
                            ckpt["hidden_dim"], ckpt["n_hidden_layers"])
        self.net.load_state_dict(ckpt["model_state_dict"])
        self.register_buffer("log_mean", torch.as_tensor(np.asarray(ckpt["log_mean"]), dtype=torch.float32))
        self.register_buffer("log_std", torch.as_tensor(np.asarray(ckpt["log_std"]), dtype=torch.float32))
        self.n_recv = ckpt["n_receivers"]

    def forward(self, eps):
        log_P = self.net(eps) * self.log_std + self.log_mean
        return torch.expm1(log_P).clamp(min=0.0)


def parse_args():
    p = argparse.ArgumentParser(description="Grad-refine Phase 2 on Phase-1 learned angles")
    p.add_argument("--memory-bank", type=Path, required=True)
    p.add_argument("--policy", type=Path, default=None,
                   help="ESPolicy checkpoint to refine (e.g. the distilled PINN). "
                        "If omitted, starts a fresh policy (--policy-arch).")
    p.add_argument("--policy-arch", choices=["cnn", "mlp", "pinn"], default="pinn",
                   dest="policy_arch", help="Arch when --policy is omitted.")
    p.add_argument("--out", type=Path, default=Path("pretrain/policy_grad_learned.pt"))

    # differentiable env model
    p.add_argument("--power-model", choices=["fdfd", "surrogate"], default="fdfd",
                   dest="power_model")
    p.add_argument("--surrogate", type=Path, default=None,
                   help="M(ε)→P PNetwork checkpoint (required for --power-model surrogate).")

    # train_phase2_grad knobs
    p.add_argument("--reward-mode", choices=["reach_hold", "source_norm"],
                   default="source_norm", dest="reward_mode")
    p.add_argument("--p-source", type=float, default=None, dest="p_source",
                   help="source_norm denominator; auto-calibrated (free-space) if omitted.")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--bptt-truncate", type=int, default=4, dest="bptt_truncate")
    p.add_argument("--one-step", action="store_true", dest="one_step")
    p.add_argument("--batch-tasks", type=int, default=8, dest="batch_tasks")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--n-iter", type=int, default=300, dest="n_iter")
    p.add_argument("--grad-clip", type=float, default=1.0, dest="grad_clip")
    p.add_argument("--p-rand", type=float, default=0.2, dest="p_rand",
                   help="Prob. of random cold start; default 0.2 favours true "
                        "anchor→anchor retargeting among learned angles.")
    p.add_argument("--physics-loss-weight", type=float, default=0.1,
                   dest="physics_loss_weight")
    p.add_argument("--log-every", type=int, default=10, dest="log_every")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-local", action="store_true", dest="allow_local",
                   help="Permit LOCAL training (dev/smoke only). Production runs "
                        "must train on Modal — see the note in main().")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.allow_local:
        raise SystemExit(
            "Refusing to train locally. Production Phase 2 training runs on Modal "
            "with maximum parallelization via the ES parallel driver:\n\n"
            "    python spawn_phase2_parallel.py --policy-arch pinn \\\n"
            "        --reward-mode source_norm --gamma 0.95 ...\n\n"
            "That driver fans K closed-loop FDFD rollouts across Modal containers "
            "every iteration and now trains any arch (cnn/mlp/pinn) and reward mode "
            "through the SHARED phase2_rollout. The BPTT-gradient path here is "
            "inherently sequential (it backprops through a trajectory), so it is "
            "kept as a local dev/fine-tuning tool only. Re-run with --allow-local "
            "for a small local smoke test, or ask for a Modal-parallel grad driver.")

    print("Building pm_setup env (10×10)...")
    env = build_env()
    n_recv = len(env.receivers)

    memory_bank = load_memory_bank(args.memory_bank)
    goal_indices = sorted(memory_bank.keys())          # <-- LEARNED ANGLES ONLY
    print(f"Memory bank / learned angles: {goal_indices}  ({len(goal_indices)} of {n_recv})")

    # Policy: refine the distilled checkpoint, or start fresh.
    if args.policy is not None:
        policy = ESPolicy.load(args.policy, device=args.device)
        print(f"Loaded policy {args.policy} (arch={getattr(policy.config,'policy_arch','?')}, "
              f"state_shape={policy.state_shape})")
    else:
        ss = (env.grid.num_rods_x, env.grid.num_rods_y)
        policy = ESPolicy(ss, ESPolicyConfig(policy_arch=args.policy_arch, n_goals=n_recv,
                                             tanh_output_scale=0.25, device=args.device,
                                             seed=args.seed))
        print(f"Fresh {args.policy_arch} policy, state_shape={ss}")

    # Differentiable power model.
    if args.power_model == "fdfd":
        print("Power model: true ceviche FDFD adjoint (unbiased).")
        power_model = FDFDPowerModel(env).to(args.device)
    else:
        if args.surrogate is None:
            raise SystemExit("--power-model surrogate requires --surrogate <ckpt>")
        print(f"Power model: surrogate {args.surrogate} (biased by M's error).")
        power_model = SurrogatePowerModel(args.surrogate, device=args.device)

    # source_norm denominator: fixed, un-gameable; calibrate once on free space.
    p_source = args.p_source
    if args.reward_mode == "source_norm" and p_source is None:
        p_source = compute_source_power(env)
        print(f"Calibrated p_source (free-space total) = {p_source:.4g}")

    cfg = Phase2GradConfig(
        reward_mode=args.reward_mode,
        p_source=(p_source if p_source is not None else 1.0),
        gamma=args.gamma, T=args.T, bptt_truncate=args.bptt_truncate,
        one_step=args.one_step, batch_tasks=args.batch_tasks, lr=args.lr,
        n_iter=args.n_iter, grad_clip=args.grad_clip, p_rand=args.p_rand,
        physics_loss_weight=args.physics_loss_weight, log_every=args.log_every,
        seed=args.seed,
    )
    print(f"\nGrad-refine on learned angles: reward={cfg.reward_mode} gamma={cfg.gamma} "
          f"T={cfg.T} trunc={cfg.bptt_truncate} one_step={cfg.one_step} "
          f"batch={cfg.batch_tasks} lr={cfg.lr} n_iter={cfg.n_iter}")

    def _log(rec):
        extra = f" phys={rec['phys_residual']:.3g}" if "phys_residual" in rec else ""
        print(f"  iter {rec['iteration']:>4}: reward_mean={rec['reward_mean']:.4f} "
              f"loss={rec['loss']:.4f}{extra}")

    out = policy.train_phase2_grad(
        power_model, memory_bank, goal_indices, cfg,
        converged_indices=goal_indices,        # warm-start θ_prev from learned angles too
        on_iteration=_log,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.out)
    print(f"\nRefined policy → {args.out.resolve()}")
    if out["history"]:
        h = out["history"]
        print(f"reward_mean: {h[0]['reward_mean']:.4f} → {h[-1]['reward_mean']:.4f}")


if __name__ == "__main__":
    main()
