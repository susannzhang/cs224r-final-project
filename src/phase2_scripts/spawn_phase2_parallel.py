"""Spawn the Phase 2 parallel-K driver against the deployed Modal app.

Workflow:
    1. modal deploy train_phase2_parallel_modal.py        (one-time)
    2. python spawn_phase2_parallel.py ...                 (queues one driver)
    3. modal run train_phase2_parallel_modal.py::collect \\
         --launch-id <id>                                  (pull results)

Architecture: the driver runs on Modal as a single 24h container that
internally dispatches K rollouts in parallel each ES iter via
rollout_one.map(). At K=20 this gives ~20× more iters per unit wall time
than the sequential train_phase2_modal.py.

Usage:
    python spawn_phase2_parallel.py \\
        --memory-bank phase1-uniform-init-output \\
        --goal-indices 1,4,7,10,13,16,19,22,25,28 \\
        --K 20 --T 50 --N-iter 100 --alpha-2 0.02
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import modal
import numpy as np

APP_NAME = os.environ.get("PHASE2_APP_NAME", "cs224r-phase2-parallel")
DRIVER_FN = "driver"


def parse_args():
    p = argparse.ArgumentParser(
        description="Spawn the Phase 2 K-parallel Modal driver")

    # Inputs
    p.add_argument("--memory-bank", type=Path, required=True,
                   help="Phase 1 output dir with target_<NN>/eps_star.npy.")
    p.add_argument("--goal-indices", type=str, required=True, dest="goal_indices",
                   help='Phase 2 retargeting goals, e.g. "1,4,7,10,13,16,19,22,25,28".')
    p.add_argument("--converged-indices", type=str, default=None,
                   dest="converged_indices",
                   help="Override for the Phase 1 angle pool (default = memory_bank keys).")

    # Optional warm-start
    p.add_argument("--policy", type=Path, default=None,
                   help="Warm-start policy_awr.pt (optional).")

    # Fresh-policy config (used only when --policy is omitted; if --policy is
    # given, the architecture is loaded from that checkpoint's saved config).
    p.add_argument("--n-goals", type=int, default=30)
    p.add_argument("--policy-arch", choices=["cnn", "mlp", "pinn"], default="cnn",
                   dest="policy_arch",
                   help="Architecture for a fresh policy (ignored when --policy warm-starts).")
    p.add_argument("--hidden-dim", type=int, default=100)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--no-tanh", action="store_true")
    p.add_argument("--tanh-output-scale", type=float, default=0.25,
                   dest="tanh_output_scale",
                   help="Cap on |δ| per element (= scale·tanh). Default 0.25 — "
                        "produces graduated multi-step trajectories instead of "
                        "saturating jumps, leaving room for ES to refine "
                        "(σ-perturbations on tanh-saturated outputs are no-ops).")

    # ES outer loop
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--alpha-2", type=float, default=0.02, dest="alpha_2")
    p.add_argument("--N-iter", type=int, default=100, dest="N_iter")
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--eta", type=float, default=1e-2)
    p.add_argument("--p-rand", type=float, default=0.3, dest="p_rand")
    p.add_argument("--log-every", type=int, default=1, dest="log_every")
    p.add_argument("--seed", type=int, default=0)

    # Reward shaping. Default = "retarget": r_t = Q(s_t) − Q(s_{t-1}) with
    # Q(s) = P_target² / (P_total + EPS). Trajectory fitness telescopes to
    # Q(s_T) − Q(s_0); attenuated/center-trapped configs score NEGATIVE
    # because P_target² collapses faster than P_total. Replaces
    # "step_improvement" as the default after the center-localization
    # failure of the additive Q (see writeup/phase2_negative_result.tex).
    p.add_argument("--reward-mode", type=str, default="retarget",
                   choices=["retarget", "step_improvement", "absolute", "target_frac",
                            "reach_hold", "source_norm"],
                   dest="reward_mode",
                   help='"retarget" (default): r_t = ΔQ with Q = P_target² / P_total. '
                        '"step_improvement": r_t = ΔQ with additive Q. '
                        '"absolute": r_t = Q(s_t). "target_frac": r_t = scale·P_θ/ΣP. '
                        '"reach_hold": infinite-horizon P_θ/ΣP + plateau bonus (set --gamma<1). '
                        '"source_norm": P_θ/p_source, fixed un-gameable denominator '
                        '(p_source auto-calibrated on Modal).')
    p.add_argument("--target-frac-scale", type=float, default=1.0e+5,
                   dest="target_frac_scale",
                   help="Only used when --reward-mode=target_frac.")
    # Infinite-horizon / reach-and-hold knobs (used by reach_hold; γ also
    # discounts the fitness for every mode).
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Discount on the MC-return fitness. <1 for reach_hold/source_norm.")
    p.add_argument("--p-source", type=float, default=0.0, dest="p_source",
                   help="source_norm denominator. 0 (default) → auto-calibrate "
                        "(free-space total power) on Modal in the driver.")
    p.add_argument("--hold-threshold", type=float, default=0.9, dest="hold_threshold")
    p.add_argument("--hold-bonus", type=float, default=1.0, dest="hold_bonus")
    p.add_argument("--w-crosstalk", type=float, default=0.3, dest="w_crosstalk")
    p.add_argument("--w-loss", type=float, default=1e-4, dest="w_loss",
                   help="In retarget mode: soft penalty on absolute P_loss "
                        "(telescoped to endpoint diff), defends against the "
                        "back-redirect-toward-source-wall failure mode. "
                        "Default 1e-4. In step_improvement mode: ignored "
                        "(loss term hardcoded to 0 in rollout). In absolute "
                        "mode: Phase 1's ΔP_loss penalty; set to 0 for Phase 2.")
    p.add_argument("--w-energy", type=float, default=0.0, dest="w_energy",
                   help="In retarget mode: penalty on absolute E_rods "
                        "(telescoped). Default 0. Set to 1e-2-ish only if "
                        "rod-voltage waste becomes a deployment concern.")

    # M (FDFD surrogate) pre-filter + online training
    p.add_argument("--m-surrogate", type=str, default=None,
                   dest="m_surrogate",
                   help="Path to FDFD surrogate M(ε)→P[30] checkpoint "
                        "(e.g. pretrain/M_fdfd_surrogate.pt). If given, driver "
                        "M-ghost-rollouts both ±ξ for each ES noise vector and "
                        "saves only the sign whose trajectory ends with M-argmax "
                        "matching the goal. Each iter also trains M on the "
                        "FDFD-rollout-derived (state, P) pairs (DAGGER-style).")
    p.add_argument("--m-filter-K", type=int, default=10, dest="m_filter_K",
                   help="When M-filter active: number of approved candidates "
                        "to gather per ES iter (no mirroring). Default 10.")
    p.add_argument("--m-filter-max-attempts", type=int, default=1000,
                   dest="m_filter_max_attempts",
                   help="Cap on ξ resamples per iter when M-filter active.")

    # Early-stop + checkpointing
    p.add_argument("--early-stop-patience", type=int, default=0,
                   dest="early_stop_patience",
                   help="Stop if best fitness doesn't improve for this many "
                        "consecutive iters. 0 = disabled. Typical: 20-30.")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                   dest="early_stop_min_delta",
                   help="Minimum fitness improvement to count as 'improving'. "
                        "Default 0.0 (any improvement counts).")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   dest="checkpoint_every",
                   help="Save policy + history to Volume every N iters. "
                        "0 = disable (final save only).")

    # Wandb
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="cs224r-phase2-parallel")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--launch-id", type=str, default="")
    p.add_argument("--out-dir", type=str, default="phase2_parallel_output")
    return p.parse_args()


def load_memory_bank(memory_bank_dir: Path) -> dict:
    """Load every target_<NN>/eps_star.npy found in the directory."""
    memory_bank = {}
    for target_dir in sorted(memory_bank_dir.glob("target_*")):
        if not target_dir.is_dir():
            continue
        try:
            idx = int(target_dir.name.split("_")[-1])
        except ValueError:
            continue
        eps_star_path = target_dir / "eps_star.npy"
        if eps_star_path.exists():
            memory_bank[idx] = np.load(eps_star_path)
    if not memory_bank:
        raise FileNotFoundError(
            f"No target_<NN>/eps_star.npy found under {memory_bank_dir}.")
    return memory_bank


def main():
    args = parse_args()
    goal_indices = [int(x) for x in args.goal_indices.split(",")]
    converged_indices = (
        [int(x) for x in args.converged_indices.split(",")]
        if args.converged_indices else None
    )

    launch_id = args.launch_id or (
        "phase2-parallel-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # ----- Memory bank --------------------------------------------
    memory_bank = load_memory_bank(args.memory_bank)
    print(f"Loaded memory bank: {len(memory_bank)} entries "
          f"(angles {sorted(memory_bank.keys())}) from {args.memory_bank}")
    effective_converged = (converged_indices if converged_indices is not None
                           else sorted(memory_bank.keys()))
    print(f"  converged_indices: {effective_converged}")
    print(f"  goal_indices:      {goal_indices}")
    print(f"  reward P_total:    sum over all 30 receivers (driver-side)")

    # ----- Warm-start policy (optional) ---------------------------
    policy_bytes = None
    policy_config_kwargs = None
    if args.policy is not None:
        if not args.policy.exists():
            raise FileNotFoundError(f"Policy not found: {args.policy}")
        policy_bytes = args.policy.read_bytes()
        print(f"Warm-start policy: {args.policy} ({len(policy_bytes)/1024:.1f} KB)")
    else:
        first_eps = next(iter(memory_bank.values()))
        state_shape = tuple(int(x) for x in first_eps.shape)
        policy_config_kwargs = {
            "state_shape": state_shape,
            "n_goals": args.n_goals,
            "policy_arch": args.policy_arch,
            "hidden_dim": args.hidden_dim,
            "n_hidden_layers": args.n_hidden_layers,
            "tanh_output": not args.no_tanh,
            "tanh_output_scale": args.tanh_output_scale,
        }
        print(f"No --policy provided. Fresh ESPolicy: state_shape={state_shape}  "
              f"n_goals={args.n_goals}  "
              f"hidden_dim={args.hidden_dim}×{args.n_hidden_layers}  "
              f"tanh_scale={args.tanh_output_scale}  "
              f"tanh={not args.no_tanh}")

    # ----- Wandb config -------------------------------------------
    wandb_cfg = None
    if not args.no_wandb:
        wandb_cfg = {
            "project": args.wandb_project,
            "entity": args.wandb_entity or None,
        }

    # ----- Build config + payload ---------------------------------
    config_kwargs = dict(
        K=args.K, sigma=args.sigma, alpha_2=args.alpha_2,
        N_iter=args.N_iter, T=args.T, eta=args.eta,
        p_rand=args.p_rand, log_every=args.log_every, seed=args.seed,
        w_crosstalk=args.w_crosstalk, w_loss=args.w_loss, w_energy=args.w_energy,
        reward_mode=args.reward_mode, target_frac_scale=args.target_frac_scale,
        gamma=args.gamma, hold_threshold=args.hold_threshold, hold_bonus=args.hold_bonus,
    )
    # p_source==0 is the "auto-calibrate on Modal" sentinel; only pass an explicit
    # value through (Phase2Config rejects p_source==0 for source_norm, and the
    # driver fills it in via compute_source_power before building the config).
    if args.p_source != 0.0:
        config_kwargs["p_source"] = args.p_source

    print()
    print(f"Phase 2 (parallel) on Modal app={APP_NAME!r}")
    print(f"  launch_id: {launch_id}")
    print(f"  K={args.K}  σ={args.sigma}  α_2={args.alpha_2}  "
          f"N_iter={args.N_iter}  T={args.T}  p_rand={args.p_rand}")
    print(f"  total FDFDs (upper bound): {args.K * args.T * args.N_iter:,}")
    print(f"  per-iter wall (target): ~{args.T * 10 / 60:.1f} min "
          f"(dominated by slowest of K={args.K} parallel rollouts)")
    print(f"  expected total wall: ~{args.N_iter * args.T * 10 / 3600:.1f} hr")
    print()

    # M-surrogate inside Modal image is at /root/app/<relative path>.
    # Seed data (mean_states_P.npz) lives under phase1-uniform-init-output/,
    # which is ignored by the image build → ship its bytes via payload.
    m_surrogate_path = None
    m_training_data_bytes = None
    if args.m_surrogate is not None:
        m_surrogate_path = "/root/app/" + str(Path(args.m_surrogate))
        print(f"M-filter active: surrogate at container path {m_surrogate_path}")
        print(f"  filter K={args.m_filter_K}  max_attempts={args.m_filter_max_attempts}")
        seed_path = args.memory_bank / "mean_states_P.npz"
        if seed_path.exists():
            m_training_data_bytes = seed_path.read_bytes()
            print(f"  online-M seed data: {seed_path} "
                  f"({len(m_training_data_bytes)/1e6:.1f} MB)")
        else:
            print(f"  WARN: no seed data at {seed_path} — M will train only from "
                  f"new FDFD rollouts (slow start)")

    payload = {
        "launch_id": launch_id,
        "goal_indices": goal_indices,
        "converged_indices": converged_indices,
        "config_kwargs": config_kwargs,
        "policy_bytes": policy_bytes,
        "policy_config_kwargs": policy_config_kwargs,
        "memory_bank": {int(k): np.asarray(v) for k, v in memory_bank.items()},
        "wandb": wandb_cfg,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "checkpoint_every": args.checkpoint_every,
        "m_surrogate_path": m_surrogate_path,
        "m_filter_K": args.m_filter_K,
        "m_filter_max_attempts": args.m_filter_max_attempts,
        "m_training_data_bytes": m_training_data_bytes,
    }

    f = modal.Function.from_name(APP_NAME, DRIVER_FN)
    fc = f.spawn(payload)
    print(f"✓ driver spawned → {fc.object_id}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = {
        "launch_id": launch_id,
        "driver_function_call_id": fc.object_id,
        "config_kwargs": config_kwargs,
        "goal_indices": goal_indices,
        "converged_indices": converged_indices,
        "warm_start_policy": str(args.policy.resolve()) if args.policy else None,
        "policy_config_kwargs": policy_config_kwargs,
        "memory_bank_dir": str(args.memory_bank.resolve()),
        "architecture": "K-parallel via rollout_one.map()",
    }
    with open(out_dir / f"{launch_id}.spawn.json", "w") as fh:
        json.dump(ledger, fh, indent=2)
    print(f"Ledger → {(out_dir / f'{launch_id}.spawn.json').resolve()}")

    print()
    print("Driver runs on Modal; close this terminal anytime.")
    print(f"Collect results when done:")
    print(f"  PHASE2_VOLUME_NAME={os.environ.get('PHASE2_VOLUME_NAME', 'cs224r-phase2-parallel-buffer')} \\")
    print(f"    python -m modal run train_phase2_parallel_modal.py::collect "
          f"--launch-id {launch_id} --out-dir {args.out_dir}")


if __name__ == "__main__":
    main()
