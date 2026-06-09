"""Spawn a Phase 2 ES training run against the deployed Modal app.

Workflow:
    1. modal deploy train_phase2_modal.py         (one-time / after worker edits)
    2. python spawn_phase2.py                     (queues one run)
    3. python -m modal run train_phase2_modal.py::collect \\
         --launch-id <id>                         (pull results later)

The spawn ships policy_awr.pt + memory_bank as bytes/dict in the payload
(~770 KB total), so no pre-upload to the Volume is needed. The deployed
function runs independently of this local script — close the terminal
anytime, the run continues.

Usage:
    python spawn_phase2.py \\
        --memory-bank phase1-uniform-init-output \\
        --goal-indices 1,4,7,10,13,16,19,22,25,28 \\
        --K 20 --T 10 --N-iter 20    # smoke (no warm-start)

    python spawn_phase2.py \\
        --policy phase1-uniform-init-output/policy_awr.pt \\
        --memory-bank phase1-uniform-init-output \\
        --goal-indices 1,4,7,10,13,16,19,22,25,28 \\
        --K 20 --T 50 --N-iter 100    # real run with warm-start
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import modal
import numpy as np

APP_NAME = os.environ.get("PHASE2_APP_NAME", "cs224r-phase2-es")
FUNCTION_NAME = "train_phase2_worker"


def parse_args():
    p = argparse.ArgumentParser(description="Spawn a Phase 2 Modal training run")

    # Inputs (memory bank + goal_indices required)
    p.add_argument("--memory-bank", type=Path, required=True,
                   help="Phase 1 output dir with target_<NN>/eps_star.npy. "
                        "All target_<NN>/eps_star.npy files become Phase 1's "
                        "converged_indices (warm-start θ_prev pool).")
    p.add_argument("--goal-indices", type=str, required=True, dest="goal_indices",
                   help="Phase 2 retargeting goals (θ_k sampled from here). "
                        'Comma-separated, e.g. "1,4,7,10,13,16,19,22,25,28".')
    p.add_argument("--converged-indices", type=str, default=None,
                   dest="converged_indices",
                   help="Optional override for the Phase 1 angles used as "
                        "warm-start θ_prev pool. Defaults to the memory bank's "
                        "actual keys.")

    # Optional warm-start
    p.add_argument("--policy", type=Path, default=None,
                   help="Optional warm-start policy .pt (e.g. policy_awr.pt). "
                        "If omitted, Phase 2 starts from a randomly-initialized "
                        "ESPolicy built from --hidden-dim / --n-hidden-layers / "
                        "--n-goals and the state_shape inferred from the memory bank.")

    # Optional buffer seed
    p.add_argument("--buffer", type=Path, default=None,
                   help="Optional Phase 1 replay_buffer.pkl to seed Phase 2.")

    # Fresh-policy config (used only when --policy is omitted)
    p.add_argument("--n-goals", type=int, default=30,
                   help="Total angles for sin/cos encoding modulus (default 30).")
    p.add_argument("--policy-arch", choices=["cnn", "mlp", "pinn"], default="cnn",
                   dest="policy_arch",
                   help="Architecture for a fresh policy (ignored when --policy warm-starts).")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--no-tanh", action="store_true",
                   help="Disable tanh output squash on the fresh policy.")

    # Phase 2 ES config
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--alpha-2", type=float, default=0.005, dest="alpha_2")
    p.add_argument("--N-iter", type=int, default=20, dest="N_iter")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--eta", type=float, default=1e-2)
    p.add_argument("--p-rand", type=float, default=0.3, dest="p_rand")
    p.add_argument("--log-every", type=int, default=1, dest="log_every")
    p.add_argument("--seed", type=int, default=0)

    # Reward weights
    p.add_argument("--w-crosstalk", type=float, default=0.3, dest="w_crosstalk")
    p.add_argument("--w-loss", type=float, default=1e-3, dest="w_loss")
    p.add_argument("--w-energy", type=float, default=0.1, dest="w_energy")

    # Wandb
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="cs224r-phase2")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--launch-id", type=str, default="",
                   help="Identifier for this run. Default = timestamp.")

    # Local spawn ledger
    p.add_argument("--out-dir", type=str, default="phase2_output")
    return p.parse_args()


def load_memory_bank(memory_bank_dir: Path) -> dict:
    """Load every target_<NN>/eps_star.npy found in the directory.

    Returns a dict keyed by the actual target index Phase 1 trained on
    (e.g. {0, 3, 6, ..., 27} for the Phase 1 random/uniform-init runs).
    These keys become the default converged_indices passed to Phase 2 and
    need not match Phase 2's goal_indices — the whole point of the held-out
    experiment is that Phase 2 trains on different angles than Phase 1.
    """
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
            f"No target_<NN>/eps_star.npy found under {memory_bank_dir}."
        )
    return memory_bank


def main():
    args = parse_args()
    goal_indices = [int(x) for x in args.goal_indices.split(",")]
    converged_indices = (
        [int(x) for x in args.converged_indices.split(",")]
        if args.converged_indices else None
    )

    launch_id = args.launch_id or (
        "phase2-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # ----- Validate / read local inputs ------------------------------
    memory_bank = load_memory_bank(args.memory_bank)
    print(f"Loaded memory bank: {len(memory_bank)} entries "
          f"(angles {sorted(memory_bank.keys())}) from {args.memory_bank}")
    # Resolve converged_indices: explicit CLI override > memory_bank keys.
    effective_converged = (converged_indices
                           if converged_indices is not None
                           else sorted(memory_bank.keys()))
    print(f"  converged_indices (warm-start θ_prev pool): {effective_converged}")
    print(f"  goal_indices (Phase 2 retargeting targets): {goal_indices}")
    union = sorted(set(effective_converged) | set(goal_indices))
    print(f"  logged_reward_indices (crosstalk + multi-goal logging): "
          f"{len(union)} angles → {union}")
    # Sanity check: goal_indices and converged_indices can be disjoint (the
    # held-out-set experiment); warn if they overlap unexpectedly.
    overlap = set(effective_converged) & set(goal_indices)
    if overlap:
        print(f"  ⚠ {len(overlap)} converged_indices overlap with "
              f"goal_indices: {sorted(overlap)}. Phase 2 will still draw "
              f"θ_prev from converged_indices \\ {{θ_target}} per the design.")

    # Warm-start policy (optional). When omitted, the worker constructs a
    # fresh ESPolicy from policy_config_kwargs.
    policy_bytes = None
    policy_config_kwargs = None
    if args.policy is not None:
        if not args.policy.exists():
            raise FileNotFoundError(f"Policy not found: {args.policy}")
        policy_bytes = args.policy.read_bytes()
        print(f"Loaded warm-start policy from {args.policy} "
              f"({len(policy_bytes) / 1024:.1f} KB)")
    else:
        # Infer state_shape from the memory bank (any entry will do — all
        # ε* configs share the same shape).
        first_eps = next(iter(memory_bank.values()))
        state_shape = tuple(int(x) for x in first_eps.shape)
        policy_config_kwargs = {
            "state_shape": state_shape,
            "n_goals": args.n_goals,
            "policy_arch": args.policy_arch,
            "hidden_dim": args.hidden_dim,
            "n_hidden_layers": args.n_hidden_layers,
            "tanh_output": not args.no_tanh,
        }
        print(f"No --policy provided. Worker will build a fresh ESPolicy: "
              f"state_shape={state_shape}  n_goals={args.n_goals}  "
              f"hidden_dim={args.hidden_dim} × {args.n_hidden_layers}  "
              f"tanh={not args.no_tanh}")

    buffer_bytes = None
    if args.buffer is not None:
        if not args.buffer.exists():
            raise FileNotFoundError(f"Buffer not found: {args.buffer}")
        size_mb = args.buffer.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            print(f"⚠ Buffer is {size_mb:.0f} MB. Shipping in spawn payload may "
                  f"be slow / hit Modal limits. Consider running without --buffer.")
        buffer_bytes = args.buffer.read_bytes()
        print(f"Loaded buffer seed from {args.buffer} ({size_mb:.1f} MB)")

    # ----- Wandb config ----------------------------------------------
    wandb_cfg = None
    if not args.no_wandb:
        wandb_cfg = {
            "project": args.wandb_project,
            "entity": args.wandb_entity or None,
        }

    # ----- Build payload + spawn -------------------------------------
    config_kwargs = dict(
        K=args.K, sigma=args.sigma, alpha_2=args.alpha_2,
        N_iter=args.N_iter, T=args.T, eta=args.eta,
        p_rand=args.p_rand, log_every=args.log_every, seed=args.seed,
        w_crosstalk=args.w_crosstalk, w_loss=args.w_loss, w_energy=args.w_energy,
    )

    print()
    print(f"Phase 2 on Modal (app={APP_NAME!r})")
    print(f"  launch_id: {launch_id}")
    print(f"  K={args.K}  σ={args.sigma}  α_2={args.alpha_2}  "
          f"N_iter={args.N_iter}  T={args.T}")
    n_logged = len(union)
    expected_fdfds = args.K * args.T * args.N_iter
    print(f"  expected FDFDs (upper bound): {expected_fdfds:,}")
    print(f"  expected new transitions per iter: "
          f"K·T·|logged_reward_indices| = {args.K * args.T * n_logged:,}")
    print()

    f = modal.Function.from_name(APP_NAME, FUNCTION_NAME)

    payload = {
        "launch_id": launch_id,
        "goal_indices": goal_indices,
        "converged_indices": converged_indices,
        "config_kwargs": config_kwargs,
        "policy_bytes": policy_bytes,
        "policy_config_kwargs": policy_config_kwargs,
        "memory_bank": {int(k): np.asarray(v) for k, v in memory_bank.items()},
        "buffer_bytes": buffer_bytes,
        "wandb": wandb_cfg,
    }

    fc = f.spawn(payload)
    print(f"✓ spawned → {fc.object_id}")

    # ----- Save ledger so collect can find this run ------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = {
        "launch_id": launch_id,
        "function_call_id": fc.object_id,
        "config_kwargs": config_kwargs,
        "goal_indices": goal_indices,
        "converged_indices": converged_indices,
        "warm_start_policy": str(args.policy.resolve()) if args.policy else None,
        "policy_config_kwargs": policy_config_kwargs,
        "memory_bank_dir": str(args.memory_bank.resolve()),
    }
    with open(out_dir / f"{launch_id}.spawn.json", "w") as fh:
        json.dump(ledger, fh, indent=2)

    print()
    print(f"Run queued on persistent app {APP_NAME!r}. Close this terminal anytime.")
    print(f"Ledger → {(out_dir / f'{launch_id}.spawn.json').resolve()}")
    print()
    print(f"When ready to collect results:")
    print(f"  PHASE2_VOLUME_NAME={os.environ.get('PHASE2_VOLUME_NAME', 'cs224r-phase2-buffer')} \\")
    print(f"    python -m modal run train_phase2_modal.py::collect "
          f"--launch-id {launch_id} --out-dir {args.out_dir}")


if __name__ == "__main__":
    main()
