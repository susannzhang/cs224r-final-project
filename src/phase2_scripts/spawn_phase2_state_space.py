"""Spawn the Phase 2 state-space-ES driver on Modal.

Workflow:
    1. modal deploy train_phase2_state_space_modal.py     (one-time)
    2. python spawn_phase2_state_space.py ...              (queues main_driver)
    3. modal run train_phase2_state_space_modal.py::collect \\
         --launch-id <id>                                  (download)

The spawned main_driver fans out one goal_driver per goal in `--goal-indices`
via Function.map(); each goal_driver runs ESStateSpacePolicy.run_one_goal
and dispatches its K_real FDFDs per iter via a nested fdfd_one.map(). So
total parallelism per outer iter ≈ N_goals × (1 + K_real) FDFD containers.

Usage:
    python spawn_phase2_state_space.py \\
        --memory-bank phase1-uniform-init-output \\
        --goal-indices 2,5,8,11,14,17,20,23,26,29 \\
        --m-surrogate pretrain/M_fdfd_surrogate_v2.pt \\
        --K-cand 200 --K-real 20 --sigma 0.05 --alpha 0.05 --N-iter 50 \\
        --init-mode interp
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import modal
import numpy as np

APP_NAME = os.environ.get("PHASE2_SS_APP_NAME", "cs224r-phase2-state-space")
DRIVER_FN = "main_driver"


def parse_args():
    p = argparse.ArgumentParser(
        description="Spawn the Phase 2 state-space-ES Modal driver")

    # --- Inputs ---------------------------------------------------------
    p.add_argument("--memory-bank", type=Path, required=True,
                   help="Phase 1 output dir with target_<NN>/eps_star.npy.")
    p.add_argument("--goal-indices", type=str, required=True, dest="goal_indices",
                   help='Goals to optimize, e.g. "2,5,8,11,14,17,20,23,26,29".')
    p.add_argument("--training-indices", type=str, default=None,
                   dest="training_indices",
                   help="Receiver indices for crosstalk + multi-goal logging "
                        "(default: 0..29, all 30 receivers).")
    p.add_argument("--init-mode", type=str, default="interp",
                   dest="init_mode", choices=["interp", "nearest", "uniform"],
                   help='Warm-start ε per goal. "interp" (default) = linear '
                        'interp between the two nearest Phase 1 anchors. '
                        '"nearest" = ε* of the nearest anchor verbatim. '
                        '"uniform" = U([-1, 1]) random.')

    # --- ES outer loop --------------------------------------------------
    p.add_argument("--K-cand", type=int, default=200, dest="K_cand",
                   help="Candidates evaluated by M each iter (filter pool).")
    p.add_argument("--K-real", type=int, default=20, dest="K_real",
                   help="Top survivors per iter that get FDFD'd.")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--N-iter", type=int, default=50, dest="N_iter")
    p.add_argument("--eta", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1, dest="log_every")

    # --- M filter -------------------------------------------------------
    p.add_argument("--m-surrogate", type=str, default=None,
                   dest="m_surrogate",
                   help="Path to M(ε) → P[30] surrogate checkpoint. Required "
                        "unless --filter-mode=off.")
    p.add_argument("--filter-mode", type=str, default="q",
                   dest="filter_mode", choices=["q", "argmax", "off"],
                   help='"q" (default): top K_real by predicted Q = P_θ² / P_total. '
                        '"argmax": only candidates whose M-argmax == goal, '
                        'tiebroken by Q (falls back to "q" if too few pass). '
                        '"off": disable M filter; K_real == K_cand FDFDs/iter.')
    p.add_argument("--online-m", action="store_true", dest="online_m",
                   help="Train M online on FDFD-derived (ε, P) pairs each "
                        "iter (DAGGER). Off by default — driver returns all "
                        "FDFD pairs for offline aggregation + retraining.")
    p.add_argument("--m-train-epochs-per-iter", type=int, default=5,
                   dest="m_train_epochs_per_iter")
    p.add_argument("--m-train-batch-size", type=int, default=256,
                   dest="m_train_batch_size")
    p.add_argument("--m-train-lr", type=float, default=1e-3,
                   dest="m_train_lr")
    p.add_argument("--m-train-weight-decay", type=float, default=1e-5,
                   dest="m_train_weight_decay")
    p.add_argument("--m-warmup-buffer", type=int, default=256,
                   dest="m_warmup_buffer")

    # --- Reward (Phase 1 defaults) --------------------------------------
    p.add_argument("--reward-mode", type=str, default="absolute",
                   dest="reward_mode",
                   choices=["absolute", "retarget", "target_frac"],
                   help='"absolute" (default, matches Phase 1): '
                        'r = P_θ − w_c·λ_c·P_others − w_loss·ΔP_loss − w_energy·ΔE_rods. '
                        '"retarget": r = P_θ² / P_total − w_loss·P_loss − w_energy·E_rods. '
                        '"target_frac": r = scale · P_θ / ΣP.')
    p.add_argument("--target-frac-scale", type=float, default=1e5,
                   dest="target_frac_scale")
    p.add_argument("--w-crosstalk", type=float, default=0.3, dest="w_crosstalk")
    p.add_argument("--w-loss", type=float, default=1e-3, dest="w_loss",
                   help="Phase 1 default 1e-3.")
    p.add_argument("--w-energy", type=float, default=0.1, dest="w_energy",
                   help="Phase 1 default 0.1.")

    # --- Checkpointing + wandb -----------------------------------------
    p.add_argument("--checkpoint-every", type=int, default=5,
                   dest="checkpoint_every")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str,
                   default="cs224r-phase2-state-space")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--launch-id", type=str, default="")
    p.add_argument("--out-dir", type=str, default="phase2_state_space_output")
    return p.parse_args()


def load_memory_bank(d: Path) -> dict:
    bank = {}
    for td in sorted(d.glob("target_*")):
        if not td.is_dir():
            continue
        try:
            idx = int(td.name.split("_")[-1])
        except ValueError:
            continue
        eps_path = td / "eps_star.npy"
        if eps_path.exists():
            bank[idx] = np.load(eps_path)
    if not bank:
        raise FileNotFoundError(
            f"No target_<NN>/eps_star.npy under {d}.")
    return bank


def main():
    args = parse_args()
    goal_indices = [int(x) for x in args.goal_indices.split(",")]
    training_indices = ([int(x) for x in args.training_indices.split(",")]
                        if args.training_indices else list(range(30)))

    if args.filter_mode != "off" and args.m_surrogate is None:
        raise ValueError(
            f"--filter-mode={args.filter_mode!r} requires --m-surrogate. "
            f"Pass --filter-mode=off to skip the M filter.")

    launch_id = args.launch_id or (
        "phase2-ss-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # --- Memory bank ----------------------------------------------------
    memory_bank = load_memory_bank(args.memory_bank)
    print(f"Loaded memory bank: {len(memory_bank)} entries "
          f"(angles {sorted(memory_bank.keys())}) from {args.memory_bank}")

    # --- M surrogate + seed data ---------------------------------------
    m_surrogate_path = None
    m_training_data_bytes = None
    if args.m_surrogate is not None:
        m_path_local = Path(args.m_surrogate)
        if not m_path_local.exists():
            raise FileNotFoundError(f"M surrogate not found: {m_path_local}")
        # Container path — image's add_local_dir mounts repo root at /root/app.
        m_surrogate_path = "/root/app/" + str(m_path_local)
        print(f"M surrogate: {m_path_local}  (container: {m_surrogate_path})")
        # Seed data for online-M only; loaded if --online-m.
        if args.online_m:
            seed_paths = [
                Path("pretrain/combined_eps_P.npz"),
                args.memory_bank / "mean_states_P.npz",
            ]
            for p in seed_paths:
                if p.exists():
                    m_training_data_bytes = p.read_bytes()
                    print(f"  online-M seed: {p} "
                          f"({len(m_training_data_bytes)/1e6:.1f} MB)")
                    break
            else:
                print(f"  WARN: no online-M seed data found at "
                      f"{[str(p) for p in seed_paths]}")

    # --- Config kwargs --------------------------------------------------
    config_kwargs = dict(
        K_cand=args.K_cand, K_real=args.K_real,
        sigma=args.sigma, alpha=args.alpha,
        N_iter=args.N_iter, eta=args.eta, seed=args.seed,
        log_every=args.log_every,
        filter_mode=args.filter_mode,
        online_m=args.online_m,
        m_train_epochs_per_iter=args.m_train_epochs_per_iter,
        m_train_batch_size=args.m_train_batch_size,
        m_train_lr=args.m_train_lr,
        m_train_weight_decay=args.m_train_weight_decay,
        m_warmup_buffer=args.m_warmup_buffer,
        reward_mode=args.reward_mode,
        target_frac_scale=args.target_frac_scale,
        w_crosstalk=args.w_crosstalk,
        w_loss=args.w_loss,
        w_energy=args.w_energy,
    )

    wandb_cfg = None if args.no_wandb else {
        "project": args.wandb_project,
        "entity": args.wandb_entity or None,
    }

    payload = {
        "launch_id": launch_id,
        "goal_indices": goal_indices,
        "training_indices": training_indices,
        "memory_bank": {int(k): np.asarray(v) for k, v in memory_bank.items()},
        "config_kwargs": config_kwargs,
        "init_mode": args.init_mode,
        "m_surrogate_path": m_surrogate_path,
        "m_training_data_bytes": m_training_data_bytes,
        "checkpoint_every": args.checkpoint_every,
        "wandb": wandb_cfg,
    }

    fdfd_per_iter_per_goal = 1 + args.K_real
    total_fdfd_upper = (len(goal_indices) * args.N_iter
                        * fdfd_per_iter_per_goal)
    print()
    print(f"Phase 2 state-space ES on Modal app={APP_NAME!r}")
    print(f"  launch_id:        {launch_id}")
    print(f"  goals:            {goal_indices}")
    print(f"  init_mode:        {args.init_mode}")
    print(f"  K_cand={args.K_cand}  K_real={args.K_real}  "
          f"σ={args.sigma}  α={args.alpha}  N_iter={args.N_iter}")
    print(f"  filter:           {args.filter_mode}  online_m={args.online_m}")
    print(f"  reward_mode:      {args.reward_mode}  "
          f"(w_c={args.w_crosstalk}, w_loss={args.w_loss}, "
          f"w_energy={args.w_energy})")
    print(f"  total FDFD (max): {total_fdfd_upper:,}  "
          f"({len(goal_indices)} goals × {args.N_iter} iters × "
          f"{fdfd_per_iter_per_goal} fdfd/iter)")
    print()

    f = modal.Function.from_name(APP_NAME, DRIVER_FN)
    fc = f.spawn(payload)
    print(f"✓ main_driver spawned → {fc.object_id}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = {
        "launch_id": launch_id,
        "main_driver_function_call_id": fc.object_id,
        "goal_indices": goal_indices,
        "init_mode": args.init_mode,
        "config_kwargs": config_kwargs,
        "m_surrogate": str(args.m_surrogate) if args.m_surrogate else None,
        "memory_bank_dir": str(args.memory_bank.resolve()),
        "architecture": ("per-goal state-space ES + M-filter on Modal; "
                         "main_driver → goal_driver.map() → fdfd_one.map()"),
    }
    with open(out_dir / f"{launch_id}.spawn.json", "w") as fh:
        json.dump(ledger, fh, indent=2)
    print(f"Ledger → {(out_dir / f'{launch_id}.spawn.json').resolve()}")

    print()
    print("Driver runs on Modal; close this terminal anytime.")
    print(f"Collect results when done:")
    print(f"  PHASE2_SS_VOLUME_NAME="
          f"{os.environ.get('PHASE2_SS_VOLUME_NAME', 'cs224r-phase2-state-space-buffer')} \\")
    print(f"    python -m modal run train_phase2_state_space_modal.py::collect "
          f"--launch-id {launch_id} --out-dir {args.out_dir}")


if __name__ == "__main__":
    main()
