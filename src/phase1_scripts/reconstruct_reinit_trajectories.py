"""Reconstruct per-iter ε_curr trajectories from Phase 1 reinit runs.

The reinit driver (train_phase1_reinit_modal.py) wrote a flat eps_buffer with
layout:
    eps_buffer[0]            = initial ε (pre-loop FDFD)
    eps_buffer[1 + t*21]     = baseline at start of iter t  (ε_curr after t-1's update)
    eps_buffer[1 + t*21 + k] = candidate k at iter t (k=1..20)
    eps_buffer[1 + N_iter*21] = final ε_mean (only present if loop ran to completion)

That layout lets us pull a clean trajectory:
    eps_traj[t] = eps_buffer[1 + t*21] for t in 0..(N_iter-1)
    plus the final post-update state if the run completed.

This script writes `goal_<NN>/eps_traj.npy` files in a fresh output directory,
matching the layout train_phase2_distill_closed_loop.py expects. Each run's
trajectories become drop-in distillation supervision: every consecutive
(ε_t, ε_{t+1}) pair across the entire reinit ES descent is one (state,
next_state, goal) tuple for π_φ to imitate.

Usage:
    # K=20 reinit (10 goals, σ=0.1 constant) → high-quality supervision
    python reconstruct_reinit_trajectories.py \\
        --state-pkl-dir /tmp/phase1_reinit_dl \\
        --out-dir distill_inputs/reinit_K20

    # σ-anneal (8 goals, σ=0.5→0.1 schedule). Optionally drop the high-σ
    # early portion and goals where the basin destabilized.
    python reconstruct_reinit_trajectories.py \\
        --state-pkl-dir /tmp/phase1_reinit_anneal_dl \\
        --out-dir distill_inputs/sigma_anneal \\
        --drop-first-iters 50 \\
        --exclude-goals 9,18,21

Then distill on the combined set:
    python train_phase2_distill_closed_loop.py \\
        --runs-dir distill_inputs/reinit_K20 \\
        --runs-dir distill_inputs/sigma_anneal \\
        --out pretrain/policy_distilled.pt
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

# The reinit driver hardcoded K=20 mirrored ES candidates per iter.
# Layout is 1 baseline + K candidates per iter → 21 entries per iter,
# plus 1 leading initial-FDFD entry, plus (when complete) 1 trailing
# post-loop ε_mean FDFD.
K = 20
PER_ITER = 1 + K  # = 21


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Phase 1 reinit state.pkl files into per-goal "
                    "eps_traj.npy for closed-loop distillation.")
    p.add_argument("--state-pkl-dir", required=True, type=Path,
                   help="Directory with state_target_<NN>.pkl files.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output dir; writes goal_<NN>/eps_traj.npy and "
                        "summary.json (consumable by train_phase2_distill_"
                        "closed_loop.py --runs-dir).")
    p.add_argument("--drop-first-iters", type=int, default=0,
                   help="Skip the first N trajectory steps per goal — useful "
                        "for σ-anneal runs where the high-σ exploration "
                        "phase produces noisy descent that doesn't match the "
                        "deployed-σ regime. Default 0.")
    p.add_argument("--exclude-goals", type=str, default=None,
                   help="Comma-separated goals to skip entirely (e.g. ones "
                        "that destabilized into low-tf basins).")
    p.add_argument("--min-final-tf", type=float, default=0.0,
                   help="Skip goals whose final best_tf is below this. 0 = "
                        "include everything. Useful for filtering out σ-anneal "
                        "goals where the run destabilized.")
    return p.parse_args()


def reconstruct_one(state: dict, drop_first: int) -> tuple[np.ndarray, dict]:
    """Reconstruct eps_traj from one goal's state.pkl. Returns (traj, info)."""
    eps_buffer = np.stack(state["eps_buffer"]).astype(np.float32)
    n_iter = int(state.get("iter", 0))
    complete = bool(state.get("complete", False))

    # Baseline at start of each iter.
    baseline_indices = [1 + t * PER_ITER for t in range(n_iter)]
    # Filter to only valid indices (defensive; partial runs sometimes end
    # mid-iter with an incomplete per-iter chunk).
    baseline_indices = [i for i in baseline_indices if i < len(eps_buffer)]
    # Post-loop final ε_mean (only present if loop ran the full N_iter).
    final_idx = 1 + n_iter * PER_ITER
    if complete and final_idx < len(eps_buffer):
        baseline_indices.append(final_idx)

    traj = eps_buffer[baseline_indices]
    if drop_first > 0:
        traj = traj[drop_first:]

    info = {
        "n_iter": n_iter,
        "complete": complete,
        "buffer_len": int(len(eps_buffer)),
        "n_baselines_extracted": int(len(baseline_indices)),
        "traj_steps_after_drop": int(traj.shape[0]),
        "drop_first": drop_first,
        "best_tf": float(state.get("best_tf", 0.0)),
    }
    return traj, info


def main():
    args = parse_args()
    exclude = (set(int(x) for x in args.exclude_goals.split(","))
               if args.exclude_goals else set())

    state_files = sorted(args.state_pkl_dir.glob("state_target_*.pkl"))
    if not state_files:
        raise FileNotFoundError(
            f"No state_target_*.pkl under {args.state_pkl_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    total_steps = 0
    n_kept = n_skipped = 0
    for sp in state_files:
        try:
            goal = int(sp.stem.split("_")[-1])
        except ValueError:
            continue
        if goal in exclude:
            print(f"  goal {goal:>2}: EXCLUDED (--exclude-goals)")
            n_skipped += 1
            continue

        state = pickle.loads(sp.read_bytes())
        best_tf = float(state.get("best_tf", 0.0))
        if best_tf < args.min_final_tf:
            print(f"  goal {goal:>2}: skipped (best_tf={best_tf:.4f} < "
                  f"--min-final-tf={args.min_final_tf})")
            n_skipped += 1
            continue

        traj, info = reconstruct_one(state, args.drop_first_iters)
        if traj.shape[0] < 2:
            print(f"  goal {goal:>2}: skipped (only {traj.shape[0]} trajectory "
                  f"steps after drop)")
            n_skipped += 1
            continue

        gd = args.out_dir / f"goal_{goal:02d}"
        gd.mkdir(parents=True, exist_ok=True)
        np.save(gd / "eps_traj.npy", traj)
        info["goal"] = goal
        info["source"] = str(sp.resolve())
        summary.append(info)
        n_kept += 1
        total_steps += traj.shape[0] - 1     # n transitions = n_states - 1
        print(f"  goal {goal:>2}: traj shape={traj.shape}  "
              f"(buffer={info['buffer_len']:,}, iter={info['n_iter']}, "
              f"complete={info['complete']}, best_tf={best_tf:.4f})  "
              f"→ {gd.name}/eps_traj.npy")

    (args.out_dir / "summary.json").write_text(json.dumps({
        "source_dir": str(args.state_pkl_dir.resolve()),
        "drop_first_iters": args.drop_first_iters,
        "exclude_goals": sorted(exclude),
        "min_final_tf": args.min_final_tf,
        "n_goals_kept": n_kept,
        "n_goals_skipped": n_skipped,
        "total_transitions": total_steps,
        "per_goal": summary,
    }, indent=2))
    print(f"\n{n_kept} goals → {args.out_dir.resolve()}")
    print(f"Total (ε_t, ε_t+1, goal) supervision tuples: {total_steps:,}")


if __name__ == "__main__":
    main()
