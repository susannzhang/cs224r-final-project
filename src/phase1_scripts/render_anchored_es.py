"""Render PNGs for the anchored online ES result (max of interp vs online).

For each Phase 2 goal, the anchored ε was saved by deploy_phase2_online.py
to `checkpoint_output/phase2/online-state-es/eps_anchored_target_NN.npy`.
This script renders the standard 2×2 PNG (initial ε + field, anchored ε +
field) for each.

Output: checkpoint_output/phase2/online-state-es-anchored/target_NN.png
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent          # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                       # cs153 repo root (geometry, simulation)
for _p in (PROJECT_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from render_phase2_ckpt import (build_pm_env, load_memory_bank,
                                 pick_theta_prev, render_2x2_png)


def main():
    src_dir = Path("checkpoint_output/phase2/online-state-es")
    out_dir = Path("checkpoint_output/phase2/online-state-es-anchored")
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_memory_bank(Path("phase1-uniform-init-output"))
    converged_indices = sorted(bank.keys())
    summary_path = src_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No summary at {summary_path} — run deploy_phase2_online.py first.")
    summary = json.loads(summary_path.read_text())

    env = build_pm_env()
    print(f"Output → {out_dir}\n")

    new_summary = []
    for entry in summary["per_goal"]:
        goal = entry["goal"]
        prev = entry["prev"]
        tf_warm = entry["tf_warm"]
        tf_anchored = entry["tf_anchored"]
        pick = entry["pick"]

        # Load the anchored ε
        anch_path = src_dir / f"eps_anchored_target_{goal:02d}.npy"
        eps_anchored = np.load(anch_path)
        eps_init = bank[prev]

        title_suffix = (f"[anchored:{pick}] "
                        f"tf {tf_warm:.3f} → {tf_anchored:.3f}  "
                        f"(Δ {tf_anchored - tf_warm:+.3f})")
        save_path = out_dir / f"target_{goal:02d}.png"
        P_init, P_final = render_2x2_png(
            env, eps_init, eps_anchored, goal, save_path,
            title_suffix=title_suffix)
        tf_init = float(P_init[goal] / max(P_init.sum(), 1e-9))
        tf_final = float(P_final[goal] / max(P_final.sum(), 1e-9))
        print(f"  goal={goal:>2}  prev={prev:>2}  pick={pick:>6}  "
              f"tf: {tf_init:.4f} → {tf_final:.4f}  → {save_path.name}")
        new_summary.append({
            "target_idx": goal,
            "theta_prev": prev,
            "pick": pick,
            "target_frac_initial": float(tf_init),
            "target_frac_final": float(tf_final),
            "png": str(save_path),
        })

    (out_dir / "viz_summary.json").write_text(json.dumps({
        "method": "anchored_online_es",
        "memory_bank_dir": "phase1-uniform-init-output",
        "results": new_summary,
    }, indent=2))

    m_init = np.mean([r["target_frac_initial"] for r in new_summary])
    m_final = np.mean([r["target_frac_final"] for r in new_summary])
    n_improved = sum(1 for r in new_summary
                     if r["target_frac_final"] > r["target_frac_initial"])
    print()
    print(f"Mean target_frac:  init={m_init:.4f}  anchored={m_final:.4f}  "
          f"Δ={m_final - m_init:+.4f}")
    print(f"Improved: {n_improved}/{len(new_summary)}")
    print(f"\nWrote {len(new_summary)} PNGs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
