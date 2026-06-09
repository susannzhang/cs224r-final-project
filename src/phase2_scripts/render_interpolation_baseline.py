"""Render PNGs for the linear-interpolation baseline.

Same 2x2 PNG format as render_phase2_ckpt.py (initial ε + field, final ε +
field), but the 'final' state is the linear-interpolated ε rather than a
T-step policy rollout. No NN, no rollout — just weighted-average of the
two nearest Phase 1 ε* configs.

Output: checkpoint_output/phase2/interpolation-baseline/target_NN.png
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
from test_interpolation_baseline import interp_eps


def main():
    memory_bank_dir = Path("phase1-uniform-init-output")
    out_dir = Path("checkpoint_output/phase2/interpolation-baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_memory_bank(memory_bank_dir)
    print(f"Memory bank: {sorted(bank.keys())}")
    converged_indices = sorted(bank.keys())
    goal_indices = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    print(f"Phase 2 goals: {goal_indices}")
    print(f"Output → {out_dir}")

    env = build_pm_env()
    print("Built pm_setup env (10×10 grid, 30 receivers)")

    summary = []
    for goal in goal_indices:
        theta_prev = pick_theta_prev(goal, converged_indices, "nearest", 0)
        eps_init = bank[theta_prev]
        eps_final = interp_eps(goal, bank)

        # find the two anchors used in interpolation (for the title)
        known = sorted(bank.keys())
        lower = max((a for a in known if a < goal), default=None)
        upper = min((a for a in known if a > goal), default=None)
        if lower is not None and upper is not None:
            alpha = (goal - lower) / (upper - lower)
            mix_str = (f"{1-alpha:.2f}·ε*({lower}) + {alpha:.2f}·ε*({upper})")
        elif lower is not None:
            mix_str = f"ε*({lower}) (boundary)"
        else:
            mix_str = f"ε*({upper}) (boundary)"

        save_path = out_dir / f"target_{goal:02d}.png"
        title_suffix = f"[interp] {mix_str}  from ε*(θ_prev={theta_prev})"
        P_initial, P_final = render_2x2_png(
            env, eps_init, eps_final, goal, save_path,
            title_suffix=title_suffix)

        ti = float(P_initial.sum())
        tf = float(P_final.sum())
        fi = P_initial[goal] / ti if ti > 0 else 0.0
        ff = P_final[goal] / tf if tf > 0 else 0.0
        improvement = ff > fi
        flag = "✓" if improvement else "×"
        print(f"  goal={goal:>2}  prev={theta_prev:>2}  "
              f"target_frac: {fi:.3f} → {ff:.3f}  ({flag})  → {save_path.name}")
        summary.append({
            "target_idx": goal,
            "theta_prev": theta_prev,
            "interp_lower": lower,
            "interp_upper": upper,
            "target_frac_initial": float(fi),
            "target_frac_final": float(ff),
            "png": str(save_path),
        })

    (out_dir / "viz_summary.json").write_text(json.dumps({
        "method": "linear_interpolation",
        "memory_bank_dir": str(memory_bank_dir),
        "results": summary,
    }, indent=2))

    means_init = np.mean([r["target_frac_initial"] for r in summary])
    means_final = np.mean([r["target_frac_final"] for r in summary])
    n_improved = sum(1 for r in summary
                     if r["target_frac_final"] > r["target_frac_initial"])
    print()
    print(f"Mean target_frac:  init={means_init:.4f}  interp={means_final:.4f}  "
          f"Δ={means_final - means_init:+.4f}")
    print(f"Improved: {n_improved}/{len(summary)}")
    print(f"\nWrote {len(summary)} PNGs + viz_summary.json to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
