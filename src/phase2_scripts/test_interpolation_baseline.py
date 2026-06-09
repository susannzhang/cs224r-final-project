"""Direct linear interpolation baseline for Phase 2 retargeting.

For each Phase 2 goal θ, compute ε(θ) as the linear interpolation between
the nearest two Phase 1 anchor ε* configurations, then FDFD it and report
the deployment metric (target_frac). No NN, no rollouts, no ES — just
weighted-average of the two nearest Phase 1 designs.

If this matches or beats the buffer-traj imitation policy, the closed-loop
framing was unnecessary structure.

Usage:
    python test_interpolation_baseline.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent          # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                       # cs153 repo root (geometry, simulation)
for _p in (PROJECT_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from algorithms.agents.es_agent import apply_eps_to_canvas
from geometry import (create_design_region, create_environment, create_grid,
                      create_receiver, create_source)
from simulation import initialize_environment, simulate_ez_fields_per_source


def build_pm_env():
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0,
                                  margin_cells=20)
    grid = create_grid(num_rods_x=10, num_rods_y=10, radius=0.01,
                       distance=0.002, rod_permittivity=1.0)
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


def fdfd_P(env, eps):
    apply_eps_to_canvas(env, eps)
    ez = sum(simulate_ez_fields_per_source(env).values())
    intensity = np.abs(ez) ** 2
    return np.array([float(np.sum(intensity * r._mask))
                     for r in env.receivers])


def interp_eps(goal: int, memory_bank: dict) -> np.ndarray:
    """Linear interpolation of ε between nearest Phase 1 anchors."""
    if goal in memory_bank:
        return memory_bank[goal].copy()
    known = sorted(memory_bank.keys())
    lower = max((a for a in known if a < goal), default=None)
    upper = min((a for a in known if a > goal), default=None)
    if lower is None:
        return memory_bank[upper].copy()
    if upper is None:
        return memory_bank[lower].copy()
    alpha = (goal - lower) / (upper - lower)
    return ((1 - alpha) * memory_bank[lower] + alpha * memory_bank[upper]
            ).astype(np.float32)


def main():
    bank = {}
    for d in sorted(Path("phase1-uniform-init-output").glob("target_*")):
        if not d.is_dir():
            continue
        try:
            idx = int(d.name.split("_")[-1])
        except ValueError:
            continue
        p = d / "eps_star.npy"
        if p.exists():
            bank[idx] = np.load(p).astype(np.float32)
    print(f"Memory bank angles: {sorted(bank.keys())}")

    goal_indices = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    env = build_pm_env()

    # For each goal, compute the warm-start (from nearest Phase 1 anchor)
    # AND the interpolated ε, then FDFD both.
    print()
    print(f"{'goal':>4}  {'nearest':>7}  α    "
          f"init_frac  interp_frac  Δ")
    print("-" * 60)
    total_init = total_interp = 0.0
    n_improved = 0
    rows = []
    for goal in goal_indices:
        # warm-start: nearest Phase 1 anchor by index distance
        nearest = min(bank.keys(), key=lambda a: abs(a - goal))
        eps_init = bank[nearest]
        P_init = fdfd_P(env, eps_init)
        tf_init = float(P_init[goal] / max(P_init.sum(), 1e-9))

        # interpolated ε
        eps_interp = interp_eps(goal, bank)
        # for our goals (1, 4, …, 28), they're always 1/3 from lower anchor
        known = sorted(bank.keys())
        lower = max((a for a in known if a < goal), default=None)
        upper = min((a for a in known if a > goal), default=None)
        if lower is not None and upper is not None:
            alpha = (goal - lower) / (upper - lower)
        else:
            alpha = 0.0
        P_interp = fdfd_P(env, eps_interp)
        tf_interp = float(P_interp[goal] / max(P_interp.sum(), 1e-9))

        rows.append((goal, nearest, alpha, tf_init, tf_interp))
        total_init += tf_init
        total_interp += tf_interp
        if tf_interp > tf_init:
            n_improved += 1

        flag = '✓' if tf_interp > tf_init else ('~' if tf_interp > 0.9 * tf_init else '×')
        print(f"  {goal:>2}     {nearest:>3}     {alpha:.2f}   "
              f"{tf_init:.4f}    {tf_interp:.4f}    "
              f"{tf_interp - tf_init:+.4f}  {flag}")
    n = len(rows)
    print("-" * 60)
    print(f"  mean                {total_init/n:.4f}    {total_interp/n:.4f}    "
          f"{(total_interp-total_init)/n:+.4f}")
    print(f"  improved: {n_improved}/{n}")

    print()
    print("Comparison to NN-based methods:")
    print(f"  warm-start (init)                 : mean = {total_init/n:.4f}")
    print(f"  direct interpolation              : mean = {total_interp/n:.4f}  ({n_improved}/{n})")
    print(f"  buffer-traj imitation NN          : mean = 0.142             (6/10)")
    print(f"  ES (σ=0.02, iter 4)               : mean = 0.017             (0/10)")
    print(f"  BPTT through M (300 iters)        : mean = 0.047             (0/10)")


if __name__ == "__main__":
    main()
