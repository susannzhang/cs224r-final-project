"""Analyze where the iter-4 saved policy actually directs energy.

For each Phase 2 goal, rolls out the policy from ε*(θ_prev) for T=20 steps
and computes the full P[30] vector at the final state via FDFD. Shows which
receivers actually receive energy, what fraction goes to the intended target,
and identifies the failure mode (attenuation, off-target peak, etc).
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
from algorithms.policies.es_policy import ESPolicy
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
    return np.array([float(np.sum(intensity * r._mask)) for r in env.receivers])


def load_memory_bank(d):
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
    return bank


def main():
    ckpt_dir = Path("checkpoint_output/phase2/phase2-mfilter1k-20260601-232950")
    policy_path = ckpt_dir / "policy_phase2.pt"
    print(f"Loading policy from {policy_path}")
    policy = ESPolicy.load(policy_path)

    bank = load_memory_bank(Path("phase1-uniform-init-output"))
    env = build_pm_env()

    pairs = [(0,1),(3,4),(6,7),(9,10),(12,13),(15,16),(18,19),(21,22),(24,25),(27,28)]
    T = 20

    print()
    print(f"{'targ':>4}  {'prev':>4}  init_argmax  init_top3        "
          f"final_argmax  final_top3       final_top3_P            "
          f"P_total_30  target_frac")
    print("-" * 125)

    for prev, target in pairs:
        eps = bank[prev].astype(np.float32).copy()
        P_init = fdfd_P(env, eps)

        # Run T=20 policy steps
        for _ in range(T):
            delta = policy.predict(eps, target).astype(np.float32)
            eps = np.clip(eps + delta, -1.0, 1.0).astype(np.float32)

        P_fin = fdfd_P(env, eps)
        total = float(P_fin.sum())
        tf = float(P_fin[target] / total) if total > 0 else 0.0
        top3_idx = np.argsort(-P_fin)[:3]
        top3_init = np.argsort(-P_init)[:3]

        # show top-3 with their P values
        top3_str = "  ".join(f"r{i:>2}({P_fin[i]:.0f})" for i in top3_idx)

        print(f"  {target:>2}    {prev:>2}      "
              f"{int(P_init.argmax()):>2}        "
              f"[{','.join(str(int(x)) for x in top3_init):<10}]   "
              f"  {int(P_fin.argmax()):>2}        "
              f"[{','.join(str(int(x)) for x in top3_idx):<10}]   "
              f"{top3_str:<22}  "
              f"{total:>10.1f}  "
              f"{tf:.4f}")


if __name__ == "__main__":
    main()
