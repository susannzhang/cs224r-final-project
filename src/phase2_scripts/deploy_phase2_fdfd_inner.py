"""Online state-space ES with FDFD (not M) in the inner loop.

For the 3 Phase 2 goals where M-inner-loop ES failed (10, 13, 19), run
state-space ES with FDFD ground-truth scoring per iter. Each iter
dispatches K=20 candidate ε's to Modal in parallel for FDFD evaluation.

This is unbiased (no M-extrapolation pathology) but ~K× more expensive
than the M-inner-loop version. Used only for the hard goals where M's
bias prevents convergence to a useful optimum.

Usage:
    modal run deploy_phase2_fdfd_inner.py::main \\
        --memory-bank phase1-uniform-init-output \\
        --goals 10,13,19 \\
        --n-iter 30 --pop-size 20 --sigma 0.05 --alpha 0.05
"""

import json
import os
import pickle
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root (for geometry, simulation)
APP_NAME = os.environ.get("FDFD_INNER_APP_NAME", "cs224r-fdfd-inner")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install("numpy", "scipy", "scikit-image", "autograd", "ceviche")
    .add_local_dir(
        PROJECT_ROOT, "/root/app", copy=True,
        ignore=["__pycache__", "*.pyc", "phase1_checkpoints",
                "phase1_training_output", "phase1-uniform-init-output",
                "checkpoint_output", "phase2_tiny_smoke",
                "phase2_output", "phase2_parallel_output",
                "tests/visual_output", ".git", ".venv", ".pytest_cache",
                "*.pkl", "wandb"],
    )
)
app = modal.App(APP_NAME)


def _build_pm_env():
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")
    from geometry import (create_design_region, create_environment,
                          create_grid, create_receiver, create_source)
    from simulation import initialize_environment
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


@app.function(image=image, cpu=2, memory=4096, timeout=600)
def fdfd_batch(eps_batch_packed: bytes) -> bytes:
    """FDFD a batch of ε's; return packed (P_batch, receiver_indices)."""
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")
    import numpy as np
    from algorithms.agents.es_agent import apply_eps_to_canvas
    from simulation import simulate_ez_fields_per_source

    eps_batch = pickle.loads(eps_batch_packed)
    env = _build_pm_env()
    P_list = []
    for eps in eps_batch:
        eps = np.asarray(eps, dtype=np.float32)
        apply_eps_to_canvas(env, eps)
        ez = sum(simulate_ez_fields_per_source(env).values())
        intensity = np.abs(ez) ** 2
        P = np.array([float(np.sum(intensity * r._mask))
                      for r in env.receivers], dtype=np.float64)
        P_list.append(P)
    return pickle.dumps(np.stack(P_list))


def _load_memory_bank(d: Path):
    import numpy as np
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


def _interp_eps(goal, bank):
    import numpy as np
    if goal in bank:
        return bank[goal].copy()
    known = sorted(bank.keys())
    lower = max((a for a in known if a < goal), default=None)
    upper = min((a for a in known if a > goal), default=None)
    if lower is None:
        return bank[upper].copy()
    if upper is None:
        return bank[lower].copy()
    alpha = (goal - lower) / (upper - lower)
    return ((1 - alpha) * bank[lower] + alpha * bank[upper]).astype(np.float32)


def _centered_ranks(fitnesses):
    import numpy as np
    K = len(fitnesses)
    ranks = np.argsort(np.argsort(fitnesses))
    return ranks.astype(float) / K - 0.5


@app.local_entrypoint()
def main(
    memory_bank: str = "phase1-uniform-init-output",
    goals: str = "10,13,19",
    n_iter: int = 30,
    pop_size: int = 20,
    sigma: float = 0.05,
    alpha: float = 0.05,
    out_dir: str = "checkpoint_output/phase2/online-state-es-fdfd-inner",
    seed: int = 0,
):
    """Per-goal state-space ES with FDFD inner loop. Goals run serially;
    within each goal, K=20 FDFDs dispatch in parallel per iter."""
    import numpy as np

    rng = np.random.default_rng(seed)
    bank = _load_memory_bank(Path(memory_bank))
    print(f"Memory bank: {sorted(bank.keys())}")
    goal_list = [int(g) for g in goals.split(",")]
    print(f"Hard goals: {goal_list}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nConfig: n_iter={n_iter}  K={pop_size}  σ={sigma}  α={alpha}")
    print(f"Total FDFDs: {len(goal_list)} goals × {n_iter} iters × {pop_size} = "
          f"{len(goal_list) * n_iter * pop_size:,}")
    print()

    summary = []
    overall_t0 = time.time()
    half_K = pop_size // 2

    for goal in goal_list:
        print(f"=== goal {goal} ===")
        eps = _interp_eps(goal, bank).astype(np.float32)
        Q_history = []
        t0 = time.time()

        for it in range(n_iter):
            xi_half = rng.standard_normal((half_K, *eps.shape)).astype(np.float32)
            xi_pop = np.concatenate([xi_half, -xi_half], axis=0)
            eps_pop = np.clip(eps[None] + sigma * xi_pop, -1.0, 1.0)

            # FDFD the K candidates on Modal (one batch)
            batch_packed = pickle.dumps(list(eps_pop))
            chunk_bytes = list(fdfd_batch.map([batch_packed]))[0]
            P_pop = pickle.loads(chunk_bytes)         # (K, 30)

            P_target = P_pop[:, goal]
            P_total = P_pop.sum(axis=1)
            Q_pop = (P_target ** 2) / (P_total + 1e-9)
            Q_history.append({
                "iter": it,
                "Q_mean": float(Q_pop.mean()),
                "Q_max": float(Q_pop.max()),
            })

            u = _centered_ranks(Q_pop)
            grad = np.einsum('k,kij->ij', u, xi_pop) / (pop_size * sigma)
            eps = np.clip(eps + alpha * grad, -1.0, 1.0).astype(np.float32)

            if (it + 1) % 5 == 0:
                tf_iter = P_pop[Q_pop.argmax(), goal] / max(
                    P_pop[Q_pop.argmax()].sum(), 1e-9)
                print(f"  iter {it + 1:>3}: Q_max={Q_history[-1]['Q_max']:.1f}  "
                      f"Q_mean={Q_history[-1]['Q_mean']:.1f}  "
                      f"best_cand_target_frac={float(tf_iter):.4f}")

        elapsed = time.time() - t0

        # Final FDFD verification on the converged ε
        final_batch = pickle.dumps([eps])
        chunk = list(fdfd_batch.map([final_batch]))[0]
        P_final = pickle.loads(chunk)[0]
        tf_final = float(P_final[goal] / max(P_final.sum(), 1e-9))

        # Warm-start and interp baselines for comparison
        prev = min(bank.keys(), key=lambda a: abs(a - goal))
        baseline_batch = pickle.dumps([bank[prev], _interp_eps(goal, bank)])
        chunk = list(fdfd_batch.map([baseline_batch]))[0]
        P_baselines = pickle.loads(chunk)
        tf_warm = float(P_baselines[0, goal] / max(P_baselines[0].sum(), 1e-9))
        tf_interp = float(P_baselines[1, goal] / max(P_baselines[1].sum(), 1e-9))

        np.save(out_path / f"eps_target_{goal:02d}.npy", eps)
        summary.append({
            "goal": goal, "prev": prev,
            "tf_warm": tf_warm, "tf_interp": tf_interp, "tf_fdfd_es": tf_final,
            "elapsed_s": elapsed,
            "Q_history": Q_history,
        })
        print(f"  result: warm={tf_warm:.4f}  interp={tf_interp:.4f}  "
              f"FDFD_ES={tf_final:.4f}  ({elapsed:.1f}s)\n")

    overall_t = time.time() - overall_t0

    print(f"{'goal':>4}  {'prev':>4}  {'warm_tf':>7}  {'interp_tf':>9}  "
          f"{'FDFD_ES':>7}  {'Δ vs interp':>11}")
    print("-" * 65)
    n = len(summary)
    m_warm = sum(s["tf_warm"] for s in summary) / n
    m_interp = sum(s["tf_interp"] for s in summary) / n
    m_fdfd = sum(s["tf_fdfd_es"] for s in summary) / n
    n_beat_interp = sum(1 for s in summary if s["tf_fdfd_es"] > s["tf_interp"])
    n_beat_warm = sum(1 for s in summary if s["tf_fdfd_es"] > s["tf_warm"])
    for s in summary:
        flag = "✓" if s["tf_fdfd_es"] > s["tf_interp"] else "×"
        print(f"  {s['goal']:>2}    {s['prev']:>2}    "
              f"{s['tf_warm']:.4f}   {s['tf_interp']:.4f}    "
              f"{s['tf_fdfd_es']:.4f}   "
              f"{s['tf_fdfd_es'] - s['tf_interp']:+.4f}    {flag}")
    print("-" * 65)
    print(f"  mean         {m_warm:.4f}   {m_interp:.4f}    "
          f"{m_fdfd:.4f}   {m_fdfd - m_interp:+.4f}")
    print(f"  beat interp: {n_beat_interp}/{n}    beat warm: {n_beat_warm}/{n}")
    print(f"  total wall:  {overall_t:.1f}s ({overall_t/60:.1f} min)")

    (out_path / "summary.json").write_text(json.dumps({
        "config": {"n_iter": n_iter, "pop_size": pop_size,
                   "sigma": sigma, "alpha": alpha, "seed": seed},
        "goals": goal_list,
        "mean_tf_warm": m_warm, "mean_tf_interp": m_interp,
        "mean_tf_fdfd_es": m_fdfd,
        "n_beat_interp": n_beat_interp, "n_beat_warm": n_beat_warm,
        "total_wall_s": overall_t,
        "per_goal": summary,
    }, indent=2))

    print()
    print("If these 3 goals were swapped INTO the anchored result:")
    print(f"  original anchored mean (10 goals)         : 0.185")
    new_mean_10 = (0.185 * 10 - (
        sum(0.085 for g in goal_list if g == 10)
        + sum(0.053 for g in goal_list if g == 13)
        + sum(0.059 for g in goal_list if g == 19)
    ) + sum(s["tf_fdfd_es"] for s in summary)) / 10
    print(f"  with FDFD-ES on hard goals (projection)   : {new_mean_10:.3f}")
