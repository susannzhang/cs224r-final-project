# =============================================================================
# Phase 1 Training Runner — parallel ES over all target angles
# =============================================================================
"""
Runs Phase 1 ES (es_agent.ESAgent) once per target receiver, in parallel.

Each target angle is trained in its OWN process. Workers reconstruct the
environment from scratch (no shared canvas → no races on env.design_region._canvas).
Within a worker, the ES inner loop is serial; parallelism is one-process-per-angle.

Defaults: K=20, M=500, σ=0.1, α_1=0.02, η=1e-2. The default target set is every
3rd receiver (indices 0, 3, 6, ..., 27) → 10 angles, matching the proposal's
"10 training angles spanning [0°, 180°]" specification.

Usage:
    python train_phase1.py                                            # 10 angles, defaults
    python train_phase1.py --population-size 4 --max-iterations 10    # quick smoke
    python train_phase1.py --targets 10 13 16 19                      # custom subset
    python train_phase1.py --workers 4                                # cap parallelism

Outputs (under ./phase1_checkpoints/):
    target_<NN>/eps_star.npy        # converged ε*(θ) of shape (N_x, N_y)  [memory bank]
    target_<NN>/eps_initial.npy     # random ε the inner loop started from
    target_<NN>/metadata.json       # convergence info + history + receiver powers
    target_<NN>/before_after.png    # 2x2 initial-vs-converged visualization
    target_<NN>/critic.pt           # only when --use-critic: DQNCritic checkpoint
    replay_buffer.pkl               # merged Transitions across all workers
    summary.json                    # one-line-per-angle convergence report
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import List

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_DBS = _Path(__file__).resolve().parent           # dynamic_beam_steering/
_PROJ = _DBS.parent                                # cs153 repo root
for _p in (_PROJ, _DBS):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
from geometry import (create_design_region, create_grid, create_source,
                      create_receiver, create_environment)
from simulation import initialize_environment

from algorithms.agents.es_agent import ESAgent, ESAgentConfig
from algorithms.infrastructure.utils import ReplayBuffer
# DQNCritic is imported lazily inside the worker so that runs with --no-use-critic
# don't require torch to be installed.


CHECKPOINT_DIR = Path("phase1_checkpoints")


# =============================================================================
# Per-worker env factory — must be picklable / re-importable
# =============================================================================

def build_env():
    """Recreate the pm_setup.py environment from scratch.

    Each parallel worker calls this so that mutations to
    env.design_region._canvas (driven by apply_eps_to_canvas) are isolated.
    """
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0,
                                  margin_cells=20)
    grid = create_grid(num_rods_x=10, num_rods_y=10,
                       radius=0.01, distance=0.002, rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)

    # Receiver indices assigned to ascend with angle (CCW from bottom-left
    # through right to top-left). Right side ascends with rod_index already;
    # top side needs rod_index reversed because grid_y increases left → right
    # but angle in the top half ascends right → left.
    receivers = []
    for i in range(1, 11):
        receivers.append(create_receiver(
            index=i, length=0.02, side='bottom', rod_index=i,
        ))
    for i in range(1, 11):
        receivers.append(create_receiver(
            index=10 + i, length=0.02, side='right', rod_index=i,
        ))
    for i in range(1, 11):
        receivers.append(create_receiver(
            index=20 + i, length=0.02, side='top', rod_index=11 - i,
        ))

    env = create_environment(design_region=region, grid=grid,
                             sources=[source], receivers=receivers)
    initialize_environment(env)
    return env


# =============================================================================
# Before/after visualization — produces a 2x2 PNG comparing initial vs ε*
# =============================================================================

def _render_before_after_png(env, eps_initial, eps_final,
                             target_idx, cfg, result):
    """
    Render a 2x2 figure (initial vs converged permittivity + field intensity)
    matching the layout used by tests/test_es_agent.py::test_visualize_initial_vs_converged.

    Returns (png_bytes, P_initial, P_final) where P_* are per-receiver |E_z|²
    integrals on the initial and converged designs.
    """
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from algorithms.agents.es_agent import apply_eps_to_canvas
    from simulation import simulate_ez_fields_per_source

    # --- Capture INITIAL state ---
    apply_eps_to_canvas(env, eps_initial)
    canvas_initial = env.design_region._canvas.copy()
    ez_initial = sum(simulate_ez_fields_per_source(env).values())
    intensity_initial = np.abs(ez_initial) ** 2
    P_initial = np.array([
        float(np.sum(intensity_initial * r._mask)) for r in env.receivers
    ])

    # --- Capture FINAL state ---
    apply_eps_to_canvas(env, eps_final)
    canvas_final = env.design_region._canvas.copy()
    ez_final = sum(simulate_ez_fields_per_source(env).values())
    intensity_final = np.abs(ez_final) ** 2
    P_final = np.array([
        float(np.sum(intensity_final * r._mask)) for r in env.receivers
    ])

    # --- Build 2x2 figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    def _draw_permittivity(ax, canvas, title):
        clipped = np.clip(canvas, 0, 10)
        im = ax.imshow(clipped, cmap='plasma', origin='lower', vmin=0, vmax=10)
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='ε')

    def _draw_intensity(ax, intensity, canvas, title, target_receiver):
        vmax = np.percentile(intensity, 98)
        im = ax.imshow(intensity, cmap='inferno', origin='lower', vmin=0, vmax=vmax)
        ax.contour(canvas, [3.0, 5e5], colors='white', alpha=0.5, linewidths=0.6)
        ax.contour(target_receiver._mask, [0.5], colors='cyan', linewidths=1.5)
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='|E_z|²')

    target_receiver = env.receivers[target_idx]
    _draw_permittivity(axes[0, 0], canvas_initial,
                       'Initial permittivity (random ε)')
    _draw_intensity(axes[0, 1], intensity_initial, canvas_initial,
                    'Initial field intensity', target_receiver)
    _draw_permittivity(axes[1, 0], canvas_final,
                       f'Converged ε* (target receiver {target_idx}, '
                       f'{result.iterations} iters)')
    _draw_intensity(axes[1, 1], intensity_final, canvas_final,
                    f'Converged field intensity  (reward={result.best_reward:+.2e})',
                    target_receiver)

    fig.suptitle(f'ES Agent target {target_idx}: initial vs converged  '
                 f'(K={cfg.K}, M={cfg.M}, σ={cfg.sigma})',
                 fontsize=13)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue(), P_initial, P_final


# =============================================================================
# Worker — runs one target angle end-to-end + writes its per-target outputs
# =============================================================================

def _train_one_target(payload):
    """
    Train ε*(θ) for a single target receiver and write per-target outputs:
      - eps_star.npy        (memory bank entry)
      - eps_initial.npy     (snapshot of where the inner loop started)
      - before_after.png    (2x2 viz: initial vs converged permittivity + field)
      - per-target metadata (lives in metadata.json, written by main after merge)

    Transitions are returned to main so they can be merged into the shared
    replay buffer pickle (one .pkl across all workers).
    """
    target_idx, training_indices, config_kwargs, seed, out_dir_str, use_critic = payload
    out_dir = Path(out_dir_str)

    env = build_env()
    config = ESAgentConfig(**config_kwargs, seed=seed)

    # Optional critic: trained interleaved with ES, one TD(0) batch per ES iter.
    # Per-worker (own buffer, own Q-net) — a shared critic across workers would
    # be the next step but requires weight sync; not v0.
    critic = None
    if use_critic:
        from algorithms.critics.dqn_critic import DQNCritic, DQNCriticConfig
        critic = DQNCritic(
            state_shape=(env.grid.num_rods_x, env.grid.num_rods_y),
            config=DQNCriticConfig(n_goals=len(env.receivers), seed=seed),
        )

    agent = ESAgent(
        env=env,
        training_indices=training_indices,
        config=config,
        critic=critic,
        verbose=False,  # parallel stdout would interleave incoherently
    )
    result = agent.train_one_angle(target_idx)
    transitions = list(agent.buffer.transitions)

    # --- Memory bank: per-target ε* + initial ε ---
    angle_dir = out_dir / f"target_{target_idx:02d}"
    angle_dir.mkdir(parents=True, exist_ok=True)
    np.save(angle_dir / "eps_star.npy", result.eps_star)
    if result.eps_initial is not None:
        np.save(angle_dir / "eps_initial.npy", result.eps_initial)
    # Critic checkpoint (Phase 2 will consume this; pre-trained on this target's transitions)
    if critic is not None:
        critic.save(angle_dir / "critic.pt")

    # --- Before/after visualization (uses 2 extra FDFD solves per target) ---
    P_initial = P_final = None
    if result.eps_initial is not None:
        png_bytes, P_initial, P_final = _render_before_after_png(
            env, result.eps_initial, result.eps_star, target_idx, config, result,
        )
        (angle_dir / "before_after.png").write_bytes(png_bytes)

    # --- One-line summary (with initial vs final target-power fraction) ---
    status = "✓" if result.converged else "·"
    summary_extra = ""
    if P_initial is not None:
        ti = float(P_initial.sum())
        tf = float(P_final.sum())
        frac_i = float(P_initial[target_idx] / ti) if ti > 0 else 0.0
        frac_f = float(P_final[target_idx] / tf) if tf > 0 else 0.0
        summary_extra = f"  target_frac: {frac_i:.3f} → {frac_f:.3f}"
    print(f"  {status} target {target_idx:02d}  "
          f"iter={result.iterations:>4}  "
          f"reward={result.best_reward:+.3e}"
          f"{summary_extra}", flush=True)

    return (target_idx, result, transitions,
            None if P_initial is None else P_initial.tolist(),
            None if P_final is None else P_final.tolist())


# =============================================================================
# Orchestrator
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 parallel ES trainer")
    p.add_argument("--targets", type=int, nargs="+", default=None,
                   help="Target receiver indices (0-based). "
                        "Default: every 3rd receiver = [0, 3, 6, ..., 27] (10 angles).")
    p.add_argument("--training-indices", type=int, nargs="+", default=None,
                   help="Receivers used in the reward as training angles. "
                        "Default: same as --targets.")
    p.add_argument("--population-size", type=int, default=20, help="K (must be even)")
    p.add_argument("--sigma", type=float, default=0.1, help="σ, ε-space perturbation scale")
    p.add_argument("--learning-rate", type=float, default=0.05, help="α_1")
    p.add_argument("--max-iterations", type=int, default=500, help="M per angle")
    p.add_argument("--convergence-eta", type=float, default=1e-2,
                   help="early stop when P_target / P_total ≥ 1 - η")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel processes. Default: min(n_targets, cpu_count-1)")
    p.add_argument("--out", type=Path, default=CHECKPOINT_DIR,
                   help="Output directory for checkpoints.")
    p.add_argument("--use-critic", action=argparse.BooleanOptionalAction, default=False,
                   help="Train a DQNCritic interleaved with the ES update each "
                        "iteration. Saves critic.pt per target.")
    return p.parse_args()


def main():
    args = parse_args()

    # Default: every 3rd receiver — 10 evenly-spaced training angles out of 30.
    target_indices: List[int] = (
        list(args.targets) if args.targets is not None else list(range(0, 30, 3))
    )
    training_indices: List[int] = (
        list(args.training_indices)
        if args.training_indices is not None
        else list(target_indices)
    )

    if any(t not in training_indices for t in target_indices):
        raise ValueError(
            "Every --target must also appear in --training-indices "
            f"(targets={target_indices}, training={training_indices})"
        )

    config_kwargs = dict(
        K=args.population_size,
        sigma=args.sigma,
        alpha_1=args.learning_rate,
        M=args.max_iterations,
        eta=args.convergence_eta,
        log_every=args.log_every,
    )

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distinct seeds per target so identical-config workers don't sample identical noise.
    payloads = [
        (target_idx, training_indices, config_kwargs, seed, str(out_dir), args.use_critic)
        for seed, target_idx in enumerate(target_indices)
    ]

    n_workers = args.workers or min(len(payloads), max(1, mp.cpu_count() - 1))
    print(f"Phase 1: training {len(payloads)} target angles on {n_workers} workers.")
    print(f"  config: K={args.population_size}  M={args.max_iterations}  "
          f"σ={args.sigma}  α_1={args.learning_rate}  η={args.convergence_eta}")
    print(f"  targets: {target_indices}")
    print(f"  training_indices: {training_indices}")
    print(f"  critic:  {'ON (per-worker DQNCritic)' if args.use_critic else 'off'}")
    print(f"  outputs → {out_dir.resolve()}")
    print()

    # spawn context is the macOS default and avoids issues with fork+Ceviche.
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_train_one_target, payloads)

    # --- Merge per-target metadata + replay buffer ---
    buffer = ReplayBuffer()
    summary = []
    for target_idx, result, transitions, P_initial, P_final in results:
        angle_dir = out_dir / f"target_{target_idx:02d}"
        metadata = {
            "target_idx": target_idx,
            "best_reward": result.best_reward,
            "iterations": result.iterations,
            "converged": result.converged,
            "config": config_kwargs,
            "training_indices": training_indices,
            "history": result.history,
            "P_initial": P_initial,
            "P_final": P_final,
        }
        with open(angle_dir / "metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)

        buffer.extend(transitions)
        summary.append({
            "target_idx": target_idx,
            "iterations": result.iterations,
            "converged": result.converged,
            "best_reward": result.best_reward,
            "n_transitions": len(transitions),
        })

    with open(out_dir / "replay_buffer.pkl", "wb") as fh:
        pickle.dump(buffer.transitions, fh)

    with open(out_dir / "summary.json", "w") as fh:
        json.dump({
            "config": config_kwargs,
            "targets": target_indices,
            "training_indices": training_indices,
            "results": summary,
            "total_transitions": len(buffer),
        }, fh, indent=2)

    n_converged = sum(1 for r in summary if r["converged"])
    print()
    print(f"Done. {n_converged}/{len(summary)} angles converged. "
          f"Buffer: {len(buffer)} transitions across {len(summary)} workers.")
    print(f"Outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
