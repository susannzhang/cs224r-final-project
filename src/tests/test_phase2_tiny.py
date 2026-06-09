"""
End-to-end Phase 1 → BC init → Phase 2 smoke run on the tiny env, with
before/after visualizations comparing the BC-initialized policy against the
Phase-2-trained policy.

Runnable two ways:
    pytest tests/test_phase2_tiny.py -m slow          # one slow integration test
    python tests/test_phase2_tiny.py                  # script mode with argparse

Expected wall time at defaults (K=20, N_iter=30, T=5): ~3 minutes.

Outputs under ./phase2_tiny_smoke/ (sibling of tests/):
    phase1_results/eps_star_NN.npy      - per-target ε* (memory bank)
    phase1_results/replay_buffer.pkl    - merged transitions
    policy_awr.pt                       - policy after AWR warm-start
    policy_phase2.pt                    - policy after Phase 2 ES
    phase2_history.json                 - per-log_every-step training log
    viz/retargeting_target_NN.png       - 3x2 before/after per target
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path so `python tests/test_phase2_tiny.py`
# (script mode) can import the project's top-level modules. Pytest from the
# repo root doesn't need this, but the redundancy is harmless.
_DBS_ROOT = Path(__file__).resolve().parents[1]      # dynamic_beam_steering/
_PROJ_ROOT = _DBS_ROOT.parent                           # cs153 repo root (geometry, simulation)
for _p in (_PROJ_ROOT, _DBS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pytest

from geometry import (create_design_region, create_grid, create_source,
                      create_receiver, create_environment)
from simulation import initialize_environment, simulate_ez_fields_per_source

from algorithms.agents.es_agent import (ESAgent, ESAgentConfig,
                                         apply_eps_to_canvas)
from algorithms.policies.es_policy import ESPolicy, ESPolicyConfig, Phase2Config
from algorithms.infrastructure.utils import ReplayBuffer


OUT = Path(__file__).resolve().parents[1] / "phase2_tiny_smoke"


# =============================================================================
# Env factory + rollout helpers
# =============================================================================

def _build_tiny_env():
    """3×3 grid, 3 receivers (bottom / right / top centered)."""
    region = create_design_region(resolution=0.005, bg_permittivity=1.0,
                                  margin_cells=10)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.01, distance=0.005,
                       rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
    receivers = [
        create_receiver(index=1, length=0.01, side='bottom', rod_index=2),
        create_receiver(index=2, length=0.01, side='right',  rod_index=2),
        create_receiver(index=3, length=0.01, side='top',    rod_index=2),
    ]
    env = create_environment(design_region=region, grid=grid,
                             sources=[source], receivers=receivers)
    initialize_environment(env)
    return env


def _eval_rollout(policy, eps_0, goal, T):
    """
    Inference rollout under `policy`. No noise, no learning, no FDFD.
    Just steps δ_t = π_φ(ε_t, goal); ε_{t+1} = clip(ε_t + δ_t).
    Returns the final ε after T steps.
    """
    eps = eps_0.astype(np.float32, copy=True)
    for _ in range(T):
        delta = policy.predict(eps, int(goal)).astype(np.float32)
        eps = np.clip(eps + delta, -1.0, 1.0).astype(np.float32)
    return eps


def _fdfd_intensity_and_powers(env, eps):
    """Apply ε, run FDFD once, return (canvas, intensity, per-receiver powers)."""
    apply_eps_to_canvas(env, eps)
    canvas = env.design_region._canvas.copy()
    ez = sum(simulate_ez_fields_per_source(env).values())
    intensity = np.abs(ez) ** 2
    P = np.array([float(np.sum(intensity * r._mask)) for r in env.receivers])
    return canvas, intensity, P


# =============================================================================
# Visualization — 3×2 before/after per target
# =============================================================================

def _render_retargeting_png(env, target_idx, initial_eps, bc_eps, p2_eps,
                            save_path, title_suffix=""):
    """
    3×2 figure per target:
        row 1: initial ε        | initial field intensity
        row 2: BC-final ε       | BC-final field intensity
        row 3: Phase 2-final ε  | Phase 2-final field intensity
    Target receiver is outlined in cyan on each intensity panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    states = [
        ("Initial (cold start / retargeting init)", initial_eps),
        ("After BC init only", bc_eps),
        ("After Phase 2 (BC + ES on φ)", p2_eps),
    ]
    rendered = [_fdfd_intensity_and_powers(env, e) for _, e in states]
    target_receiver = env.receivers[target_idx]

    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)

    def _draw_perm(ax, canvas, title):
        clipped = np.clip(canvas, 0, 10)
        im = ax.imshow(clipped, cmap='plasma', origin='lower', vmin=0, vmax=10)
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='ε')

    def _draw_int(ax, intensity, canvas, title, P):
        vmax = np.percentile(intensity, 98)
        im = ax.imshow(intensity, cmap='inferno', origin='lower', vmin=0, vmax=vmax)
        ax.contour(canvas, [3.0, 5e5], colors='white', alpha=0.5, linewidths=0.6)
        ax.contour(target_receiver._mask, [0.5], colors='cyan', linewidths=1.5)
        total = float(P.sum())
        frac = float(P[target_idx] / total) if total > 0 else 0.0
        ax.set_title(f"{title}\ntarget_frac = {frac:.3f}")
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='|E_z|²')

    for row, ((label, _), (canvas, intensity, P)) in enumerate(zip(states, rendered)):
        _draw_perm(axes[row, 0], canvas, f"{label}\npermittivity")
        _draw_int(axes[row, 1], intensity, canvas, f"{label}\nfield intensity", P)

    fig.suptitle(f"Phase 2 retargeting toward receiver {target_idx} {title_suffix}",
                 fontsize=13)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Default config + pipeline
# =============================================================================

def _default_args():
    """Defaults used by both the pytest path and `python tests/test_phase2_tiny.py`."""
    return argparse.Namespace(
        # Phase 1
        phase1_K=10, phase1_M=20, phase1_lr=0.05,
        # AWR warm-start
        awr_epochs=20,
        # Phase 2
        K=20, N_iter=30, T=5, alpha2=0.005, sigma=0.05,
        seed=0,
    )


def _run_pipeline(args):
    """Full Phase 1 → BC → Phase 2 → before/after viz pipeline."""
    OUT.mkdir(exist_ok=True)

    print("=" * 64)
    print("Phase 1 → AWR warm-start → Phase 2 smoke run on tiny env (3×3 grid)")
    print("=" * 64)
    print(f"  Phase 1: K={args.phase1_K}, M={args.phase1_M}, lr={args.phase1_lr}")
    print(f"  AWR:     epochs={args.awr_epochs}")
    print(f"  Phase 2: K={args.K}, N_iter={args.N_iter}, T={args.T}, "
          f"σ={args.sigma}, α_2={args.alpha2}")
    print()

    env = _build_tiny_env()
    state_shape = (env.grid.num_rods_x, env.grid.num_rods_y)
    training_indices = [0, 1, 2]
    n_goals = len(env.receivers)

    # ============== Phase 1 ==============
    print("[Phase 1] training each of 3 target angles...")
    t0 = time.time()
    buffer = ReplayBuffer()
    memory_bank = {}
    p1_cfg = ESAgentConfig(K=args.phase1_K, sigma=0.1, alpha_1=args.phase1_lr,
                           M=args.phase1_M, eta=-1.0,
                           log_every=max(1, args.phase1_M // 4), seed=args.seed)
    for target_idx in training_indices:
        agent = ESAgent(env=env, training_indices=training_indices,
                        config=p1_cfg, buffer=buffer, verbose=False)
        result = agent.train_one_angle(target_idx)
        memory_bank[target_idx] = result.eps_star
        print(f"   target {target_idx}: best_reward={result.best_reward:+.3e}  "
              f"iters={result.iterations}  buffer={len(buffer)}")
    print(f"   Phase 1 wall time: {time.time() - t0:.1f}s")

    # Persist Phase 1 artifacts
    p1_out = OUT / "phase1_results"
    p1_out.mkdir(exist_ok=True)
    for k, v in memory_bank.items():
        np.save(p1_out / f"eps_star_{k:02d}.npy", v)
    with open(p1_out / "replay_buffer.pkl", "wb") as fh:
        pickle.dump(buffer.transitions, fh)

    # ============== AWR warm-start ==============
    print(f"\n[AWR init] training policy on {len(buffer)} buffer transitions...")
    t0 = time.time()
    policy = ESPolicy(
        state_shape=state_shape,
        config=ESPolicyConfig(n_goals=n_goals, awr_epochs=args.awr_epochs,
                              awr_lr=1e-3, seed=args.seed),
    )
    awr_hist = policy.awr_init(buffer)
    policy.save(OUT / "policy_awr.pt")
    print(f"   train_loss: {awr_hist['train_loss'][0]:.4f} → {awr_hist['train_loss'][-1]:.4f}")
    print(f"   val_loss:   {awr_hist['val_loss'][0]:.4f} → {awr_hist['val_loss'][-1]:.4f}")
    print(f"   weight_stats: {awr_hist['weight_stats']}")
    print(f"   AWR wall time: {time.time() - t0:.1f}s")

    # ============== Capture warm-start rollouts (BEFORE Phase 2) ==============
    print("\n[viz] capturing AWR-only rollouts for the 'before' snapshot...")
    awr_snapshots = {}  # {target_idx: (initial_eps, awr_final_eps)}
    for target_idx in training_indices:
        # Retargeting scenario: start from the previous angle's ε*
        prev = (target_idx + 1) % len(training_indices)
        eps_0 = memory_bank[prev].copy()
        awr_final = _eval_rollout(policy, eps_0, goal=target_idx, T=args.T)
        awr_snapshots[target_idx] = (eps_0, awr_final)

    # ============== Phase 2 ==============
    print(f"\n[Phase 2] ES on φ, K={args.K}, N_iter={args.N_iter}, T={args.T}...")
    t0 = time.time()
    p2_cfg = Phase2Config(
        K=args.K, sigma=args.sigma, alpha_2=args.alpha2,
        N_iter=args.N_iter, T=args.T, eta=-1.0,
        p_rand=0.3,
        log_every=max(1, args.N_iter // 6), seed=args.seed,
    )
    p2_result = policy.train_phase2(
        env=env, buffer=buffer, memory_bank=memory_bank,
        goal_indices=training_indices, config=p2_cfg,
    )
    policy.save(OUT / "policy_phase2.pt")
    with open(OUT / "phase2_history.json", "w") as fh:
        json.dump(p2_result["history"], fh, indent=2)
    print(f"   Phase 2 wall time: {time.time() - t0:.1f}s")
    print(f"\n   Phase 2 history (every {p2_cfg.log_every} iters):")
    for e in p2_result["history"]:
        print(f"     iter {e['iteration']:>3}: "
              f"fitness {e['fitness_mean']:+.3e}/{e['fitness_best']:+.3e} "
              f"(mean/best)  rollout_len={e['mean_rollout_length']:.1f}")

    # ============== Capture Phase 2 rollouts (AFTER) + render figures ==============
    print("\n[viz] capturing Phase-2 rollouts and rendering 3×2 retargeting figures...")
    viz_dir = OUT / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = (f"(K={args.K}, N_iter={args.N_iter}, T={args.T}, "
                    f"σ={args.sigma}, α_2={args.alpha2})")
    for target_idx in training_indices:
        initial_eps, awr_final_eps = awr_snapshots[target_idx]
        # IMPORTANT: same eps_0 as the AWR rollout so the comparison is fair.
        p2_final_eps = _eval_rollout(policy, initial_eps,
                                     goal=target_idx, T=args.T)
        save_path = viz_dir / f"retargeting_target_{target_idx:02d}.png"
        _render_retargeting_png(env, target_idx,
                                initial_eps, awr_final_eps, p2_final_eps,
                                save_path, title_suffix=title_suffix)
        print(f"   ✓ wrote {save_path}")

    print()
    print(f"Buffer at end: {len(buffer):,} transitions")
    print(f"Outputs → {OUT.resolve()}")


# =============================================================================
# Pytest entry point + CLI entry point
# =============================================================================

@pytest.mark.slow
def test_phase2_tiny_end_to_end():
    """Full Phase 1 → AWR → Phase 2 pipeline on the tiny env. Roughly 3 min."""
    _run_pipeline(_default_args())
    # Sanity: outputs exist on disk
    assert (OUT / "policy_awr.pt").exists()
    assert (OUT / "policy_phase2.pt").exists()
    assert (OUT / "viz" / "retargeting_target_00.png").exists()


def _parse_cli():
    d = _default_args()
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-K", dest="phase1_K", type=int, default=d.phase1_K)
    p.add_argument("--phase1-M", dest="phase1_M", type=int, default=d.phase1_M)
    p.add_argument("--phase1-lr", dest="phase1_lr", type=float, default=d.phase1_lr)
    p.add_argument("--awr-epochs", dest="awr_epochs", type=int, default=d.awr_epochs)
    p.add_argument("--K", type=int, default=d.K)
    p.add_argument("--N_iter", type=int, default=d.N_iter)
    p.add_argument("--T", type=int, default=d.T)
    p.add_argument("--alpha2", type=float, default=d.alpha2)
    p.add_argument("--sigma", type=float, default=d.sigma)
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


if __name__ == "__main__":
    _run_pipeline(_parse_cli())
