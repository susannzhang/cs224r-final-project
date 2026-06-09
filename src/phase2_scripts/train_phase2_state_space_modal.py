# =============================================================================
# Phase 2 (state-space ES, parallel) — per-goal ε-ES with M filter on Modal
# =============================================================================
"""
Parallel state-space ES on Modal. Runs ESStateSpacePolicy per goal:
  - One `goal_driver` container per goal (so N_goals goals fan out as N_goals
    drivers concurrently).
  - Inside each goal_driver, every outer iter dispatches K_real FDFDs via
    `fdfd_one.map([...])` so the inner FDFDs also fan out across Modal workers.

Total parallelism: N_goals × (1 baseline + K_real) FDFDs in flight per outer
iter. At N_goals=10, K_real=20, that's up to 210 containers concurrent.

Wall time per goal: N_iter × (~FDFD time + Modal overhead). FDFD ~10-30s
per solve; outer iter ~30-60s including overhead and baseline. N_iter=50 →
~30-50 min per goal. All goals fan out → end-to-end ~30-50 min.

Checkpointing: per-goal state.pkl committed to a Modal Volume every
checkpoint_every iters. The driver is resumable from any checkpoint via the
ESStateSpaceCheckpoint shape in algorithms/policies/es_state_space_policy.py.

M training data: every FDFD'd (ε, P) pair is accumulated per-goal and
returned in the result. The local entrypoint aggregates across goals into
one .npz at the end, suitable for retraining M offline (mirrors the
phase1-reinit data-collection pattern).

Workflow:
    1. modal deploy train_phase2_state_space_modal.py        (one-time)
    2. python spawn_phase2_state_space.py ...                 (queues main)
    3. modal run train_phase2_state_space_modal.py::collect \\
         --launch-id <id>                                    (download)
"""

import io
import json
import os
import pickle
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root (for geometry, simulation)
APP_NAME = os.environ.get("PHASE2_SS_APP_NAME", "cs224r-phase2-state-space")
VOLUME_NAME = os.environ.get("PHASE2_SS_VOLUME_NAME",
                             "cs224r-phase2-state-space-buffer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install("numpy", "scipy", "scikit-image", "matplotlib",
                 "autograd", "ceviche", "wandb")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(
        PROJECT_ROOT, "/root/app", copy=True,
        ignore=["__pycache__", "*.pyc",
                "phase1_checkpoints", "phase1_training_output",
                "phase1-uniform-init-output", "phase1-reinit-output",
                "checkpoint_output", "phase2_tiny_smoke",
                "phase2_output", "phase2_parallel_output",
                "tests/visual_output", ".git", ".venv", ".pytest_cache",
                "*.pkl", "wandb"],
    )
)
app = modal.App(APP_NAME)
buffer_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

try:
    wandb_secret = modal.Secret.from_name("wandb")
except modal.exception.NotFoundError:
    wandb_secret = None


# =============================================================================
# Env builder — kept in sync with the other Phase 2 pipelines
# =============================================================================

def _build_pm_env():
    """Standard 10×10 pm_setup env with 30 receivers."""
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")
    from geometry import (create_design_region, create_environment,
                          create_grid, create_receiver, create_source)
    from simulation import initialize_environment

    region = create_design_region(resolution=0.0005, bg_permittivity=1.0,
                                  margin_cells=20)
    grid = create_grid(num_rods_x=10, num_rods_y=10,
                       radius=0.01, distance=0.002, rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
    receivers = []
    for i in range(1, 11):
        receivers.append(create_receiver(index=i,        length=0.02, side='bottom', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=10 + i,   length=0.02, side='right',  rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=20 + i,   length=0.02, side='top',    rod_index=11 - i))
    env = create_environment(design_region=region, grid=grid,
                             sources=[source], receivers=receivers)
    initialize_environment(env)
    return env


# =============================================================================
# fdfd_one — single FDFD solve, returns (P[30], P_loss)
# =============================================================================

@app.function(image=image, cpu=2, memory=4096, timeout=600)
def fdfd_one(eps_pkl: bytes) -> bytes:
    """One FDFD solve on a single ε. Returns pickled (P, P_loss)."""
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")
    import numpy as np
    from algorithms.agents.es_agent import apply_eps_to_canvas, compute_P_loss
    from simulation import simulate_ez_fields_per_source

    eps = pickle.loads(eps_pkl)
    env = _build_pm_env()
    apply_eps_to_canvas(env, np.asarray(eps, dtype=np.float32))
    ez = sum(simulate_ez_fields_per_source(env).values())
    intensity = np.abs(ez) ** 2
    P = np.array([
        float(np.sum(intensity * r._mask)) for r in env.receivers
    ], dtype=np.float64)
    P_loss = compute_P_loss(env, intensity)
    return pickle.dumps((P, float(P_loss)))


# =============================================================================
# goal_driver — ESStateSpacePolicy.run_one_goal for ONE goal
# =============================================================================

@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 6,                # 6h per goal (max)
    volumes={"/buffer": buffer_volume},
    secrets=[wandb_secret] if wandb_secret is not None else [],
)
def goal_driver(payload: dict) -> dict:
    """Run state-space ES for ONE goal. Dispatches K_real FDFDs per iter
    via fdfd_one.map() so the inner cohort runs in parallel."""
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")

    import numpy as np

    from algorithms.policies.es_state_space_policy import (
        ESStateSpaceCheckpoint, ESStateSpaceConfig, ESStateSpacePolicy,
        MSurrogate, interpolate_anchors, nearest_anchor,
    )

    # --- Unpack ----------------------------------------------------------
    launch_id = payload["launch_id"]
    goal = int(payload["goal"])
    memory_bank_arrays = payload["memory_bank"]
    config_kwargs = payload["config_kwargs"]
    init_mode = payload.get("init_mode", "interp")    # "interp" | "nearest" | "uniform"
    training_indices = list(payload.get("training_indices", list(range(30))))
    m_surrogate_path = payload.get("m_surrogate_path", None)
    m_training_data_bytes = payload.get("m_training_data_bytes", None)
    checkpoint_every = int(payload.get("checkpoint_every", 5))
    wandb_cfg = payload.get("wandb")

    memory_bank = {int(k): np.asarray(v, dtype=np.float32)
                   for k, v in memory_bank_arrays.items()}

    # --- Wandb -----------------------------------------------------------
    wandb_run = None
    if wandb_cfg is not None and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity") or None,
            name=f"phase2-ss-{launch_id}-g{goal:02d}",
            group=launch_id,
            tags=["state-space", f"goal={goal}"],
            config={**config_kwargs, "goal": goal, "init_mode": init_mode,
                    "architecture": "per-goal ES on ε + M filter + nested .map()"},
            reinit=True,
        )

    cfg = ESStateSpaceConfig(**config_kwargs)
    first_eps = next(iter(memory_bank.values()))
    state_shape = tuple(int(x) for x in first_eps.shape)

    # --- M surrogate (optional) -----------------------------------------
    M = None
    if m_surrogate_path is not None:
        M = MSurrogate(m_surrogate_path,
                       lr=cfg.m_train_lr,
                       weight_decay=cfg.m_train_weight_decay)
        if m_training_data_bytes:
            init_data = np.load(io.BytesIO(m_training_data_bytes))
            # Seed M's online buffer if cfg.online_m=True. Otherwise the seed
            # data is unused here (the driver aggregates fresh FDFD pairs and
            # M retraining happens offline post-run).
            if cfg.online_m:
                pass        # MSurrogate doesn't own the buffer; policy does.
        print(f"[g{goal:02d}] M surrogate loaded: {m_surrogate_path}  "
              f"online_m={cfg.online_m}", flush=True)

    policy = ESStateSpacePolicy(state_shape=state_shape,
                                m_surrogate=M, config=cfg)

    # --- Seed online-M buffer (if relevant) -----------------------------
    if cfg.online_m and m_training_data_bytes:
        init_data = np.load(io.BytesIO(m_training_data_bytes))
        # MSurrogate has no buffer; policy._m_buffer_* does. Inject directly.
        for e, p in zip(init_data["eps"], init_data["P"]):
            policy._m_buffer_eps.append(np.asarray(e, dtype=np.float32))
            policy._m_buffer_P.append(np.asarray(p, dtype=np.float64))
        print(f"[g{goal:02d}] online-M buffer seeded with "
              f"{len(policy._m_buffer_eps):,} pairs", flush=True)

    # --- Warm-start ε ---------------------------------------------------
    if init_mode == "interp":
        eps_init = interpolate_anchors(memory_bank, goal)
    elif init_mode == "nearest":
        eps_init = nearest_anchor(memory_bank, goal)
    elif init_mode == "uniform":
        rng_seed = np.random.default_rng(cfg.seed + goal)
        eps_init = rng_seed.uniform(-1.0, 1.0, size=state_shape).astype(np.float32)
    else:
        raise ValueError(f"init_mode={init_mode!r}; expected "
                         f"'interp' | 'nearest' | 'uniform'.")
    print(f"[g{goal:02d}] init_mode={init_mode}  K_cand={cfg.K_cand}  "
          f"K_real={cfg.K_real}  σ={cfg.sigma}  α={cfg.alpha}  "
          f"N_iter={cfg.N_iter}  filter={cfg.filter_mode}", flush=True)

    # --- Resume from checkpoint if present ------------------------------
    out_dir = Path("/buffer") / launch_id / f"goal_{goal:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "state.pkl"
    resume_state = None
    if ckpt_path.exists():
        try:
            resume_state = pickle.loads(ckpt_path.read_bytes())
            print(f"[g{goal:02d}] RESUMING from {ckpt_path} @ iter "
                  f"{resume_state.next_iteration}", flush=True)
        except Exception as e:
            print(f"[g{goal:02d}] checkpoint unreadable ({e}); restarting",
                  flush=True)
            resume_state = None

    # --- FDFD callback: dispatches a batch via fdfd_one.map() ----------
    def fdfd_batch_fn(eps_batch):
        eps_pkls = [pickle.dumps(np.asarray(e, dtype=np.float32))
                    for e in eps_batch]
        results = list(fdfd_one.map(eps_pkls))
        Ps, Plosses = [], []
        for r in results:
            P, P_loss = pickle.loads(r)
            Ps.append(P)
            Plosses.append(P_loss)
        return np.stack(Ps), np.asarray(Plosses, dtype=np.float64)

    # --- Wandb iteration callback ---------------------------------------
    def on_iter(entry):
        if wandb_run is not None:
            import wandb
            wandb.log({
                "iteration": entry["iteration"],
                "fitness/mean": entry["fitness_mean"],
                "fitness/best": entry["fitness_best"],
                "Q_true/mean": entry["Q_true_mean"],
                "Q_true/best": entry["Q_true_best"],
                "target_frac/iter_best": entry["target_frac_best_iter"],
                "target_frac/ever_best": entry["best_ever_target_frac"],
                "Q_true/ever_best": entry["best_ever_Q"],
                "fdfd/n_total": entry["n_fdfd_total"],
                **({"M/train_loss_first": entry["m_train_loss_first"]}
                   if entry["m_train_loss_first"] is not None else {}),
                **({"M/train_loss_last": entry["m_train_loss_last"]}
                   if entry["m_train_loss_last"] is not None else {}),
            }, step=entry["iteration"])
        print(f"[g{goal:02d} iter {entry['iteration']:>3}/{cfg.N_iter}]  "
              f"f_mean={entry['fitness_mean']:+.3e}  "
              f"Q_best_iter={entry['Q_true_best']:.3e}  "
              f"tf_iter={entry['target_frac_best_iter']:.4f}  "
              f"tf_ever={entry['best_ever_target_frac']:.4f}  "
              f"n_fdfd={entry['n_fdfd_total']}", flush=True)

    # --- Checkpoint sink ------------------------------------------------
    def on_ckpt(ckpt: ESStateSpaceCheckpoint):
        ckpt_path.write_bytes(pickle.dumps(ckpt))
        buffer_volume.commit()
        print(f"[g{goal:02d}] checkpoint @ iter {ckpt.next_iteration-1}: "
              f"best_tf={ckpt.best_target_frac:.4f}  "
              f"fdfd={len(ckpt.fdfd_eps)}", flush=True)

    # --- Run -------------------------------------------------------------
    t0 = time.time()
    result = policy.run_one_goal(
        eps_init=eps_init,
        goal=goal,
        training_indices=training_indices,
        fdfd_batch_fn=fdfd_batch_fn,
        on_iteration=on_iter,
        on_checkpoint=on_ckpt,
        checkpoint_every=checkpoint_every,
        resume_state=resume_state,
    )
    elapsed = time.time() - t0
    print(f"[g{goal:02d}] DONE in {elapsed/60:.1f} min  "
          f"best_tf={result.best_target_frac:.4f}  "
          f"best_Q={result.best_Q:.3e}  "
          f"iters={result.iterations}  fdfd={len(result.fdfd_eps)}",
          flush=True)

    # --- Persist final outputs -----------------------------------------
    final_dir = out_dir
    np.save(final_dir / "eps_star.npy", result.eps_star)
    np.save(final_dir / "eps_initial.npy", result.eps_initial)
    np.savez(final_dir / "fdfd_data.npz",
             eps=result.fdfd_eps, P=result.fdfd_P)
    # Closed-loop trajectory — (iterations + 1, N_x, N_y). Consumed by
    # train_phase2_distill_closed_loop.py to extract per-step (ε_t, ε_{t+1},
    # goal) supervision tuples for the distilled neural policy π_φ.
    if result.eps_traj is not None:
        np.save(final_dir / "eps_traj.npy", result.eps_traj)
    (final_dir / "history.json").write_text(json.dumps({
        "goal": goal,
        "init_mode": init_mode,
        "config": config_kwargs,
        "best_target_frac": float(result.best_target_frac),
        "best_Q": float(result.best_Q),
        "iterations": result.iterations,
        "converged": result.converged,
        "elapsed_seconds": elapsed,
        "n_fdfd_total": int(len(result.fdfd_eps)),
        "history": result.history,
    }, indent=2))
    final_state = ESStateSpaceCheckpoint(
        eps_curr=result.eps_star.copy(),
        eps_initial=result.eps_initial.copy(),
        best_eps=result.eps_star.copy(),
        best_Q=result.best_Q,
        best_target_frac=result.best_target_frac,
        history=list(result.history),
        next_iteration=result.iterations,
        rng_state=np.random.default_rng(cfg.seed + goal).bit_generator.state,
        fdfd_eps=list(result.fdfd_eps),
        fdfd_P=list(result.fdfd_P),
        m_state_dict=(policy.M.state_dict()
                      if (policy.M is not None and cfg.online_m) else None),
    )
    ckpt_path.write_bytes(pickle.dumps(final_state))

    # --- Auto-render before/after PNG ----------------------------------
    # Builds env once locally (cheap — ~1s) and runs 2 FDFDs (the initial
    # warm-start ε and the final ε*) to produce a 2×2 viz. Saved next to
    # eps_star.npy on the Volume; downloaded by `collect`. Also logged
    # to the goal's wandb run as an image.
    render_tf_init = render_tf_final = None
    try:
        from render_phase2_ckpt import render_2x2_png
        env_render = _build_pm_env()
        suffix = (f"[state-space-ES] init={init_mode}  "
                  f"K_cand={cfg.K_cand}/K_real={cfg.K_real}  "
                  f"filter={cfg.filter_mode}  "
                  f"best_tf={result.best_target_frac:.3f}  "
                  f"iters={result.iterations}")
        png_path = final_dir / "before_after.png"
        P_i, P_f = render_2x2_png(
            env_render, result.eps_initial, result.eps_star,
            goal, png_path, title_suffix=suffix,
        )
        render_tf_init = float(P_i[goal] / max(P_i.sum(), 1e-9))
        render_tf_final = float(P_f[goal] / max(P_f.sum(), 1e-9))
        print(f"[g{goal:02d}] PNG saved: tf {render_tf_init:.4f} → "
              f"{render_tf_final:.4f}  ({png_path})", flush=True)
    except Exception as e:
        print(f"[g{goal:02d}] PNG render failed: {e}", flush=True)

    buffer_volume.commit()

    if wandb_run is not None:
        import wandb
        log_payload = {
            "final/best_target_frac": float(result.best_target_frac),
            "final/best_Q": float(result.best_Q),
            "final/iterations": result.iterations,
            "final/n_fdfd_total": int(len(result.fdfd_eps)),
            "final/elapsed_minutes": elapsed / 60,
        }
        if render_tf_init is not None:
            log_payload["final/render_tf_init"] = render_tf_init
            log_payload["final/render_tf_final"] = render_tf_final
        # Embed the PNG so it shows up in the wandb run's media tab.
        png_path = final_dir / "before_after.png"
        if png_path.exists():
            log_payload["viz/before_after"] = wandb.Image(
                str(png_path),
                caption=f"goal {goal:02d}  tf={result.best_target_frac:.3f}",
            )
        wandb.log(log_payload)
        wandb_run.finish()

    # Slim summary; bulk data lives on Volume to avoid local BlobGet errors.
    return {
        "goal": goal,
        "best_target_frac": float(result.best_target_frac),
        "best_Q": float(result.best_Q),
        "iterations": result.iterations,
        "converged": result.converged,
        "n_fdfd_total": int(len(result.fdfd_eps)),
        "elapsed_seconds": elapsed,
    }


# =============================================================================
# main — fan out goal_driver.map() across N_goals
# =============================================================================

@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=60 * 60 * 12,           # outer driver — waits on all goal drivers
    volumes={"/buffer": buffer_volume},
    secrets=[wandb_secret] if wandb_secret is not None else [],
)
def main_driver(payload: dict) -> dict:
    """Top-level orchestrator. Fans out one goal_driver per goal via
    Function.map(), aggregates summaries. Bulk per-goal artifacts (eps_star,
    fdfd_data.npz, history.json, state.pkl) live on the Volume."""
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")

    launch_id = payload["launch_id"]
    goal_indices = list(payload["goal_indices"])
    wandb_cfg = payload.get("wandb")
    print(f"[main] launch_id={launch_id}  goals={goal_indices}", flush=True)

    # Top-level summary wandb run; per-goal runs share the same group=launch_id
    # so they all show up grouped in the wandb UI.
    main_wandb = None
    if wandb_cfg is not None and os.environ.get("WANDB_API_KEY"):
        import wandb
        main_wandb = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity") or None,
            name=f"phase2-ss-{launch_id}-MAIN",
            group=launch_id,
            tags=["state-space", "main"],
            config={"launch_id": launch_id, "goal_indices": goal_indices,
                    **payload.get("config_kwargs", {})},
            reinit=True,
        )

    # Build per-goal payloads (just inject goal into the shared payload).
    per_goal_payloads = []
    for g in goal_indices:
        pg = dict(payload)
        pg["goal"] = int(g)
        per_goal_payloads.append(pg)

    t0 = time.time()
    summaries = list(goal_driver.map(per_goal_payloads))
    elapsed = time.time() - t0

    print(f"\n[main] ALL DONE in {elapsed/60:.1f} min", flush=True)
    print(f"{'goal':>4}  {'best_tf':>8}  {'best_Q':>10}  "
          f"{'iters':>5}  {'n_fdfd':>6}  {'conv':>4}  {'wall(min)':>9}")
    print("-" * 75)
    for s in summaries:
        print(f"  {s['goal']:>2}    {s['best_target_frac']:.4f}   "
              f"{s['best_Q']:.3e}    {s['iterations']:>3}    "
              f"{s['n_fdfd_total']:>4}    "
              f"{'✓' if s['converged'] else '✗'}      "
              f"{s['elapsed_seconds']/60:.1f}")
    mean_tf = sum(s["best_target_frac"] for s in summaries) / len(summaries)
    mean_Q = sum(s["best_Q"] for s in summaries) / len(summaries)
    total_fdfd = sum(s["n_fdfd_total"] for s in summaries)
    print("-" * 75)
    print(f"  mean   {mean_tf:.4f}   {mean_Q:.3e}             "
          f"{total_fdfd:>4}             {elapsed/60:.1f}")

    # Log launch-level summary to wandb (per-goal scalars + an aggregated table).
    if main_wandb is not None:
        import wandb
        main_wandb.log({
            "summary/mean_target_frac": float(mean_tf),
            "summary/mean_Q": float(mean_Q),
            "summary/total_fdfd_solves": int(total_fdfd),
            "summary/elapsed_minutes": elapsed / 60,
            "summary/n_goals": len(summaries),
            "summary/n_converged": sum(1 for s in summaries if s["converged"]),
        })
        table = wandb.Table(
            columns=["goal", "best_target_frac", "best_Q",
                     "iterations", "n_fdfd_total", "converged",
                     "elapsed_minutes"],
            data=[[s["goal"], s["best_target_frac"], s["best_Q"],
                   s["iterations"], s["n_fdfd_total"],
                   bool(s["converged"]),
                   s["elapsed_seconds"] / 60] for s in summaries],
        )
        main_wandb.log({"summary/per_goal_table": table})
        main_wandb.finish()

    # Write top-level summary to Volume.
    out_dir = Path("/buffer") / launch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "launch_id": launch_id,
        "goal_indices": goal_indices,
        "mean_target_frac": float(mean_tf),
        "mean_Q": float(mean_Q),
        "total_fdfd_solves": int(total_fdfd),
        "elapsed_seconds": elapsed,
        "per_goal": summaries,
    }, indent=2))
    buffer_volume.commit()

    return {
        "launch_id": launch_id,
        "n_goals": len(summaries),
        "mean_target_frac": float(mean_tf),
        "mean_Q": float(mean_Q),
        "total_fdfd_solves": int(total_fdfd),
        "elapsed_seconds": elapsed,
    }


# =============================================================================
# Collect entrypoint
# =============================================================================

@app.local_entrypoint()
def collect(
    launch_id: str = None,
    out_dir: str = "phase2_state_space_output",
    aggregate_fdfd: bool = True,
):
    """Download per-goal outputs and (optionally) aggregate FDFD data into
    one .npz for offline M retraining.

    Per-goal files on Volume:
        goal_<NN>/eps_star.npy
        goal_<NN>/eps_initial.npy
        goal_<NN>/fdfd_data.npz   (eps: (N, 10, 10), P: (N, 30))
        goal_<NN>/history.json
        goal_<NN>/state.pkl       (final checkpoint)

    Top-level:
        summary.json
        all_fdfd_data.npz         (aggregated; only when --aggregate-fdfd)
    """
    import numpy as np

    if not launch_id:
        raise ValueError("--launch-id required")
    out = Path(out_dir) / launch_id
    out.mkdir(parents=True, exist_ok=True)

    def _try_read(remote: str, dst: Path) -> bool:
        try:
            blob = b"".join(buffer_volume.read_file(remote))
        except Exception as e:
            print(f"  ✗ {remote}: {e}")
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(blob)
        print(f"  ✓ {remote}  ({len(blob)/1024:.1f} KB)")
        return True

    # Top-level summary
    _try_read(f"{launch_id}/summary.json", out / "summary.json")

    # Per-goal artifacts — discover goal dirs from summary.json if present
    goal_indices = []
    summary_path = out / "summary.json"
    if summary_path.exists():
        meta = json.loads(summary_path.read_text())
        goal_indices = list(meta.get("goal_indices", []))
    if not goal_indices:
        print("  (no goal_indices in summary; collecting goal_00..goal_29)")
        goal_indices = list(range(30))

    per_goal_files = ["eps_star.npy", "eps_initial.npy", "eps_traj.npy",
                      "fdfd_data.npz", "history.json", "before_after.png"]
    aggregated_eps, aggregated_P = [], []
    for g in goal_indices:
        gtag = f"goal_{g:02d}"
        any_ok = False
        for fname in per_goal_files:
            ok = _try_read(f"{launch_id}/{gtag}/{fname}",
                           out / gtag / fname)
            any_ok = any_ok or ok
        if not any_ok:
            continue
        if aggregate_fdfd:
            data_path = out / gtag / "fdfd_data.npz"
            if data_path.exists():
                d = np.load(data_path)
                aggregated_eps.append(d["eps"])
                aggregated_P.append(d["P"])

    if aggregate_fdfd and aggregated_eps:
        eps_all = np.concatenate(aggregated_eps, axis=0)
        P_all = np.concatenate(aggregated_P, axis=0)
        np.savez(out / "all_fdfd_data.npz", eps=eps_all, P=P_all)
        print(f"\nAggregated {len(eps_all):,} (ε, P) pairs → "
              f"{(out / 'all_fdfd_data.npz').resolve()}")

    print(f"\nOutputs → {out.resolve()}")
