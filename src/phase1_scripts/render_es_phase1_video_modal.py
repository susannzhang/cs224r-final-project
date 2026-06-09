# =============================================================================
# Phase 1 demo-video renderer on Modal — one MP4 per target angle
# =============================================================================
"""
Records the Phase 1 ES agent's optimization journey for each training angle and
encodes it into an MP4. One Modal container per target; each container rebuilds
the pm_setup environment, runs `train_one_angle` with a per-iteration callback
that renders a 4-panel frame (ε | |E_z|², Re(E_z) | target_frac trace), then
stitches the frames into a video. Frames keep coming until either max_iterations
hit OR best_target_frac plateaus over a rolling window.

Usage:
    modal run render_es_phase1_video_modal.py
    modal run render_es_phase1_video_modal.py --targets "15"
    modal run render_es_phase1_video_modal.py --max-iterations 60 --fps 6
    modal run render_es_phase1_video_modal.py::collect      # download MP4s

Per-target outputs (downloaded to ./phase1_video_output/ after `collect`):
    target_<NN>/demo.mp4              # 4-panel MP4 (one frame per ES iter)
    target_<NN>/frames/iter_<III>.png # individual frames (kept for cherry-pick)
    target_<NN>/metadata.json         # iters rendered, plateau-stop flag, history

Layout per frame:
    ┌─────────────────┬─────────────────┐
    │ Permittivity ε  │  |E_z|² + recv  │
    ├─────────────────┼─────────────────┤
    │   Re(E_z)       │ target_frac     │
    │   + recv        │  (running line) │
    └─────────────────┴─────────────────┘
    Title: target NN  —  iter k/M  —  reward / best_target_frac
    Cyan / lime contour outlines the target receiver mask.

The rendering does one extra FDFD solve per iteration (on top of the K+1
solves the ES loop already does), so per-iter cost is ~5% more than training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = os.environ.get("PHASE1_VIDEO_APP_NAME", "cs224r-phase1-video")
VOLUME_NAME = os.environ.get("PHASE1_VIDEO_VOLUME_NAME", "cs224r-phase1-videos")

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install(
        "numpy",
        "scipy",
        "scikit-image",
        "matplotlib",
        "autograd",
        "ceviche",
        "imageio",
        "imageio-ffmpeg",
        "av",                 # pyav — used by imageio's "pyav" plugin for h264
        "pillow",
    )
    .add_local_dir(PROJECT_ROOT, "/root/app", copy=True,
                   ignore=["__pycache__", "*.pyc", "phase1_checkpoints",
                           "phase1_training_output", "phase1_video_output",
                           "tests/visual_output", ".git", ".venv",
                           ".pytest_cache", "wandb"])
)

app = modal.App(APP_NAME)
videos_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# =============================================================================
# Per-target Modal function
# =============================================================================

@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60 * 6,        # 6 hr ceiling per target; M=80 typically << 1 hr
    retries=modal.Retries(max_retries=1, backoff_coefficient=1.0,
                          initial_delay=10.0),
    volumes={"/buffer": videos_volume},
)
def render_one_target_video(payload: dict) -> dict:
    """One container, one target. Trains ES with a per-iter render callback;
    encodes MP4; writes everything to the Modal Volume.
    """
    import sys as _sys
    _sys.path.insert(0, "/root/app")
    _sys.path.insert(0, "/root/app/dynamic_beam_steering")

    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v3 as iio

    from geometry import (create_design_region, create_grid, create_source,
                          create_receiver, create_environment)
    from simulation import (initialize_environment,
                            simulate_ez_fields_per_source)
    from algorithms.agents.es_agent import (ESAgent, ESAgentConfig,
                                            apply_eps_to_canvas)

    target_idx        = payload["target_idx"]
    training_indices  = payload["training_indices"]
    config_kwargs     = payload["config_kwargs"]
    seed              = payload["seed"]
    fps               = payload["fps"]
    plateau_window    = payload["plateau_window"]
    plateau_tol       = payload["plateau_tol"]
    figsize           = tuple(payload["figsize"])
    dpi               = payload["dpi"]
    # launch_id identifies the user-issued spawn. Used to decide whether a
    # checkpoint on the Volume is "stale from a prior launch" (wipe + start
    # fresh) or "preemption restart of THIS launch" (resume from it).
    launch_id         = payload["launch_id"]
    checkpoint_every  = payload.get("checkpoint_every", 1)
    commit_every      = payload.get("commit_every", 5)

    # --- Build the same env as train_phase1_modal -----------------------
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0,
                                  margin_cells=20)
    grid = create_grid(num_rods_x=10, num_rods_y=10,
                       radius=0.01, distance=0.002, rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
    receivers = []
    for i in range(1, 11):
        receivers.append(create_receiver(index=i,      length=0.02,
                                         side='bottom', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=10 + i, length=0.02,
                                         side='right',  rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=20 + i, length=0.02,
                                         side='top',    rod_index=11 - i))
    env = create_environment(design_region=region, grid=grid,
                             sources=[source], receivers=receivers)
    initialize_environment(env)

    cfg = ESAgentConfig(**config_kwargs, seed=seed)
    agent = ESAgent(env=env, training_indices=training_indices,
                    config=cfg, verbose=False)

    # --- Output staging --------------------------------------------------
    out_root = f"/buffer/target_{target_idx:02d}"
    frames_dir = f"{out_root}/frames"
    os.makedirs(frames_dir, exist_ok=True)
    mp4_path = f"{out_root}/demo.mp4"
    meta_path = f"{out_root}/metadata.json"
    ckpt_path = f"{out_root}/ckpt.pkl"
    sidecar_path = f"{out_root}/plateau_history.json"

    # --- Resume logic (launch_id-keyed) ---------------------------------
    # Same shape as train_phase1_modal.py: if a checkpoint pickle exists
    # from a prior worker run of THIS launch_id, hydrate from it; otherwise
    # wipe the frames dir + checkpoint + sidecar so we start clean.
    import pickle
    from algorithms.agents.es_agent import ESCheckpoint  # noqa: F401
    resume_state = None
    plateau_history: list[float] = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as fh:
            candidate = pickle.load(fh)
        if getattr(candidate, "run_id", None) == launch_id:
            resume_state = candidate
            if os.path.exists(sidecar_path):
                with open(sidecar_path) as fh:
                    plateau_history = json.load(fh)
            print(f"[target {target_idx:02d}] RESUMING from iter "
                  f"{resume_state.next_iteration}/{cfg.M}  "
                  f"({len(plateau_history)} plateau-history entries, "
                  f"{len([f for f in os.listdir(frames_dir) if f.startswith('iter_')])} "
                  f"frames already on volume)",
                  flush=True)

    if resume_state is None:
        # Fresh start OR stale checkpoint from a prior launch — wipe everything.
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"[target {target_idx:02d}] stale ckpt cleared "
                  f"(launch_id mismatch)", flush=True)
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
        for fn in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, fn))
        videos_volume.commit()

    stopped_for_plateau = False

    class PlateauReached(Exception):
        pass

    def _render_frame(eps: np.ndarray,
                      iter_n: int,
                      best_reward: float,
                      best_target_frac: float,
                      target_frac_trace: list[float]) -> np.ndarray:
        """Apply eps, run FDFD once, render 4-panel RGB uint8 frame."""
        apply_eps_to_canvas(env, eps)
        canvas = env.design_region._canvas.copy()
        ez = sum(simulate_ez_fields_per_source(env).values())
        intensity = np.abs(ez) ** 2
        re_ez = np.real(ez)
        target_receiver = env.receivers[target_idx]

        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi,
                                 constrained_layout=True)

        # ── top-left: permittivity ε ────────────────────────────────────
        ax = axes[0, 0]
        clipped = np.clip(canvas, 0, 10)
        im = ax.imshow(clipped, cmap='plasma', origin='lower',
                       vmin=0, vmax=10)
        ax.set_title('Permittivity ε (best so far)')
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, label='ε', shrink=0.85)

        # ── top-right: |E_z|² intensity + target receiver outline ───────
        ax = axes[0, 1]
        vmax = float(np.percentile(intensity, 98)) or 1.0
        im = ax.imshow(intensity, cmap='inferno', origin='lower',
                       vmin=0, vmax=vmax)
        ax.contour(canvas, [3.0, 5e5], colors='white', alpha=0.5,
                   linewidths=0.6)
        ax.contour(target_receiver._mask, [0.5], colors='cyan',
                   linewidths=1.5)
        ax.set_title('Field intensity |E_z|²  (cyan = target)')
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, label='|E_z|²', shrink=0.85)

        # ── bottom-left: Re(E_z), signed ────────────────────────────────
        ax = axes[1, 0]
        abs_max = float(np.max(np.abs(re_ez))) or 1.0
        im = ax.imshow(re_ez, cmap='seismic', origin='lower',
                       vmin=-abs_max, vmax=abs_max)
        ax.contour(canvas, [3.0, 5e5], colors='black', alpha=0.4,
                   linewidths=0.6)
        ax.contour(target_receiver._mask, [0.5], colors='lime',
                   linewidths=1.5)
        ax.set_title('Re(E_z)  (lime = target)')
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, label='Re(E_z)', shrink=0.85)

        # ── bottom-right: target_frac running trace ─────────────────────
        ax = axes[1, 1]
        trace = np.asarray(target_frac_trace, dtype=float)
        iters = np.arange(1, len(trace) + 1)
        ax.plot(iters, trace, lw=2, color='C0')
        ax.scatter([iters[-1]], [trace[-1]], color='C0', s=40, zorder=5)
        ax.axhline(1.0 - cfg.eta, color='gray', ls='--', lw=0.8,
                   label=f'η-stop (1−η = {1.0 - cfg.eta:.2f})')
        ax.set_xlabel('iteration')
        ax.set_ylabel('best target_frac')
        ax.set_xlim(0, cfg.M)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_title(f'target_frac (best so far: {best_target_frac:.3f})')

        fig.suptitle(
            f'ES Phase 1  —  target {target_idx:02d}  —  '
            f'iter {iter_n}/{cfg.M}  —  reward {best_reward:+.2e}',
            fontsize=13,
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:, :, :3].copy()
        plt.close(fig)
        # h264 needs even dimensions — trim odd rows/cols if necessary.
        h, w = rgb.shape[:2]
        if h % 2: rgb = rgb[:-1]
        if w % 2: rgb = rgb[:, :-1]
        return rgb

    def _on_checkpoint(ckpt):
        nonlocal stopped_for_plateau
        # Iteration index that JUST completed (next_iteration is the index the
        # resumed loop would start at, so the completed iter is next_iteration - 1).
        iter_n = ckpt.next_iteration
        plateau_history.append(float(ckpt.best_target_frac))

        # Render + write PNG idempotently. If the frame already exists on the
        # Volume (e.g. a prior worker run of THIS launch_id rendered it before
        # being preempted), skip the FDFD + matplotlib work entirely. Resume
        # is deterministic (RNG state is restored), so a regenerated frame at
        # iter k would be identical to the saved one anyway.
        frame_path = f"{frames_dir}/iter_{iter_n:03d}.png"
        if not os.path.exists(frame_path):
            rgb = _render_frame(
                eps=ckpt.best_eps,
                iter_n=iter_n,
                best_reward=float(ckpt.best_reward),
                best_target_frac=float(ckpt.best_target_frac),
                target_frac_trace=plateau_history,
            )
            from PIL import Image as _PILImage
            _PILImage.fromarray(rgb).save(frame_path, optimize=True)

        # Persist resume state every iter (cheap — ckpt is small, sidecar is
        # a tiny JSON list). Commit to Volume periodically; full commit on
        # every iter would add latency.
        ckpt.run_id = launch_id
        with open(ckpt_path, "wb") as fh:
            pickle.dump(ckpt, fh)
        with open(sidecar_path, "w") as fh:
            json.dump(plateau_history, fh)
        if (iter_n % commit_every == 0) or iter_n <= 3 or iter_n == cfg.M:
            videos_volume.commit()

        # Plateau: max best_target_frac over the most recent `plateau_window`
        # iters minus the max over the prior window < tolerance → stop.
        # Need at least 2*window samples before we can compare windows.
        if len(plateau_history) >= 2 * plateau_window:
            recent = max(plateau_history[-plateau_window:])
            prior  = max(plateau_history[-2 * plateau_window:-plateau_window])
            if (recent - prior) < plateau_tol:
                stopped_for_plateau = True
                videos_volume.commit()    # flush last frame + sidecar
                print(f"[target {target_idx:02d}] plateau @ iter {iter_n}: "
                      f"max({recent:.3f}) - prior_max({prior:.3f}) "
                      f"= {recent - prior:+.4f} < tol {plateau_tol}",
                      flush=True)
                raise PlateauReached

        if iter_n % 5 == 0 or iter_n == 1:
            print(f"[target {target_idx:02d}] frame {iter_n}/{cfg.M}  "
                  f"best_target_frac={ckpt.best_target_frac:.3f}  "
                  f"best_reward={ckpt.best_reward:+.3e}", flush=True)

    # --- Run ESagent; the on_checkpoint hook drives rendering -----------
    history: list[dict] = []
    iterations_run = 0
    try:
        result = agent.train_one_angle(
            target_idx,
            on_checkpoint=_on_checkpoint,
            checkpoint_every=checkpoint_every,
            resume_state=resume_state,
        )
        history = result.history
        iterations_run = result.iterations
        converged_eta = result.converged
        best_reward_final = float(result.best_reward)
    except PlateauReached:
        converged_eta = False
        iterations_run = len(plateau_history)
        best_reward_final = float(plateau_history[-1]) if plateau_history else float('nan')

    # --- Encode MP4 from PNGs on the Volume -----------------------------
    # Read frames in iter order from /buffer/target_NN/frames/iter_NNN.png.
    # This is robust to retries: frames written by prior worker runs of
    # THIS launch survive because the resume path doesn't wipe them.
    from PIL import Image as _PILImage
    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.startswith("iter_") and f.endswith(".png")
    )
    if not frame_files:
        raise RuntimeError(
            f"target {target_idx:02d}: no PNG frames on volume "
            f"(checkpoint callback never fired)."
        )
    loaded = []
    for fn in frame_files:
        img = np.asarray(_PILImage.open(f"{frames_dir}/{fn}"))
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        loaded.append(img)
    video_array = np.stack(loaded)
    print(f"[target {target_idx:02d}] encoding MP4: "
          f"{len(loaded)} frames @ {fps} fps  shape={video_array.shape}",
          flush=True)
    iio.imwrite(mp4_path, video_array, fps=fps, codec="h264", plugin="pyav")

    # --- Per-target metadata --------------------------------------------
    metadata = {
        "target_idx": target_idx,
        "frames_rendered": len(loaded),
        "iterations_run": iterations_run,
        "stopped_for_plateau": stopped_for_plateau,
        "converged_eta": converged_eta,
        "best_reward_final": best_reward_final,
        "best_target_frac_final": (plateau_history[-1]
                                   if plateau_history else None),
        "target_frac_history": plateau_history,
        "history": history,
        "config_kwargs": config_kwargs,
        "fps": fps,
        "plateau_window": plateau_window,
        "plateau_tol": plateau_tol,
        "launch_id": launch_id,
    }
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    # Run complete → clear the resume checkpoint + sidecar so a future
    # launch starts clean (the metadata + frames + mp4 still live on the
    # volume for collect to download).
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if os.path.exists(sidecar_path):
        os.remove(sidecar_path)
    videos_volume.commit()

    mp4_size = os.path.getsize(mp4_path)
    print(f"[✓ target {target_idx:02d}] {len(loaded)} frames  "
          f"plateau={stopped_for_plateau}  "
          f"final_target_frac={metadata['best_target_frac_final']}  "
          f"mp4={mp4_size / 1024:.0f} KB", flush=True)

    return {
        "target_idx": target_idx,
        "frames_rendered": len(loaded),
        "iterations_run": iterations_run,
        "stopped_for_plateau": stopped_for_plateau,
        "best_reward_final": best_reward_final,
        "best_target_frac_final": metadata["best_target_frac_final"],
        "mp4_volume_path": mp4_path,
        "meta_volume_path": meta_path,
        "frames_volume_dir": frames_dir,
    }


# =============================================================================
# Debug entrypoint — synchronous .remote() so worker exceptions surface inline
# =============================================================================

@app.local_entrypoint()
def debug(
    target_idx: int = 15,
    population_size: int = 20,
    max_iterations: int = 12,
    fps: int = 4,
    plateau_window: int = 5,
    plateau_tol: float = 0.005,
):
    """Synchronous single-target run for debugging. Streams the worker's stdout
    and re-raises any exception with full traceback locally."""
    config_kwargs = dict(
        K=population_size, sigma=0.1, alpha_1=0.05, M=max_iterations,
        eta=1e-2, log_every=1, K_elite=None,
        w_crosstalk=0.3, w_loss=1e-3, w_energy=0.1,
    )
    from datetime import datetime
    payload = {
        "target_idx": target_idx,
        "training_indices": list(range(0, 30, 3)),
        "config_kwargs": config_kwargs,
        "seed": target_idx,
        "fps": fps,
        "plateau_window": plateau_window,
        "plateau_tol": plateau_tol,
        "figsize": [12.0, 10.0],
        "dpi": 90,
        "launch_id": "video-debug-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        "checkpoint_every": 1,
        "commit_every": 5,
    }
    print(f"[debug] calling render_one_target_video.remote(target={target_idx}, M={max_iterations})")
    result = render_one_target_video.remote(payload)
    print("[debug] result:", result)


# =============================================================================
# Local entrypoint — fan-out via spawn(), then exit. `collect` downloads later.
# =============================================================================

@app.local_entrypoint()
def main(
    targets: str = None,                # "0,3,15"; default = every 3rd of 30
    population_size: int = 20,          # K — matches train_phase1_modal default
    max_iterations: int = 80,           # M — user asked for "80 or so"
    sigma: float = 0.1,
    learning_rate: float = 0.05,
    eta: float = 1e-2,
    k_elite: int = None,
    w_crosstalk: float = 0.3,
    w_loss: float = 1e-3,
    w_energy: float = 0.1,
    fps: int = 4,
    plateau_window: int = 10,
    plateau_tol: float = 0.005,
    fig_width: float = 12.0,
    fig_height: float = 10.0,
    dpi: int = 90,
    checkpoint_every: int = 1,           # write resume ckpt every N iters
    commit_every: int = 5,                # flush Volume every N iters
    launch_id: str = "",                  # explicit launch_id to resume a prior spawn
    out_dir: str = "phase1_video_output",
):
    """Spawn one Modal container per target. Returns immediately with the
    spawn IDs written to a local registry; use `collect` to pull MP4s once
    workers finish (or while in flight — partial results are fine).

    Each spawn carries a `launch_id`. If a Modal container retry (Modal's
    own preemption-restart mechanism) fires, the new worker finds the prior
    ckpt on the Volume, sees matching launch_id, and resumes from the last
    saved ES iter — preserving the PNG frames the prior worker rendered.
    Pass --launch-id explicitly to re-spawn against an existing partial run
    (e.g. after the spawn handle expired); leave blank for a fresh launch.
    """
    if targets is None:
        target_indices = list(range(0, 30, 3))   # 10 evenly-spaced angles
    else:
        target_indices = [int(x) for x in targets.split(",")]
    training_indices = list(target_indices)

    config_kwargs = dict(
        K=population_size,
        sigma=sigma,
        alpha_1=learning_rate,
        M=max_iterations,
        eta=eta,
        log_every=1,
        K_elite=k_elite,
        w_crosstalk=w_crosstalk,
        w_loss=w_loss,
        w_energy=w_energy,
    )

    from datetime import datetime
    if not launch_id:
        launch_id = "video-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"Phase 1 video render on Modal: {len(target_indices)} targets")
    print(f"  modal app:    {APP_NAME}  "
          f"({'env override' if 'PHASE1_VIDEO_APP_NAME' in os.environ else 'default'})")
    print(f"  modal volume: {VOLUME_NAME}  "
          f"({'env override' if 'PHASE1_VIDEO_VOLUME_NAME' in os.environ else 'default'})")
    print(f"  local out:    {Path(out_dir).resolve()}")
    print(f"  config: K={population_size}  M={max_iterations}  "
          f"σ={sigma}  α_1={learning_rate}  η={eta}")
    print(f"  video:  fps={fps}  figsize=({fig_width}, {fig_height})  dpi={dpi}")
    print(f"  plateau: window={plateau_window} tol={plateau_tol}")
    print(f"  resume: checkpoint_every={checkpoint_every}  commit_every={commit_every}")
    print(f"  launch_id: {launch_id}")
    print(f"  targets: {target_indices}")
    print()

    payloads = [
        {
            "target_idx": idx,
            "training_indices": training_indices,
            "config_kwargs": config_kwargs,
            "seed": seed,
            "fps": fps,
            "plateau_window": plateau_window,
            "plateau_tol": plateau_tol,
            "figsize": [fig_width, fig_height],
            "dpi": dpi,
            "launch_id": launch_id,
            "checkpoint_every": checkpoint_every,
            "commit_every": commit_every,
        }
        for seed, idx in enumerate(target_indices)
    ]

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    spawned = []
    # Spawn against the DEPLOYED function, not the ephemeral local reference.
    # Spawns from a `modal run` ephemeral app context get torn down with the
    # local entrypoint; the deployed function ref persists.
    deployed_fn = modal.Function.from_name(APP_NAME, "render_one_target_video")
    print(f"Spawning Modal function calls (deployed app: {APP_NAME})...")
    for payload in payloads:
        fc = deployed_fn.spawn(payload)
        spawned.append({
            "target_idx": payload["target_idx"],
            "function_call_id": fc.object_id,
        })
        print(f"  ✓ target {payload['target_idx']:02d} → {fc.object_id}")

    with open(out / "spawned_calls.json", "w") as fh:
        json.dump({
            "config_kwargs": config_kwargs,
            "training_indices": training_indices,
            "fps": fps,
            "plateau_window": plateau_window,
            "plateau_tol": plateau_tol,
            "figsize": [fig_width, fig_height],
            "dpi": dpi,
            "launch_id": launch_id,
            "checkpoint_every": checkpoint_every,
            "commit_every": commit_every,
            "spawned": spawned,
        }, fh, indent=2)

    print()
    print(f"All {len(spawned)} workers running independently on Modal.")
    print(f"Spawn IDs → {(out / 'spawned_calls.json').resolve()}")
    print()
    print("Once they finish (or for partial results):")
    print("  modal run render_es_phase1_video_modal.py::collect")


# =============================================================================
# Collect entrypoint — pull MP4 + PNG frames + metadata from the Volume
# =============================================================================

@app.local_entrypoint()
def collect(
    out_dir: str = "phase1_video_output",
    targets: str = None,
    skip_frames: bool = False,      # set True if you only want the MP4s, not the PNGs
):
    """Download MP4s + per-frame PNGs from the Volume. Safe to run while
    workers are in flight (in-progress targets are reported, not skipped)."""
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    if targets is not None:
        target_indices = [int(x) for x in targets.split(",")]
    else:
        spawn_path = out / "spawned_calls.json"
        if not spawn_path.exists():
            raise FileNotFoundError(
                f"{spawn_path} not found. Either pass --targets explicitly "
                f"or run `main` first."
            )
        with open(spawn_path) as fh:
            target_indices = [s["target_idx"]
                              for s in json.load(fh)["spawned"]]

    print(f"Collecting {len(target_indices)} targets from Volume...")
    summary = []
    for target_idx in target_indices:
        d = out / f"target_{target_idx:02d}"
        d.mkdir(exist_ok=True)
        prefix = f"target_{target_idx:02d}"

        # MP4 — the headline artifact.
        mp4_local = d / "demo.mp4"
        try:
            blob = b"".join(
                videos_volume.read_file(f"{prefix}/demo.mp4")
            )
            mp4_local.write_bytes(blob)
            mp4_status = f"{len(blob) / 1024:.0f} KB"
        except Exception:
            mp4_status = "(not yet)"

        # Metadata.
        meta = {}
        meta_local = d / "metadata.json"
        try:
            blob = b"".join(
                videos_volume.read_file(f"{prefix}/metadata.json")
            )
            meta_local.write_bytes(blob)
            meta = json.loads(blob.decode("utf-8"))
        except Exception:
            pass

        # PNG frames (optional).
        n_pngs = 0
        if not skip_frames:
            frames_local = d / "frames"
            frames_local.mkdir(exist_ok=True)
            try:
                for entry in videos_volume.iterdir(f"{prefix}/frames"):
                    fname = Path(entry.path).name
                    if not fname.endswith(".png"):
                        continue
                    blob = b"".join(
                        videos_volume.read_file(f"{prefix}/frames/{fname}")
                    )
                    (frames_local / fname).write_bytes(blob)
                    n_pngs += 1
            except Exception:
                pass

        summary.append({
            "target_idx": target_idx,
            "mp4_status": mp4_status,
            "frames_rendered": meta.get("frames_rendered"),
            "stopped_for_plateau": meta.get("stopped_for_plateau"),
            "best_target_frac_final": meta.get("best_target_frac_final"),
            "pngs_downloaded": n_pngs,
        })
        flag = "✓" if mp4_status != "(not yet)" else "·"
        print(f"  {flag} target {target_idx:02d}: mp4={mp4_status}  "
              f"frames={meta.get('frames_rendered')}  "
              f"plateau={meta.get('stopped_for_plateau')}  "
              f"final_target_frac={meta.get('best_target_frac_final')}  "
              f"pngs={n_pngs}")

    with open(out / "summary.json", "w") as fh:
        json.dump({"results": summary}, fh, indent=2)
    n_done = sum(1 for s in summary if s["mp4_status"] != "(not yet)")
    print(f"\n{n_done}/{len(target_indices)} MP4s downloaded → {out.resolve()}")
