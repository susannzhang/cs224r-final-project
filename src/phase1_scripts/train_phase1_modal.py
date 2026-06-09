# =============================================================================
# Phase 1 Training on Modal — parallel ES on cloud containers
# =============================================================================
"""
Cloud-fanned version of train_phase1.py. One Modal container per target angle;
each container reconstructs the pm_setup environment, trains ε*, generates the
2x2 before/after viz, and writes its transitions to a shared Modal Volume.
Results stream back to your local machine.

Usage:
    modal run train_phase1_modal.py
    modal run train_phase1_modal.py --targets "10,13,16,19"
    modal run train_phase1_modal.py --population-size 4 --max-iterations 10  # smoke test
    modal run train_phase1_modal.py --population-size 20 --max-iterations 500 # full run

Outputs land in ./phase1_training_output/ after all workers complete:
    target_<NN>/eps_star.npy        # converged ε*(θ)  [MEMORY BANK]
    target_<NN>/eps_initial.npy     # random ε the inner loop started from
    target_<NN>/before_after.png    # 2x2 viz (matches test_es_agent.py layout)
    target_<NN>/metadata.json       # convergence + history + P_initial/P_final
    replay_buffer.pkl               # merged transitions (downloaded from Volume)
    summary.json                    # per-target convergence report

Transitions are persisted to a Modal Volume named "cs224r-phase1-buffer" so
they survive container shutdown. The volume is reused/overwritten across runs;
clear it manually with:
    modal volume rm cs224r-phase1-buffer

Prerequisites:
    pip install modal
    modal token set ...                  # first-time auth
"""

import json
import os
import pickle
from pathlib import Path

import modal

# App + volume names are env-overridable so you can deploy an isolated
# parallel experiment (different code, different ckpts, different wandb
# project) without disturbing a currently-running deployment. Example:
#   PHASE1_APP_NAME=cs224r-phase1-uniform-init \
#   PHASE1_VOLUME_NAME=cs224r-phase1-uniform-init-buffer \
#   python -m modal deploy train_phase1_modal.py
APP_NAME = os.environ.get("PHASE1_APP_NAME", "cs224r-phase1-es")
VOLUME_NAME = os.environ.get("PHASE1_VOLUME_NAME", "cs224r-phase1-buffer")

# =============================================================================
# Modal image — mirror what the local conda env has, plus the repo source
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root (for geometry, simulation)

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
        "wandb",
    )
    # torch CPU-only wheel (Modal containers here are CPU-only; CUDA build
    # pulls in ~3 GB of NVIDIA deps we don't use).
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    # Drop the whole repo into the container at /root/app so the worker can
    # import geometry / simulation / algorithms / train_phase1 (for the viz helper).
    .add_local_dir(PROJECT_ROOT, "/root/app", copy=True,
                   ignore=["__pycache__", "*.pyc", "phase1_checkpoints",
                           "phase1_training_output", "tests/visual_output",
                           ".git", ".venv", ".pytest_cache"])
)

app = modal.App(APP_NAME)

# Shared persistent volume so transitions outlive containers. Per-target files
# at /buffer/target_NN.pkl. Reused across runs (overwrites each invocation).
buffer_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Modal Secret named "wandb" carrying the env var WANDB_API_KEY.
# (Created via the Modal dashboard or `modal secret create wandb WANDB_API_KEY=<key>`.)
# The worker reads os.environ["WANDB_API_KEY"] at runtime; if the secret is
# missing or the var isn't present, wandb logging is silently skipped and the
# run still produces local checkpoints + viz + replay buffer.
try:
    wandb_secret = modal.Secret.from_name("wandb")
except modal.exception.NotFoundError:
    wandb_secret = None


# =============================================================================
# Per-target Modal function
# =============================================================================

@app.function(
    image=image,
    cpu=4,
    memory=8192,                # FDFD on a ~476x476 canvas + K candidates fits comfortably in 8 GB
    timeout=60 * 60 * 24,       # 24 hours per container — Modal's max
    # Auto-retry on timeout. The worker resumes from the latest checkpoint
    # on the Volume (matched by launch_id), so each retry continues where
    # the last one left off. 2 retries × 24h covers K=20 / M=250 (~35h).
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0,
                          initial_delay=10.0),
    volumes={"/buffer": buffer_volume},
    secrets=[wandb_secret] if wandb_secret is not None else [],
)
def train_one_target_modal(payload: dict) -> dict:
    """One container, one target angle.
    Reconstructs env → runs ES → renders viz → persists transitions to Volume
    → returns ε*, eps_initial, PNG bytes, and summary metadata to the entrypoint.
    """
    import sys, os, io
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")

    from geometry import (create_design_region, create_grid, create_source,
                          create_receiver, create_environment)
    from simulation import initialize_environment
    from algorithms.agents.es_agent import ESAgent, ESAgentConfig
    # Reuse the same 2x2 viz helper as the local runner.
    from train_phase1 import _render_before_after_png

    target_idx = payload["target_idx"]
    training_indices = payload["training_indices"]
    config_kwargs = payload["config_kwargs"]
    seed = payload["seed"]
    wandb_cfg = payload.get("wandb")  # {project, entity, run_id} or None
    use_critic = payload.get("use_critic", False)
    # launch_id identifies this user-issued `modal run` invocation. Used to
    # decide whether an existing checkpoint on the Volume is "fresh launch,
    # discard" or "preemption restart of THIS launch, resume".
    launch_id = payload["launch_id"]
    checkpoint_every = payload.get("checkpoint_every", 5)

    # --- Initialize wandb (one run per target, grouped by run_id) ------
    # If WANDB_API_KEY isn't in the env (no secret mounted), this no-ops.
    wandb_run = None
    if wandb_cfg is not None and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity") or None,
            name=f"target-{target_idx:02d}",
            group=wandb_cfg["run_id"],
            tags=[f"target-{target_idx:02d}", f"K={config_kwargs['K']}",
                  f"M={config_kwargs['M']}"],
            config={
                **config_kwargs,
                "target_idx": target_idx,
                "training_indices": training_indices,
                "seed": seed,
            },
            reinit=True,
        )

    # --- Build pm_setup env (must match pm_setup.py exactly) -----------
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0,
                                  margin_cells=20)
    grid = create_grid(num_rods_x=10, num_rods_y=10,
                       radius=0.01, distance=0.002, rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)

    # 30 receivers, indexed to ascend with angle (CCW from bottom-left to top-left).
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

    # --- Optional critic (interleaved TD(0) updates during ES) ---------
    critic = None
    if use_critic:
        from algorithms.critics.dqn_critic import DQNCritic, DQNCriticConfig
        critic = DQNCritic(
            state_shape=(env.grid.num_rods_x, env.grid.num_rods_y),
            config=DQNCriticConfig(n_goals=len(env.receivers), seed=seed),
        )

    # --- Train ----------------------------------------------------------
    cfg = ESAgentConfig(**config_kwargs, seed=seed)
    agent = ESAgent(env=env, training_indices=training_indices,
                    config=cfg, critic=critic, verbose=False)

    # --- Checkpoint resume logic ---------------------------------------
    # Files written by previous container runs of THIS launch (matched on
    # launch_id) are loaded back so a preemption / timeout restart picks
    # up where the prior container left off.
    #   /buffer/target_NN_ckpt.pkl       — ESCheckpoint (small)
    #   /buffer/target_NN_chunk_<III>.pkl — transitions added in window III
    from algorithms.agents.es_agent import ESCheckpoint
    ckpt_path = f"/buffer/target_{target_idx:02d}_ckpt.pkl"
    resume_state = None
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as fh:
            candidate: ESCheckpoint = pickle.load(fh)
        if candidate.run_id == launch_id:
            resume_state = candidate
            # Hydrate the in-memory buffer from chunk files written before
            # the prior container died.
            chunk_pattern = f"target_{target_idx:02d}_chunk_"
            chunk_files = sorted(
                fn for fn in os.listdir("/buffer")
                if fn.startswith(chunk_pattern) and fn.endswith(".pkl")
            )
            n_loaded = 0
            for fn in chunk_files:
                with open(f"/buffer/{fn}", "rb") as fh:
                    chunk = pickle.load(fh)
                agent.buffer.extend(chunk)
                n_loaded += len(chunk)
            print(f"[target {target_idx:02d}] RESUMING from iter "
                  f"{resume_state.next_iteration}/{cfg.M}  "
                  f"({len(chunk_files)} chunks, {n_loaded} transitions restored)",
                  flush=True)
        else:
            # Stale checkpoint from a different launch — clear it and any
            # leftover chunks so we start fresh.
            os.remove(ckpt_path)
            for fn in os.listdir("/buffer"):
                if fn.startswith(f"target_{target_idx:02d}_chunk_") \
                        and fn.endswith(".pkl"):
                    os.remove(f"/buffer/{fn}")
            buffer_volume.commit()
            print(f"[target {target_idx:02d}] stale ckpt ({candidate.run_id} "
                  f"!= {launch_id}) cleared; fresh start", flush=True)

    # Live wandb logging: streams each history entry as the agent appends it,
    # so dashboards update during training instead of in one batch at the end.
    def _on_iter(entry):
        if wandb_run is None:
            return
        import wandb
        critic_loss = entry.get("critic_loss")
        wandb.log({
            "iteration": entry["iteration"],
            "pop_reward/mean": entry["pop_reward_mean"],
            "pop_reward/best": entry["pop_reward_best"],
            "pop_target_fraction/best": entry["pop_target_fraction_best"],
            "best_ever/reward": entry["best_ever_reward"],
            "best_ever/target_fraction": entry["best_ever_target_fraction"],
            "target_power_concentration_pct":
                100.0 * entry["pop_target_fraction_best"],
            **({"critic_loss": critic_loss} if critic_loss is not None else {}),
        }, step=entry["iteration"])

    # Incremental checkpoint callback. Each fire:
    #   1) writes the new transitions (since last fire) to a fresh chunk file
    #   2) overwrites the small ESCheckpoint pickle
    #   3) commits the Volume so the next container can read them
    # buf_persisted_idx is the watermark of how many transitions are already
    # on disk; we slice the tail above it for each new chunk.
    buf_persisted_idx = len(agent.buffer.transitions)
    def _on_checkpoint(ckpt):
        nonlocal buf_persisted_idx
        ckpt.run_id = launch_id
        new_transitions = agent.buffer.transitions[buf_persisted_idx:]
        chunk_path = (f"/buffer/target_{target_idx:02d}_chunk_"
                      f"{ckpt.chunk_idx:03d}.pkl")
        with open(chunk_path, "wb") as fh:
            pickle.dump(new_transitions, fh)
        with open(ckpt_path, "wb") as fh:
            pickle.dump(ckpt, fh)
        buffer_volume.commit()
        buf_persisted_idx = len(agent.buffer.transitions)
        print(f"[target {target_idx:02d}] ckpt @ iter {ckpt.next_iteration} "
              f"(+{len(new_transitions)} transitions → chunk {ckpt.chunk_idx:03d})",
              flush=True)

    result = agent.train_one_angle(
        target_idx,
        on_iteration=_on_iter,
        on_checkpoint=_on_checkpoint,
        checkpoint_every=checkpoint_every,
        resume_state=resume_state,
    )
    transitions = list(agent.buffer.transitions)

    # --- Persist EVERYTHING to the shared Volume ------------------------
    # Belt-and-suspenders: write critical artifacts to the Volume in addition
    # to returning them through the function dict, so if the local entrypoint
    # crashes during .map() (e.g. Modal app state transitions), we can still
    # recover memory bank + viz from the Volume.
    buf_path = f"/buffer/target_{target_idx:02d}.pkl"
    with open(buf_path, "wb") as fh:
        pickle.dump(transitions, fh)
    # Memory bank entry (THE most important Phase 2 input).
    import numpy as _np
    _np.save(f"/buffer/target_{target_idx:02d}_eps_star.npy", result.eps_star)
    if result.eps_initial is not None:
        _np.save(f"/buffer/target_{target_idx:02d}_eps_initial.npy", result.eps_initial)
    # Critic (only when --use-critic is on).
    critic_path = None
    if critic is not None:
        critic_path = f"/buffer/target_{target_idx:02d}_critic.pt"
        critic.save(critic_path)
    buffer_volume.commit()

    # --- Render 2x2 before/after viz (2 extra FDFD solves) -------------
    viz_png_bytes = None
    P_initial = P_final = None
    if result.eps_initial is not None:
        viz_png_bytes, P_initial_arr, P_final_arr = _render_before_after_png(
            env, result.eps_initial, result.eps_star, target_idx, cfg, result,
        )
        P_initial = P_initial_arr.tolist()
        P_final = P_final_arr.tolist()

    # --- Second Volume commit: viz + per-target metadata --------------
    # So that if the local entrypoint crashes after .map() collects results,
    # we can still recover the PNG + receiver-power JSON from the Volume.
    if viz_png_bytes is not None:
        with open(f"/buffer/target_{target_idx:02d}_before_after.png", "wb") as fh:
            fh.write(viz_png_bytes)
    # Lightweight per-target summary in case metadata.json never gets written locally.
    import json as _json
    with open(f"/buffer/target_{target_idx:02d}_meta.json", "w") as fh:
        _json.dump({
            "target_idx": target_idx,
            "best_reward": result.best_reward,
            "iterations": result.iterations,
            "converged": result.converged,
            "history": result.history,
            "P_initial": P_initial,
            "P_final": P_final,
        }, fh, indent=2)
    # Training fully completed → clear the resume checkpoint + chunk files
    # so they don't confuse a future launch. (The final transitions still
    # live in target_NN.pkl from the first commit above, and the local
    # entrypoint downloads them from there.)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    for fn in os.listdir("/buffer"):
        if fn.startswith(f"target_{target_idx:02d}_chunk_") \
                and fn.endswith(".pkl"):
            os.remove(f"/buffer/{fn}")
    buffer_volume.commit()

    # --- One-line container log so you can watch progress live in modal logs --
    target_frac_str = ""
    frac_i = frac_f = 0.0
    if P_initial is not None:
        ti = sum(P_initial); tf = sum(P_final)
        frac_i = P_initial[target_idx] / ti if ti > 0 else 0.0
        frac_f = P_final[target_idx] / tf if tf > 0 else 0.0
        target_frac_str = f"  target_frac: {frac_i:.3f} → {frac_f:.3f}"
    status = "✓" if result.converged else "·"
    print(f"[{status} target {target_idx:02d}] iter={result.iterations:>4}  "
          f"reward={result.best_reward:+.3e}{target_frac_str}  "
          f"buffer={len(transitions)} transitions → {buf_path}",
          flush=True)

    # --- wandb logging: final summary + viz + powers ------------------
    # (Per-iteration curves are already streamed live via _on_iter above.)
    if wandb_run is not None:
        import wandb

        # Final summary scalars (single point at the last step).
        wandb.log({
            "final/converged": int(result.converged),
            "final/iterations": result.iterations,
            "final/best_reward": result.best_reward,
            "final/target_fraction_initial": frac_i,
            "final/target_fraction_final": frac_f,
            "final/n_transitions": len(transitions),
        })

        # 2x2 before/after PNG as a wandb Image — wandb.Image needs a PIL
        # Image / numpy array / file path, not a BytesIO.
        if viz_png_bytes is not None:
            from PIL import Image as _PILImage
            pil_img = _PILImage.open(io.BytesIO(viz_png_bytes))
            wandb.log({"before_after": wandb.Image(pil_img)})

        # Per-receiver power table for bar-chart visualizations on the dashboard.
        if P_initial is not None:
            table = wandb.Table(
                columns=["receiver", "is_target", "P_initial", "P_final"],
                data=[[i, i == target_idx, P_initial[i], P_final[i]]
                      for i in range(len(P_initial))],
            )
            wandb.log({"receiver_powers": table})

        wandb_run.finish()

    return {
        "target_idx": target_idx,
        "eps_star": result.eps_star,
        "eps_initial": result.eps_initial,
        "best_reward": result.best_reward,
        "iterations": result.iterations,
        "converged": result.converged,
        "history": result.history,
        "n_transitions": len(transitions),
        "viz_png_bytes": viz_png_bytes,
        "P_initial": P_initial,
        "P_final": P_final,
        "critic_volume_path": critic_path,   # None if --no-use-critic
    }


# =============================================================================
# Local entrypoint — fans out + writes results to local disk
# =============================================================================

def _read_volume_bytes(path: str) -> bytes:
    """Stream a file out of the Modal Volume into a single bytes blob."""
    return b"".join(buffer_volume.read_file(path))


def _recover_from_volume(target_idx: int) -> dict | None:
    """
    Reconstruct a worker result dict from artifacts the worker saved to the
    Modal Volume. Used when the worker function call itself returned an
    exception (e.g. due to a Modal-side hiccup), but the worker had already
    committed its outputs.

    Returns None if nothing for this target is on the Volume.
    """
    import numpy as _np
    import json as _json
    prefix = f"target_{target_idx:02d}"

    # Critical: eps_star. If missing, we can't recover (this run pre-dated
    # the volume-save fix).
    try:
        eps_star_bytes = _read_volume_bytes(f"{prefix}_eps_star.npy")
    except Exception:
        return None
    import io as _io
    eps_star = _np.load(_io.BytesIO(eps_star_bytes))

    # Optional: eps_initial.
    eps_initial = None
    try:
        eps_initial_bytes = _read_volume_bytes(f"{prefix}_eps_initial.npy")
        eps_initial = _np.load(_io.BytesIO(eps_initial_bytes))
    except Exception:
        pass

    # Optional: viz PNG.
    viz_png_bytes = None
    try:
        viz_png_bytes = _read_volume_bytes(f"{prefix}_before_after.png")
    except Exception:
        pass

    # Optional: metadata (history, P_initial, P_final, etc.).
    meta = {}
    try:
        meta_bytes = _read_volume_bytes(f"{prefix}_meta.json")
        meta = _json.loads(meta_bytes.decode("utf-8"))
    except Exception:
        pass

    # Count transitions to set n_transitions.
    try:
        trans_bytes = _read_volume_bytes(f"{prefix}.pkl")
        n_transitions = len(pickle.loads(trans_bytes))
    except Exception:
        n_transitions = 0

    return {
        "target_idx": target_idx,
        "eps_star": eps_star,
        "eps_initial": eps_initial,
        "best_reward": meta.get("best_reward", float("nan")),
        "iterations": meta.get("iterations", 0),
        "converged": meta.get("converged", False),
        "history": meta.get("history", []),
        "n_transitions": n_transitions,
        "viz_png_bytes": viz_png_bytes,
        "P_initial": meta.get("P_initial"),
        "P_final": meta.get("P_final"),
        "critic_volume_path": None,   # we don't recover critic checkpoints here
        "recovered_from_volume": True,
    }


@app.local_entrypoint()
def main(
    targets: str = None,                # e.g. "10,13,16"; default = every 3rd of 30
    population_size: int = 20,          # K
    max_iterations: int = 500,          # M
    sigma: float = 0.1,
    learning_rate: float = 0.05,
    eta: float = 1e-2,
    k_elite: int = None,                # → ESAgentConfig.K_elite (lowercased for Modal CLI)
    w_crosstalk: float = 0.3,
    w_loss: float = 1e-3,
    w_energy: float = 0.1,
    log_every: int = 1,
    out_dir: str = "phase1_training_output",
    use_critic: bool = False,           # train a DQNCritic interleaved with ES
    checkpoint_every: int = 5,          # emit a resume checkpoint every N iters
    # wandb options ------------------------------------------------------
    wandb_enabled: bool = True,
    wandb_project: str = "cs224r-phase1",
    wandb_entity: str = "",             # leave blank to use your default wandb entity
    wandb_run_id: str = "",             # optional override; default = timestamp
):
    if targets is None:
        target_indices = list(range(0, 30, 3))  # 10 evenly-spaced angles out of 30
    else:
        target_indices = [int(x) for x in targets.split(",")]

    training_indices = list(target_indices)  # treat targets as the training-angle set

    config_kwargs = dict(
        K=population_size,
        sigma=sigma,
        alpha_1=learning_rate,
        M=max_iterations,
        eta=eta,
        log_every=log_every,
        K_elite=k_elite,
        w_crosstalk=w_crosstalk,
        w_loss=w_loss,
        w_energy=w_energy,
    )

    # --- wandb config (one run per target, grouped by run_id) ----------
    # We always need a launch_id (used to validate resume checkpoints) whether
    # or not wandb is on. If wandb is enabled, we reuse its run_id as the
    # launch_id so they stay in sync.
    from datetime import datetime
    if not wandb_run_id:
        wandb_run_id = "phase1-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    launch_id = wandb_run_id

    wandb_cfg = None
    if wandb_enabled:
        wandb_cfg = {
            "project": wandb_project,
            "entity": wandb_entity or None,
            "run_id": wandb_run_id,
        }

    print(f"Phase 1 on Modal: {len(target_indices)} targets")
    print(f"  config: K={population_size}  M={max_iterations}  "
          f"σ={sigma}  α_1={learning_rate}  η={eta}")
    if k_elite is not None:
        print(f"          K_elite={k_elite} (elite truncation)")
    print(f"  weights: w_crosstalk={w_crosstalk}  w_loss={w_loss}  w_energy={w_energy}")
    print(f"  critic:  {'ON (per-worker DQNCritic)' if use_critic else 'off'}")
    print(f"  targets: {target_indices}")
    if wandb_cfg is not None:
        entity_part = f"{wandb_entity}/" if wandb_entity else ""
        print(f"  wandb:   project={entity_part}{wandb_project}  group={wandb_run_id}")
    else:
        print("  wandb:   DISABLED (--no-wandb-enabled)")
    print()

    # Build per-target payloads with distinct seeds.
    payloads = [
        {
            "target_idx": idx,
            "training_indices": training_indices,
            "config_kwargs": config_kwargs,
            "seed": seed,
            "wandb": wandb_cfg,
            "use_critic": use_critic,
            "launch_id": launch_id,
            "checkpoint_every": checkpoint_every,
        }
        for seed, idx in enumerate(target_indices)
    ]

    # Spawn each target as a detached Modal FunctionCall. .spawn() returns
    # immediately with a handle; the call runs independently on Modal's task
    # queue, surviving local-side disconnects / network blips / terminal close.
    # This is the fix for the "ConnectionError kills the run" failure mode
    # we hit twice with .map().
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    spawned = []
    print("Spawning Modal function calls...")
    for payload in payloads:
        fc = train_one_target_modal.spawn(payload)
        spawned.append({
            "target_idx": payload["target_idx"],
            "function_call_id": fc.object_id,
        })
        print(f"  ✓ target {payload['target_idx']:02d} → {fc.object_id}")

    spawn_record = {
        "launch_id": launch_id,
        "config_kwargs": config_kwargs,
        "training_indices": training_indices,
        "spawned": spawned,
    }
    with open(out / "spawned_calls.json", "w") as fh:
        json.dump(spawn_record, fh, indent=2)

    print()
    print(f"All {len(spawned)} workers running on Modal independently.")
    print(f"Spawn IDs saved → {(out / 'spawned_calls.json').resolve()}")
    print(f"Local entrypoint exiting now — workers continue regardless.")
    print()
    print(f"When workers finish (or you want partial results), run:")
    print(f"  python -m modal run train_phase1_modal.py::collect")
    print(f"That fetches results + downloads transitions from the Volume.")


# =============================================================================
# Collect entrypoint — downloads results from the Volume after workers run
# =============================================================================

@app.local_entrypoint()
def collect(
    out_dir: str = "phase1_training_output",
    targets: str = None,                # "10,13,16" or default = read from spawned_calls.json
):
    """Pull per-target outputs from the Modal Volume to local disk.

    No FunctionCall.get() blocking — every artifact we care about (eps_star,
    eps_initial, viz PNG, metadata, transitions) is written to the Volume by
    the worker, so we just read from there. Safe to run while workers are
    still in flight; targets that haven't finished yet get reported as
    in-progress (their resume checkpoint still exists on the Volume).
    """
    import numpy as np
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    if targets is not None:
        target_indices = [int(x) for x in targets.split(",")]
    else:
        spawn_path = out / "spawned_calls.json"
        if not spawn_path.exists():
            raise FileNotFoundError(
                f"{spawn_path} not found. Either pass --targets explicitly "
                f"or re-run main first to spawn workers."
            )
        with open(spawn_path) as fh:
            spawn_record = json.load(fh)
        target_indices = [s["target_idx"] for s in spawn_record["spawned"]]

    print(f"Collecting {len(target_indices)} targets from Volume...")
    summary = []
    merged_transitions = []
    for target_idx in target_indices:
        d = out / f"target_{target_idx:02d}"
        d.mkdir(exist_ok=True)
        prefix = f"target_{target_idx:02d}"

        # eps_star — the memory bank entry; if missing, the worker hasn't
        # finished yet (resume checkpoint may still exist).
        try:
            eps_bytes = b"".join(buffer_volume.read_file(f"{prefix}_eps_star.npy"))
            import io as _io
            np.save(d / "eps_star.npy", np.load(_io.BytesIO(eps_bytes)))
            status = "complete"
        except Exception:
            # Check for resume checkpoint to report meaningful in-progress state.
            try:
                ckpt_blob = b"".join(buffer_volume.read_file(f"{prefix}_ckpt.pkl"))
                ckpt = pickle.loads(ckpt_blob)
                status = f"in-progress (iter {ckpt.next_iteration})"
            except Exception:
                status = "no artifacts yet"
            print(f"  · target {target_idx:02d}: {status} — skipping")
            continue

        # eps_initial / viz PNG / metadata — best-effort.
        for fname, save_name in [
            (f"{prefix}_eps_initial.npy", "eps_initial.npy"),
            (f"{prefix}_before_after.png", "before_after.png"),
            (f"{prefix}_meta.json", "metadata.json"),
        ]:
            try:
                blob = b"".join(buffer_volume.read_file(fname))
                if save_name.endswith(".npy"):
                    import io as _io
                    np.save(d / save_name, np.load(_io.BytesIO(blob)))
                else:
                    (d / save_name).write_bytes(blob)
            except Exception:
                pass

        # Transitions — these are the big artifact for Phase 2 training.
        n_trans = 0
        try:
            blob = b"".join(buffer_volume.read_file(f"{prefix}.pkl"))
            trans = pickle.loads(blob)
            merged_transitions.extend(trans)
            n_trans = len(trans)
        except Exception as e:
            print(f"  ✗ target {target_idx:02d}: transitions missing ({e})")

        # Pull per-target summary from the metadata if available.
        meta_path = d / "metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
        summary.append({
            "target_idx": target_idx,
            "iterations": meta.get("iterations"),
            "converged": meta.get("converged"),
            "best_reward": meta.get("best_reward"),
            "n_transitions": n_trans,
        })
        print(f"  ✓ target {target_idx:02d}: complete  "
              f"iter={meta.get('iterations')}  "
              f"converged={meta.get('converged')}  "
              f"n_trans={n_trans:,}")

    # Merged replay buffer.
    if merged_transitions:
        with open(out / "replay_buffer.pkl", "wb") as fh:
            pickle.dump(merged_transitions, fh)
        print()
        print(f"Merged replay buffer → {(out / 'replay_buffer.pkl').resolve()}  "
              f"({len(merged_transitions):,} transitions)")

    # Aggregated summary.
    with open(out / "summary.json", "w") as fh:
        json.dump({"results": summary,
                   "total_transitions": len(merged_transitions)}, fh, indent=2)
    n_done = len([s for s in summary if s.get("iterations") is not None])
    print(f"\n{n_done}/{len(target_indices)} targets fully collected. "
          f"Outputs → {out.resolve()}")
