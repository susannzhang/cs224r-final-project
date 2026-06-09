# =============================================================================
# Phase 2 Training on Modal — ES on policy φ with MC-return fitness
# =============================================================================
"""
Cloud-run version of Phase 2. ONE Modal container runs the full ES loop end
to end (unlike Phase 1's fan-out, Phase 2 trains a single multi-goal policy).

Architecture mirrors Phase 1's deploy + spawn pattern:
    1. modal deploy train_phase2_modal.py     (one-time)
    2. python spawn_phase2.py ...             (queues a worker)
    3. modal run train_phase2_modal.py::collect  (download results)

The worker reads policy_awr.pt + memory_bank from the payload (small enough
to ship inline), runs train_phase2(), and writes policy_phase2.pt +
history.json to the Modal Volume.

Per-iter FDFD cost: K × T solves. At ~10-20 s/FDFD on Modal CPU, plan for
K=20, T=10, N_iter=20 (≈4 k FDFDs ≈ 1 hour) for a smoke run. Larger runs
will need either Modal retries + checkpointing (TODO) or down-scaling.

Volume layout (per launch_id):
    /buffer/<launch_id>/policy_phase2.pt
    /buffer/<launch_id>/history.json
    /buffer/<launch_id>/buffer_appended.pkl     (rollout transitions logged)
    /buffer/<launch_id>/meta.json
"""

import json
import os
import pickle
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root (for geometry, simulation)

# App + volume names — env-overridable for parallel experiments.
APP_NAME = os.environ.get("PHASE2_APP_NAME", "cs224r-phase2-es")
VOLUME_NAME = os.environ.get("PHASE2_VOLUME_NAME", "cs224r-phase2-buffer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install(
        "numpy", "scipy", "scikit-image", "matplotlib",
        "autograd", "ceviche", "wandb",
    )
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(PROJECT_ROOT, "/root/app", copy=True,
                   ignore=["__pycache__", "*.pyc", "phase1_checkpoints",
                           "phase1_training_output", "phase1-uniform-init-output",
                           "checkpoint_output", "phase2_tiny_smoke",
                           "tests/visual_output", ".git", ".venv",
                           ".pytest_cache"])
)

app = modal.App(APP_NAME)
buffer_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

try:
    wandb_secret = modal.Secret.from_name("wandb")
except modal.exception.NotFoundError:
    wandb_secret = None


# =============================================================================
# Worker — runs the full Phase 2 ES loop
# =============================================================================

@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60 * 24,        # 24 hours — Modal max
    volumes={"/buffer": buffer_volume},
    secrets=[wandb_secret] if wandb_secret is not None else [],
)
def train_phase2_worker(payload: dict) -> dict:
    """One container, full Phase 2 run. Loads warm-start policy + memory
    bank from the payload (small), runs train_phase2, persists outputs.
    """
    import io
    import sys
    import time
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")

    import numpy as np

    from algorithms.infrastructure.utils import ReplayBuffer
    from algorithms.policies.es_policy import ESPolicy, Phase2Config
    from geometry import (create_design_region, create_environment,
                          create_grid, create_receiver, create_source)
    from simulation import initialize_environment

    launch_id = payload["launch_id"]
    goal_indices = payload["goal_indices"]                 # Phase 2 targets
    converged_indices = payload.get("converged_indices")   # Phase 1 angles (optional)
    config_kwargs = payload["config_kwargs"]
    policy_bytes = payload.get("policy_bytes")             # optional
    policy_config_kwargs = payload.get("policy_config_kwargs")  # optional
    memory_bank_arrays = payload["memory_bank"]   # {idx: np.ndarray}
    seed_buffer_bytes = payload.get("buffer_bytes")  # optional
    wandb_cfg = payload.get("wandb")

    # --- Wandb init -------------------------------------------------
    wandb_run = None
    if wandb_cfg is not None and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity") or None,
            name=f"phase2-{launch_id}",
            group=launch_id,
            tags=[f"K={config_kwargs['K']}",
                  f"N_iter={config_kwargs['N_iter']}",
                  f"T={config_kwargs['T']}"],
            config={**config_kwargs,
                    "goal_indices": goal_indices,
                    "converged_indices": converged_indices},
            reinit=True,
        )

    # --- Build policy: load warm-start checkpoint OR construct fresh ----
    import torch
    from algorithms.policies.es_policy import ESPolicyConfig
    if policy_bytes is not None:
        policy_buffer = io.BytesIO(policy_bytes)
        ckpt = torch.load(policy_buffer, map_location="cpu", weights_only=False)
        state_shape = ckpt["state_shape"]
        pi_state_dict = ckpt["pi_state_dict"]
        policy_config = ckpt["config"]
        policy = ESPolicy(state_shape=state_shape, config=policy_config)
        policy.pi.load_state_dict(pi_state_dict)
        print(f"[phase2] loaded warm-start policy: "
              f"state_shape={state_shape}  n_goals={policy.config.n_goals}",
              flush=True)
    else:
        if policy_config_kwargs is None:
            raise ValueError("Payload must include policy_bytes OR "
                             "policy_config_kwargs (for a fresh policy).")
        state_shape = tuple(policy_config_kwargs["state_shape"])
        pcfg = ESPolicyConfig(
            policy_arch=policy_config_kwargs.get("policy_arch", "cnn"),
            hidden_dim=policy_config_kwargs["hidden_dim"],
            n_hidden_layers=policy_config_kwargs["n_hidden_layers"],
            tanh_output=policy_config_kwargs["tanh_output"],
            n_goals=policy_config_kwargs["n_goals"],
            seed=config_kwargs.get("seed", 0),
        )
        policy = ESPolicy(state_shape=state_shape, config=pcfg)
        print(f"[phase2] no warm-start. Built fresh ESPolicy: "
              f"state_shape={state_shape}  n_goals={pcfg.n_goals}  "
              f"hidden_dim={pcfg.hidden_dim} × {pcfg.n_hidden_layers}",
              flush=True)

    # --- Memory bank (np arrays) -----------------------------------
    memory_bank = {int(k): np.asarray(v) for k, v in memory_bank_arrays.items()}
    print(f"[phase2] memory bank: {len(memory_bank)} entries → "
          f"{sorted(memory_bank.keys())}", flush=True)

    # --- Buffer (optional seed from Phase 1) -----------------------
    buffer = ReplayBuffer()
    if seed_buffer_bytes is not None:
        seed_buf = io.BytesIO(seed_buffer_bytes)
        transitions = pickle.load(seed_buf)
        buffer.extend(transitions)
        print(f"[phase2] buffer seeded with {len(buffer):,} Phase 1 transitions",
              flush=True)

    # --- Build pm_setup env (must mirror Phase 1 worker) ------------
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
    print(f"[phase2] env built ({len(env.receivers)} receivers)", flush=True)

    # --- Phase 2 config + training ---------------------------------
    cfg = Phase2Config(**config_kwargs)
    print(f"[phase2] config: K={cfg.K}  σ={cfg.sigma}  α_2={cfg.alpha_2}  "
          f"N_iter={cfg.N_iter}  T={cfg.T}  p_rand={cfg.p_rand}  η={cfg.eta}",
          flush=True)

    # Live wandb streaming per logged iter.
    def _on_iter(entry):
        if wandb_run is None:
            return
        import wandb
        wandb.log({
            "iteration": entry["iteration"],
            "fitness/mean": entry["fitness_mean"],
            "fitness/best": entry["fitness_best"],
            "mean_rollout_length": entry["mean_rollout_length"],
        }, step=entry["iteration"])
        print(f"  [iter {entry['iteration']:>4}]  "
              f"fitness mean={entry['fitness_mean']:+.3e}  "
              f"best={entry['fitness_best']:+.3e}  "
              f"rollout_len={entry['mean_rollout_length']:.1f}",
              flush=True)

    t0 = time.time()
    result = policy.train_phase2(
        env=env, buffer=buffer, memory_bank=memory_bank,
        goal_indices=goal_indices, config=cfg,
        converged_indices=converged_indices,
        on_iteration=_on_iter,
    )
    elapsed = time.time() - t0
    print(f"[phase2] done in {elapsed/60:.1f} min "
          f"({len(result['history'])} log entries)", flush=True)

    # --- Persist to Volume under launch_id namespace ---------------
    out_dir = Path("/buffer") / launch_id
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_out = io.BytesIO()
    torch.save({
        "pi_state_dict": policy.pi.state_dict(),
        "config": policy.config,
        "state_shape": policy.state_shape,
    }, policy_out)
    policy_pt_bytes = policy_out.getvalue()
    (out_dir / "policy_phase2.pt").write_bytes(policy_pt_bytes)

    history_json = json.dumps({
        "config": config_kwargs,
        "goal_indices": goal_indices,
        "converged_indices": converged_indices,
        "history": result["history"],
        "elapsed_seconds": elapsed,
        "n_transitions_end": len(buffer),
    }, indent=2)
    (out_dir / "history.json").write_text(history_json)

    # Buffer appended during Phase 2 (initial seed + Phase 2 rollouts).
    with open(out_dir / "buffer_appended.pkl", "wb") as fh:
        pickle.dump(list(buffer.transitions), fh)

    (out_dir / "meta.json").write_text(json.dumps({
        "launch_id": launch_id,
        "goal_indices": goal_indices,
        "converged_indices": converged_indices,
        "n_history_entries": len(result["history"]),
        "n_transitions_end": len(buffer),
        "elapsed_seconds": elapsed,
    }, indent=2))
    buffer_volume.commit()

    if wandb_run is not None:
        import wandb
        wandb.log({
            "final/elapsed_minutes": elapsed / 60,
            "final/n_transitions_end": len(buffer),
            "final/n_history_entries": len(result["history"]),
        })
        wandb_run.finish()

    return {
        "launch_id": launch_id,
        "history": result["history"],
        "policy_pt_bytes": policy_pt_bytes,
        "n_transitions_end": len(buffer),
        "elapsed_seconds": elapsed,
    }


# =============================================================================
# Collect entrypoint — pull results from the Volume
# =============================================================================

@app.local_entrypoint()
def collect(
    launch_id: str = None,         # required
    out_dir: str = "phase2_output",
):
    """Download Phase 2 outputs for a given launch_id from the Volume.

    Pulls policy_phase2.pt + history.json + buffer_appended.pkl + meta.json
    into <out_dir>/<launch_id>/.
    """
    if not launch_id:
        raise ValueError("--launch-id is required (see spawned_calls.json)")
    out = Path(out_dir) / launch_id
    out.mkdir(parents=True, exist_ok=True)

    files = ["policy_phase2.pt", "history.json", "meta.json", "buffer_appended.pkl"]
    for name in files:
        src = f"{launch_id}/{name}"
        try:
            blob = b"".join(buffer_volume.read_file(src))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            continue
        (out / name).write_bytes(blob)
        print(f"  ✓ {name}  ({len(blob)/1024:.1f} KB)")

    print(f"\nOutputs → {out.resolve()}")
