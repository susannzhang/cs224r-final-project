"""Modal-parallel gradient (BPTT) Phase 2 trainer — true FDFD, fanned across workers.

The BPTT gradient is sequential over the T rollout steps (it backprops through
the trajectory), so the parallelizable dimension is the BATCH of tasks: at each
step every task's FDFD forward solve runs on its own Modal worker, and during
backward every task's adjoint (VJP) solve does too. The autograd graph (policy +
optimizer) lives on the driver; only the expensive solves are remote.

    driver (1 container): policy forward → δ, ε' = clip(ε+δ),  loss.backward(), Adam
       │  per step:  FDFDWorker.forward.map(B states) → P        (B workers)
       │  per step:  FDFDWorker.vjp.map(B (ε,∂L/∂P))  → ∂L/∂ε    (B workers, in backward)
       ▼
    RemoteFDFDPowerModel  (algorithms/infrastructure/fdfd_adjoint.py)

Wall-clock per gradient step ≈ T forward rounds + T adjoint rounds, each round B
solves in parallel — so B-way speedup over a single-container BPTT. Trained
ONLY on the Phase-1 learned angles (goal_indices = memory_bank.keys()), matching
train_phase2_grad_learned_angles.py but fully on Modal.

Usage:
    modal run train_phase2_grad_modal.py::main \\
        --memory-bank phase1-uniform-init-output \\
        --policy pretrain/policy_buffer_traj_pinn.pt \\
        --reward-mode source_norm --t 20 --batch-tasks 16 --n-iter 300 \\
        --checkpoint-every 50 --wandb-project cs153-phase2
    # recover a timed-out run's latest checkpoint from the Volume:
    modal run train_phase2_grad_modal.py::collect --run-id <run_id>
    # resume training a dying run (same run_id):
    modal run train_phase2_grad_modal.py::main --run-id <run_id> --resume ...
"""

import io
import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root
APP_NAME = os.environ.get("PHASE2_GRAD_APP_NAME", "cs224r-phase2-grad")
VOLUME_NAME = os.environ.get("PHASE2_GRAD_VOLUME_NAME", "cs224r-phase2-grad-output")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install("numpy", "scipy", "scikit-image", "autograd", "ceviche", "wandb")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(
        PROJECT_ROOT, "/root/app", copy=True,
        ignore=["__pycache__", "*.pyc", ".git", ".venv", ".pytest_cache",
                "*.pkl", "wandb", "pretrain", "phase1_training_output",
                "phase1-uniform-init-output", "phase2_parallel_output",
                "checkpoint_output", "phase2_tiny_smoke"],
    )
)
app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

try:
    wandb_secret = modal.Secret.from_name("wandb")
except modal.exception.NotFoundError:
    wandb_secret = None


def _add_paths():
    import sys
    for p in ("/root/app", "/root/app/dynamic_beam_steering"):
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_env():
    """Canonical pm_setup env (mirrors train_phase1.build_env / the parallel driver)."""
    from geometry import (create_design_region, create_environment, create_grid,
                          create_receiver, create_source)
    from simulation import initialize_environment
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0, margin_cells=20)
    grid = create_grid(num_rods_x=10, num_rods_y=10, radius=0.01, distance=0.002,
                       rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
    receivers = []
    for i in range(1, 11):
        receivers.append(create_receiver(index=i, length=0.02, side='bottom', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=10 + i, length=0.02, side='right', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=20 + i, length=0.02, side='top', rod_index=11 - i))
    env = create_environment(design_region=region, grid=grid, sources=[source], receivers=receivers)
    initialize_environment(env)
    return env


# --- the unit of remote work: one forward solve / one adjoint (VJP) solve ----
# A Cls (not bare functions) so @modal.enter() builds the FDFDAdjointSolver
# ONCE per container lifecycle (env build + 476² mask construction is
# expensive). Each container then handles many .map() items off the cached
# solver. Add min_containers=<batch_tasks> to @app.cls to pin the worker pool
# warm across the BPTT dispatch rounds (reserves cost; tune to taste).
@app.cls(image=image, cpu=4, memory=8192, timeout=600)
class FDFDWorker:
    @modal.enter()
    def _setup(self):
        _add_paths()
        from algorithms.infrastructure.fdfd_adjoint import FDFDAdjointSolver
        self.solver = FDFDAdjointSolver(_build_env())

    @modal.method()
    def forward(self, eps_np):
        from algorithms.infrastructure.fdfd_adjoint import fdfd_solve_forward
        return fdfd_solve_forward(self.solver, eps_np)

    @modal.method()
    def vjp(self, arg):
        from algorithms.infrastructure.fdfd_adjoint import fdfd_solve_vjp
        eps_np, g_np = arg
        return fdfd_solve_vjp(self.solver, eps_np, g_np)


# --- driver: holds the policy + autograd graph, fans solves to the workers ---
@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 24,                # 24 hours — Modal max
    secrets=[wandb_secret] if wandb_secret is not None else [],
    volumes={"/buffer": output_volume},  # durable checkpoints survive a dying run
)
def driver(payload: dict) -> bytes:
    _add_paths()
    import json
    import numpy as np
    import torch
    from algorithms.policies.es_policy import ESPolicy, ESPolicyConfig, Phase2GradConfig
    from algorithms.infrastructure.fdfd_adjoint import RemoteFDFDPowerModel

    env = _build_env()
    n_recv = len(env.receivers)
    ss = (env.grid.num_rods_x, env.grid.num_rods_y)

    run_id = payload.get("run_id", "grad-run")
    checkpoint_every = int(payload.get("checkpoint_every", 50))
    out_dir = Path("/buffer") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "policy_grad.pt"

    def _load_ckpt_bytes(b):
        ck = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        p = ESPolicy(state_shape=ck["state_shape"], config=ck["config"])
        p.pi.load_state_dict(ck["pi_state_dict"])
        return p

    # Policy: resume from Volume (--resume) > refine a shipped checkpoint > fresh.
    policy = None
    if payload.get("resume"):
        output_volume.reload()
        if ckpt_path.exists():
            policy = _load_ckpt_bytes(ckpt_path.read_bytes())
            print(f"[grad] RESUMED from {ckpt_path} "
                  f"(arch={vars(policy.config).get('policy_arch','mlp')})", flush=True)
    if policy is None and payload.get("policy_bytes"):
        policy = _load_ckpt_bytes(payload["policy_bytes"])
        print(f"[grad] refining policy (arch={vars(policy.config).get('policy_arch','mlp')})",
              flush=True)
    if policy is None:
        policy = ESPolicy(ss, ESPolicyConfig(policy_arch=payload["policy_arch"],
                                             n_goals=n_recv, tanh_output_scale=0.25))
        print(f"[grad] fresh {payload['policy_arch']} policy", flush=True)

    memory_bank = {int(k): np.asarray(v, np.float32) for k, v in payload["memory_bank"].items()}
    goal_indices = sorted(memory_bank.keys())           # <-- LEARNED ANGLES ONLY
    print(f"[grad] learned angles: {goal_indices}", flush=True)

    # Power model: maps = Modal .map() over the worker pool (B-way parallel).
    # FDFDWorker() is a handle; .forward.map / .vjp.map fan across containers,
    # each holding a @modal.enter()-built solver.
    worker = FDFDWorker()
    forward_map = lambda eps_list: list(worker.forward.map(list(eps_list)))
    vjp_map = lambda pairs: list(worker.vjp.map(list(pairs)))
    power_model = RemoteFDFDPowerModel(forward_map, vjp_map)

    gcfg = dict(payload["grad_cfg"])
    if gcfg.get("reward_mode") == "source_norm" and not gcfg.get("p_source"):
        from algorithms.agents.es_agent import compute_source_power
        gcfg["p_source"] = compute_source_power(env)
        print(f"[grad] source_norm: auto p_source={gcfg['p_source']:.4g}", flush=True)
    cfg = Phase2GradConfig(**gcfg)

    # --- wandb (optional; mirrors the ES drivers) -------------------
    wandb_cfg = payload.get("wandb")
    wandb_run = None
    if wandb_cfg is not None and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity") or None,
            name=wandb_cfg.get("name", "phase2-grad"),
            group=wandb_cfg.get("group"),
            tags=[f"arch={payload.get('policy_arch', '?')}",
                  f"reward={gcfg.get('reward_mode')}",
                  f"T={gcfg.get('T')}", f"batch={gcfg.get('batch_tasks')}"],
            config={**gcfg, "goal_indices": goal_indices},
            reinit=True,
        )

    def _save_checkpoint(history, label="checkpoint"):
        """Persist policy + history to the Volume so progress survives a dying
        run. Overwrites the same files each time (latest = recoverable state)."""
        b = io.BytesIO()
        policy.save(b)                       # {pi_state_dict, config, state_shape}
        ckpt_path.write_bytes(b.getvalue())
        (out_dir / "history.json").write_text(json.dumps(
            {"run_id": run_id, "grad_cfg": gcfg, "goal_indices": goal_indices,
             "history": history}, indent=2))
        output_volume.commit()
        print(f"[grad] {label}: {ckpt_path.name} + history.json "
              f"({len(history)} log entries) → Volume {run_id}/", flush=True)

    state = {"history": [], "last_ckpt": -1}

    def _on_iter(r):
        print(f"[grad] {r}", flush=True)
        state["history"].append(r)
        if wandb_run is not None:
            import wandb
            log = {"reward_mean": r["reward_mean"], "loss": r["loss"]}
            if "phys_residual" in r:
                log["phys_residual"] = r["phys_residual"]
            wandb.log(log, step=r["iteration"])
        if checkpoint_every > 0 and r["iteration"] - state["last_ckpt"] >= checkpoint_every:
            _save_checkpoint(state["history"], label=f"checkpoint_iter_{r['iteration']:04d}")
            state["last_ckpt"] = r["iteration"]

    policy.train_phase2_grad(
        power_model, memory_bank, goal_indices, cfg,
        converged_indices=goal_indices, on_iteration=_on_iter,
    )

    # Final durable save (Volume) + return bytes for the local entrypoint.
    _save_checkpoint(state["history"], label="final")
    if wandb_run is not None:
        wandb_run.finish()

    buf = io.BytesIO()
    policy.save(buf)
    return buf.getvalue()


def _load_memory_bank(d: Path) -> dict:
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
    if not bank:
        raise FileNotFoundError(f"No target_*/eps_star.npy under {d}")
    return bank


@app.local_entrypoint()
def main(memory_bank: str, policy: str = "", out: str = "pretrain/policy_grad_modal.pt",
         policy_arch: str = "pinn", reward_mode: str = "source_norm",
         n_iter: int = 300, t: int = 20, batch_tasks: int = 16, lr: float = 5e-4,
         gamma: float = 0.95, bptt_truncate: int = 4, one_step: bool = False,
         p_rand: float = 0.2, physics_loss_weight: float = 0.1, p_source: float = 0.0,
         seed: int = 0, log_every: int = 1, run_id: str = "", checkpoint_every: int = 10,
         resume: bool = False, spawn: bool = False, wandb_project: str = "",
         wandb_entity: str = "", wandb_name: str = ""):
    """Ship inputs to Modal, run the BPTT trainer, save the returned checkpoint.
    Reading the memory bank / policy file is local I/O only — no training here.
    Periodic checkpoints land in the Volume under <run_id>/ (recover with
    `modal run ...::collect --run-id <id>`); pass the SAME --run-id with
    --resume to continue a dying run. --wandb-project streams metrics (needs the
    'wandb' Modal secret)."""
    import time
    if not run_id:
        run_id = f"grad-{policy_arch}-{reward_mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    bank = _load_memory_bank(Path(memory_bank))
    payload = {
        "memory_bank": {k: v for k, v in bank.items()},
        "policy_bytes": Path(policy).read_bytes() if policy else None,
        "policy_arch": policy_arch,
        "run_id": run_id,
        "checkpoint_every": checkpoint_every,
        "resume": resume,
        "grad_cfg": dict(
            reward_mode=reward_mode, n_iter=n_iter, T=t, batch_tasks=batch_tasks,
            lr=lr, gamma=gamma, bptt_truncate=bptt_truncate, one_step=one_step,
            p_rand=p_rand, physics_loss_weight=physics_loss_weight,
            p_source=p_source, seed=seed, log_every=log_every,
        ),
        "wandb": ({"project": wandb_project, "entity": wandb_entity or None,
                   "name": wandb_name or run_id, "group": run_id}
                  if wandb_project else None),
    }
    print(f"Launching Modal BPTT trainer [run_id={run_id}]: arch={policy_arch} "
          f"reward={reward_mode} T={t} batch={batch_tasks} n_iter={n_iter} "
          f"ckpt_every={checkpoint_every} resume={resume} spawn={spawn}"
          + (f"  → wandb:{wandb_project}" if wandb_project else ""))

    if spawn:
        # Fire-and-forget: submit the call to Modal and return immediately, so
        # the local launcher exits in seconds (a clean disconnect). Combined
        # with `modal run --detach`, the driver runs fully server-side and
        # survives the local client dying. Recover the result from the Volume.
        call = driver.spawn(payload)
        print(f"Spawned driver (call {call.object_id}); run continues on Modal.\n"
              f"Recover: modal run train_phase2_grad_modal.py::collect --run-id {run_id}\n"
              f"Resume:  modal run train_phase2_grad_modal.py::main --spawn "
              f"--run-id {run_id} --resume ...")
        return

    policy_bytes = driver.remote(payload)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(policy_bytes)
    print(f"Refined policy → {out_path.resolve()}  (also in Volume {run_id}/policy_grad.pt)")


@app.local_entrypoint()
def collect(run_id: str = "", out_dir: str = "pretrain"):
    """Download a (possibly partial) checkpoint from the Volume by run_id —
    e.g. to recover a run that hit the timeout before returning."""
    if not run_id:
        raise ValueError("--run-id required")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("policy_grad.pt", "history.json"):
        try:
            blob = b"".join(output_volume.read_file(f"{run_id}/{name}"))
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            continue
        (out / f"{run_id}_{name}").write_bytes(blob)
        print(f"  ✓ {name} ({len(blob)/1024:.1f} KB) → {out / (run_id + '_' + name)}")
