"""BPTT through M for Phase 2 policy training (runs on Modal).

ES gave us noisy gradients (variance dominated the signal) and the policy
drifted away from the imitation warm-start. With M(ε) → P[30] already
trained and differentiable, we can compute the policy gradient directly:

  for each (ε_0, goal) task:
      ε_t+1 = clip(ε_t + π_φ(ε_t, goal), -1, 1)        (T=20 steps)
      P_t   = M(ε_t+1)
      Q_t   = P_t[goal]² / (P_t.sum() + ε)
      F     = Σ_t Q_t                                   (trajectory total)
  loss = -mean(F across batch of tasks)

  ∂loss / ∂φ via PyTorch autograd → Adam on φ.

No FDFD per step (no per-step Modal dispatch needed). The whole training
loop runs on a single Modal container.

Usage (no deploy needed):
    modal run train_phase2_bptt.py::main \\
        --policy pretrain/policy_buffer_traj_h100l2.pt \\
        --surrogate pretrain/M_fdfd_surrogate.pt \\
        --memory-bank phase1-uniform-init-output \\
        --out pretrain/policy_bptt.pt
"""

import io
import os
import pickle
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root (for geometry, simulation)
APP_NAME = os.environ.get("PHASE2_BPTT_APP_NAME", "cs224r-phase2-bptt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "scipy", "scikit-image")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
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


@app.function(image=image, cpu=4, memory=8192, timeout=60 * 60)
def train_bptt(payload: dict) -> bytes:
    """BPTT training loop on Modal. Returns final policy.pt bytes."""
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")

    import time
    import numpy as np
    import torch
    import torch.optim as optim

    from algorithms.policies.es_policy import ESPolicy
    from train_V_network import PNetwork

    rng = np.random.default_rng(payload["seed"])
    torch.manual_seed(payload["seed"])

    # ---- Load policy (warm-start; trainable) ----
    policy_bytes = payload["policy_bytes"]
    ckpt_pi = torch.load(io.BytesIO(policy_bytes), map_location="cpu",
                         weights_only=False)
    policy = ESPolicy(state_shape=ckpt_pi["state_shape"],
                      config=ckpt_pi["config"])
    policy.pi.load_state_dict(ckpt_pi["pi_state_dict"])
    policy.pi.train()
    n_pi = sum(p.numel() for p in policy.pi.parameters())
    print(f"[bptt] policy: state_shape={policy.state_shape}  params={n_pi:,}  "
          f"tanh_scale={getattr(policy.config, 'tanh_output_scale', 1.0)}",
          flush=True)

    # ---- Load M (frozen) ----
    m_bytes = payload["m_bytes"]
    M_ckpt = torch.load(io.BytesIO(m_bytes), map_location="cpu",
                        weights_only=False)
    M = PNetwork(M_ckpt["state_shape"], M_ckpt["n_receivers"],
                 M_ckpt["hidden_dim"], M_ckpt["n_hidden_layers"])
    M.load_state_dict(M_ckpt["model_state_dict"])
    M.eval()
    for p in M.parameters():
        p.requires_grad_(False)
    log_mean = torch.as_tensor(M_ckpt["log_mean"], dtype=torch.float32)
    log_std = torch.as_tensor(M_ckpt["log_std"], dtype=torch.float32)
    print(f"[bptt] M: hidden={M_ckpt['hidden_dim']}×{M_ckpt['n_hidden_layers']}  "
          f"params={sum(p.numel() for p in M.parameters()):,} (frozen)",
          flush=True)

    # ---- Tasks ----
    bank = {int(k): np.asarray(v, dtype=np.float32)
            for k, v in payload["memory_bank"].items()}
    goal_indices = list(payload["goal_indices"])
    converged_indices = list(payload["converged_indices"])
    state_shape = tuple(next(iter(bank.values())).shape)
    print(f"[bptt] bank: {sorted(bank.keys())}  "
          f"goals: {goal_indices}  prev: {converged_indices}", flush=True)

    # ---- Config ----
    n_iter = int(payload["n_iter"])
    batch_size = int(payload["batch_size"])
    T = int(payload["T"])
    p_rand = float(payload["p_rand"])
    lr = float(payload["lr"])
    grad_clip = float(payload["grad_clip"])
    log_every = int(payload["log_every"])

    optimizer = optim.Adam(policy.pi.parameters(), lr=lr)
    print(f"[bptt] training:  n_iter={n_iter}  batch={batch_size}  T={T}  "
          f"lr={lr}  grad_clip={grad_clip}", flush=True)

    history = []
    t0 = time.time()
    for it in range(n_iter):
        # Sample a batch of tasks (goal, ε_0)
        eps_0_list, goals_list = [], []
        for _ in range(batch_size):
            goal = int(rng.choice(goal_indices))
            cands = [g for g in converged_indices if g != goal and g in bank]
            prev = int(rng.choice(cands)) if cands else None
            use_random = (rng.random() < p_rand) or (prev is None)
            if use_random:
                eps_0 = rng.uniform(-1.0, 1.0, size=state_shape).astype(np.float32)
            else:
                eps_0 = bank[prev].astype(np.float32)
            eps_0_list.append(eps_0)
            goals_list.append(goal)

        eps = torch.as_tensor(np.stack(eps_0_list), dtype=torch.float32)
        goals_t = torch.as_tensor(goals_list, dtype=torch.long)

        # Differentiable rollout
        total_Q = 0.0
        Q_T_log = None
        for t in range(T):
            delta = policy.pi(eps, goals_t)
            eps = torch.clamp(eps + delta, -1.0, 1.0)
            normed = M(eps)
            log_P = normed * log_std + log_mean
            P_pred = torch.expm1(log_P).clamp(min=0.0)
            P_at_goal = P_pred.gather(1, goals_t.unsqueeze(1)).squeeze(1)
            P_total = P_pred.sum(dim=1)
            Q = (P_at_goal ** 2) / (P_total + 1e-9)
            total_Q = total_Q + Q.mean()
            Q_T_log = Q.mean()

        loss = -total_Q
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                policy.pi.parameters(), grad_clip))
        else:
            grad_norm = 0.0
        optimizer.step()

        history.append({
            "iter": it,
            "loss": float(loss.detach()),
            "total_Q": float(total_Q.detach()),
            "Q_T_mean": float(Q_T_log.detach()),
            "grad_norm": grad_norm,
        })

        if (it + 1) % log_every == 0 or it == 0:
            elapsed = time.time() - t0
            print(f"[bptt iter {it + 1:>4}/{n_iter}]  "
                  f"Σ_t Q_t per traj = {history[-1]['total_Q']:+.3e}  "
                  f"Q_T mean = {history[-1]['Q_T_mean']:+.3e}  "
                  f"grad_norm = {grad_norm:.2f}  "
                  f"({elapsed:.1f}s)", flush=True)

    # ---- Serialize result ----
    policy.pi.eval()
    out = io.BytesIO()
    torch.save({
        "pi_state_dict": policy.pi.state_dict(),
        "config": policy.config,
        "state_shape": policy.state_shape,
    }, out)
    return pickle.dumps({
        "policy_bytes": out.getvalue(),
        "history": history,
    })


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
    return bank


@app.local_entrypoint()
def main(
    policy: str = "pretrain/policy_buffer_traj_h100l2.pt",
    surrogate: str = "pretrain/M_fdfd_surrogate.pt",
    memory_bank: str = "phase1-uniform-init-output",
    goal_indices: str = "1,4,7,10,13,16,19,22,25,28",
    converged_indices: str = "",
    n_iter: int = 300,
    batch_size: int = 32,
    nsteps: int = 20,
    p_rand: float = 0.1,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    log_every: int = 10,
    out: str = "pretrain/policy_bptt.pt",
    seed: int = 0,
):
    """Dispatch BPTT training to Modal."""
    import numpy as np

    policy_path = Path(policy)
    surrogate_path = Path(surrogate)
    out_path = Path(out)
    memory_bank_path = Path(memory_bank)

    print(f"Loading policy: {policy_path}")
    policy_bytes = policy_path.read_bytes()
    print(f"Loading surrogate: {surrogate_path}")
    m_bytes = surrogate_path.read_bytes()
    print(f"Loading memory bank: {memory_bank_path}")
    bank = _load_memory_bank(memory_bank_path)
    print(f"  bank angles: {sorted(bank.keys())}")

    goal_indices_list = [int(x) for x in goal_indices.split(",")]
    if converged_indices:
        converged_indices_list = [int(x) for x in converged_indices.split(",")]
    else:
        converged_indices_list = sorted(bank.keys())

    payload = {
        "policy_bytes": policy_bytes,
        "m_bytes": m_bytes,
        "memory_bank": {int(k): np.asarray(v) for k, v in bank.items()},
        "goal_indices": goal_indices_list,
        "converged_indices": converged_indices_list,
        "n_iter": n_iter,
        "batch_size": batch_size,
        "T": nsteps,
        "p_rand": p_rand,
        "lr": lr,
        "grad_clip": grad_clip,
        "log_every": log_every,
        "seed": seed,
    }

    print(f"\nDispatching to Modal: n_iter={n_iter}  batch={batch_size}  "
          f"T={nsteps}  lr={lr}")
    print("(stdout from worker streams below)\n")
    result_bytes = train_bptt.remote(payload)
    result = pickle.loads(result_bytes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(result["policy_bytes"])
    print(f"\n✓ Trained policy saved → {out_path.resolve()}")

    # Also save loss history
    history = result["history"]
    np.savez(out_path.with_suffix(".loss.npz"),
             loss=np.array([h["loss"] for h in history]),
             total_Q=np.array([h["total_Q"] for h in history]),
             Q_T_mean=np.array([h["Q_T_mean"] for h in history]),
             grad_norm=np.array([h["grad_norm"] for h in history]))
    print(f"  loss history → {out_path.with_suffix('.loss.npz')}")
