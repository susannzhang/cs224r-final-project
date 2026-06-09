"""Train Phase 2 policy on anchor-to-anchor retargeting trajectories.

For each pair (a, b) of Phase 1 anchors (a ≠ b):
  - Start at ε*(a)
  - Run state-space ES on ε targeting goal b for N_iter steps
    (M-inner-loop: scoring via the FDFD surrogate, no FDFD per step)
  - Record the trajectory of ε's the ES mean visited
  - FDFD-verify the trajectory's final state
  - Compare against the FDFD-true target_frac of linear-interp at b
  - If ES endpoint is better, keep its trajectory as supervised data
  - Else, fall back to linear-interpolation trajectory (still useful data)

Then train a small policy NN on the resulting (ε_t, goal=b, δ_t=ε_{t+1}−ε_t)
tuples — closed-loop in *form*, but with training data carefully constructed
from FDFD-verified anchor-pair retargeting paths only.

Deploy by rolling out the policy for T steps from ε*(nearest) for each
Phase 2 goal, FDFD-verifying.

Usage:
    modal run train_phase2_anchor_traj.py::main \\
        --memory-bank phase1-uniform-init-output \\
        --surrogate pretrain/M_fdfd_surrogate.pt \\
        --n-iter 50 --pop-size 20 --sigma 0.05 --alpha 0.05 \\
        --t-deploy 20
"""

import json
import os
import pickle
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root (for geometry, simulation)
APP_NAME = os.environ.get("ANCHOR_TRAJ_APP_NAME", "cs224r-anchor-traj")

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
    """FDFD a batch of ε's; return packed list of P[30] arrays."""
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


def _centered_ranks(fitnesses):
    import numpy as np
    K = len(fitnesses)
    ranks = np.argsort(np.argsort(fitnesses))
    return ranks.astype(float) / K - 0.5


def _m_inner_loop_es(eps_init, theta_target, M, log_mean_t, log_std_t,
                     n_iter, K, sigma, alpha, rng):
    """State-space ES on ε with M-only inner loop. Returns trajectory
    of mean ε's visited."""
    import numpy as np
    import torch
    eps = eps_init.astype(np.float32, copy=True)
    half_K = K // 2
    trajectory = [eps.copy()]
    for it in range(n_iter):
        xi_half = rng.standard_normal((half_K, *eps.shape)).astype(np.float32)
        xi_pop = np.concatenate([xi_half, -xi_half], axis=0)
        eps_pop = np.clip(eps[None] + sigma * xi_pop, -1.0, 1.0)

        with torch.no_grad():
            eps_pop_t = torch.as_tensor(eps_pop, dtype=torch.float32)
            normed = M(eps_pop_t)
            log_P = normed * log_std_t + log_mean_t
            P_pop = torch.expm1(log_P).clamp(min=0.0).numpy()

        P_target = P_pop[:, theta_target]
        P_total = P_pop.sum(axis=1)
        Q_pop = (P_target ** 2) / (P_total + 1e-9)

        u = _centered_ranks(Q_pop)
        grad = np.einsum('k,kij->ij', u, xi_pop) / (K * sigma)
        eps = np.clip(eps + alpha * grad, -1.0, 1.0).astype(np.float32)
        trajectory.append(eps.copy())
    return trajectory


def _interp_eps(goal, bank):
    """Linear interpolation between nearest Phase 1 anchors."""
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


@app.local_entrypoint()
def main(
    memory_bank: str = "phase1-uniform-init-output",
    surrogate: str = "pretrain/M_fdfd_surrogate.pt",
    n_iter: int = 50,
    pop_size: int = 20,
    sigma: float = 0.05,
    alpha: float = 0.05,
    t_deploy: int = 20,
    out_dir: str = "checkpoint_output/phase2/anchor-traj",
    seed: int = 0,
):
    """Build anchor-pair training data, train policy, eval on Phase 2 goals."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from algorithms.policies.es_policy import ESPolicy, ESPolicyConfig
    from train_V_network import PNetwork

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ---- Load M, memory bank ----
    print(f"Loading memory bank: {memory_bank}")
    bank = _load_memory_bank(Path(memory_bank))
    anchor_angles = sorted(bank.keys())
    print(f"  anchors: {anchor_angles}")

    print(f"Loading M surrogate: {surrogate}")
    ckpt = torch.load(surrogate, weights_only=False)
    M = PNetwork(ckpt["state_shape"], ckpt["n_receivers"],
                 ckpt["hidden_dim"], ckpt["n_hidden_layers"])
    M.load_state_dict(ckpt["model_state_dict"])
    M.eval()
    for p in M.parameters():
        p.requires_grad_(False)
    log_mean_t = torch.as_tensor(ckpt["log_mean"], dtype=torch.float32)
    log_std_t = torch.as_tensor(ckpt["log_std"], dtype=torch.float32)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: M-inner-loop ES for all anchor pairs (local) ----
    pairs = [(a, b) for a in anchor_angles for b in anchor_angles if a != b]
    print(f"\nStage 1: M-inner-loop ES for {len(pairs)} (a, b) pairs  "
          f"(n_iter={n_iter}, K={pop_size}, σ={sigma}, α={alpha})")
    t0 = time.time()
    trajectories = {}
    final_eps_for_fdfd = []   # (label, pair, ε) tuples for batch FDFD
    for (a, b) in pairs:
        eps_init = bank[a]
        traj = _m_inner_loop_es(eps_init, b, M, log_mean_t, log_std_t,
                                n_iter, pop_size, sigma, alpha, rng)
        trajectories[(a, b)] = traj
        final_eps_for_fdfd.append(("es_final", (a, b), traj[-1]))
        # Linear interp endpoint for comparison (= ε*(b) itself for in-bank goals)
        final_eps_for_fdfd.append(("linear_endpoint", (a, b), bank[b]))
    stage1_t = time.time() - t0
    print(f"  M-inner-loop ES finished in {stage1_t:.1f}s "
          f"({len(pairs)} pairs × {n_iter} iters × {pop_size} M evals)")

    # ---- Stage 2: FDFD-verify endpoints (Modal) ----
    print(f"\nStage 2: FDFD verification of {len(final_eps_for_fdfd)} endpoints "
          f"on Modal (batch 10)...")
    eps_only = [e for (_, _, e) in final_eps_for_fdfd]
    batches = [eps_only[i:i + 10] for i in range(0, len(eps_only), 10)]
    batches_packed = [pickle.dumps(b) for b in batches]
    t0 = time.time()
    P_chunks = []
    for chunk in fdfd_batch.map(batches_packed):
        P_chunks.append(pickle.loads(chunk))
    P_all = np.concatenate(P_chunks, axis=0)
    stage2_t = time.time() - t0
    print(f"  FDFD verification finished in {stage2_t:.1f}s")

    # Index into per-pair endpoint results
    P_by_label = {"es_final": {}, "linear_endpoint": {}}
    for i, (label, pair, _) in enumerate(final_eps_for_fdfd):
        P_by_label[label][pair] = P_all[i]

    # ---- Stage 3: select trajectories (ES if better, else linear interp) ----
    n_kept_es = 0
    training_data = []        # list of (state, goal, delta) tuples
    for (a, b) in pairs:
        P_es = P_by_label["es_final"][(a, b)]
        P_lin = P_by_label["linear_endpoint"][(a, b)]
        tf_es = float(P_es[b] / max(P_es.sum(), 1e-9))
        tf_lin = float(P_lin[b] / max(P_lin.sum(), 1e-9))

        if tf_es > tf_lin:
            # use the ES trajectory
            traj = trajectories[(a, b)]
            n_kept_es += 1
        else:
            # use a linear interp trajectory: ε_t = (1-t/N)·ε*(a) + (t/N)·ε*(b)
            N = n_iter
            traj = []
            for t in range(N + 1):
                tau = t / N
                traj.append(((1 - tau) * bank[a] + tau * bank[b]).astype(np.float32))

        # Convert trajectory to (state, goal, delta) samples
        for t in range(len(traj) - 1):
            state = traj[t]
            delta = (traj[t + 1] - traj[t]).astype(np.float32)
            training_data.append((state, b, delta))

    print(f"\nStage 3: trajectory selection")
    print(f"  ES trajectories kept (better than linear): {n_kept_es}/{len(pairs)}")
    print(f"  Total training samples: {len(training_data):,}")

    # ---- Stage 4: train the policy NN ----
    print(f"\nStage 4: train policy NN on anchor-pair trajectories")
    state_shape = tuple(bank[anchor_angles[0]].shape)
    pcfg = ESPolicyConfig(
        hidden_dim=100, n_hidden_layers=2,
        tanh_output=True, tanh_output_scale=1.0,
        n_goals=30, device="cpu", seed=seed,
    )
    policy = ESPolicy(state_shape=state_shape, config=pcfg)
    policy.pi.train()
    n_pi = sum(p.numel() for p in policy.pi.parameters())
    print(f"  policy: hidden=100×2  params={n_pi:,}")

    states = np.stack([s for (s, _, _) in training_data]).astype(np.float32)
    goals = np.array([g for (_, g, _) in training_data], dtype=np.int64)
    deltas = np.stack([d for (_, _, d) in training_data]).astype(np.float32)

    states_t = torch.as_tensor(states, dtype=torch.float32)
    goals_t = torch.as_tensor(goals, dtype=torch.long)
    deltas_t = torch.as_tensor(deltas, dtype=torch.float32)

    opt = optim.Adam(policy.pi.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 100
    batch_size = 256
    for epoch in range(epochs):
        perm = rng.permutation(len(states))
        epoch_loss, n_batch = 0.0, 0
        for i in range(0, len(perm), batch_size):
            bi = torch.as_tensor(perm[i:i + batch_size], dtype=torch.long)
            pred = policy.pi(states_t[bi], goals_t[bi])
            loss = F.mse_loss(pred, deltas_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
            n_batch += 1
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {epoch + 1}/{epochs}: loss = {epoch_loss / n_batch:.4e}")
    policy.pi.eval()

    out_policy = out_path / "policy_anchor_traj.pt"
    policy.save(out_policy)
    print(f"  saved policy → {out_policy}")

    # ---- Stage 5: deploy on Phase 2 goals ----
    phase2_goals = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    print(f"\nStage 5: deploy policy on {len(phase2_goals)} Phase 2 goals "
          f"(T={t_deploy}-step rollout, FDFD verify)")

    deploy_eps_list = []
    for goal in phase2_goals:
        prev = min(anchor_angles, key=lambda a: abs(a - goal))
        eps = bank[prev].astype(np.float32).copy()
        for _ in range(t_deploy):
            delta = policy.predict(eps, goal).astype(np.float32)
            eps = np.clip(eps + delta, -1.0, 1.0).astype(np.float32)
        deploy_eps_list.append((prev, goal, eps))

    # Batch FDFD verify
    print(f"  dispatching {len(deploy_eps_list)} deployment FDFDs to Modal...")
    deploy_eps_packed = [pickle.dumps([e for (_, _, e) in deploy_eps_list])]
    t0 = time.time()
    P_chunks = []
    for chunk in fdfd_batch.map(deploy_eps_packed):
        P_chunks.append(pickle.loads(chunk))
    P_deploy = np.concatenate(P_chunks, axis=0)
    stage5_t = time.time() - t0
    print(f"  deployment FDFDs finished in {stage5_t:.1f}s")

    print()
    print(f"{'goal':>4}  {'prev':>4}  {'warm_tf':>7}  "
          f"{'interp_tf':>9}  {'policy_tf':>9}  {'Δ vs warm':>10}")
    print("-" * 65)
    summary = []
    for i, (prev, goal, eps_final) in enumerate(deploy_eps_list):
        # warm-start FDFD (use bank[prev]) — we don't have it from this batch
        # but we know it from elsewhere; compute via FDFD if not cached. To keep
        # things simple, FDFD again
        pass
    # We already need warm-start and interp FDFDs to compare. Dispatch those too.
    extra_eps = []
    for prev, goal, _ in deploy_eps_list:
        extra_eps.append(bank[prev])
        extra_eps.append(_interp_eps(goal, bank))
    extra_packed = [pickle.dumps(extra_eps)]
    P_chunks = []
    for chunk in fdfd_batch.map(extra_packed):
        P_chunks.append(pickle.loads(chunk))
    P_extra = np.concatenate(P_chunks, axis=0)
    P_warm = P_extra[::2]
    P_interp = P_extra[1::2]

    n = len(deploy_eps_list)
    m_warm = m_interp = m_policy = 0.0
    n_beat_warm = 0
    n_beat_interp = 0
    for i, (prev, goal, eps_final) in enumerate(deploy_eps_list):
        Pw, Pi, Pf = P_warm[i], P_interp[i], P_deploy[i]
        tw = float(Pw[goal] / max(Pw.sum(), 1e-9))
        ti = float(Pi[goal] / max(Pi.sum(), 1e-9))
        tf = float(Pf[goal] / max(Pf.sum(), 1e-9))
        m_warm += tw; m_interp += ti; m_policy += tf
        if tf > tw: n_beat_warm += 1
        if tf > ti: n_beat_interp += 1
        np.save(out_path / f"eps_target_{goal:02d}.npy", eps_final)
        flag = "✓" if tf > tw else "×"
        print(f"  {goal:>2}    {prev:>2}    {tw:.4f}   {ti:.4f}    "
              f"{tf:.4f}    {tf - tw:+.4f}  {flag}")
        summary.append({"goal": goal, "prev": prev,
                        "tf_warm": tw, "tf_interp": ti, "tf_policy": tf})

    print("-" * 65)
    print(f"  mean         {m_warm/n:.4f}   {m_interp/n:.4f}    "
          f"{m_policy/n:.4f}    {(m_policy - m_warm)/n:+.4f}")
    print(f"  beat warm: {n_beat_warm}/{n}    beat interp: {n_beat_interp}/{n}")

    (out_path / "summary.json").write_text(json.dumps({
        "config": {"n_iter": n_iter, "pop_size": pop_size, "sigma": sigma,
                   "alpha": alpha, "t_deploy": t_deploy, "seed": seed},
        "n_es_kept": n_kept_es, "n_total_pairs": len(pairs),
        "mean_tf_warm": m_warm / n, "mean_tf_interp": m_interp / n,
        "mean_tf_policy": m_policy / n,
        "n_beat_warm": n_beat_warm, "n_beat_interp": n_beat_interp,
        "stage1_s": stage1_t, "stage2_s": stage2_t, "stage5_s": stage5_t,
        "per_goal": summary,
    }, indent=2))

    print()
    print("Comparison to other methods:")
    print(f"  warm-start (ε*(nearest))              : 0.137  (—)")
    print(f"  linear interpolation                  : 0.143  (4/10)")
    print(f"  buffer-traj imitation NN              : 0.142  (6/10)")
    print(f"  online state-space ES (M-only)        : 0.147  (4/10)")
    print(f"  anchored online ES                    : 0.185  (7/10)")
    print(f"  ANCHOR-PAIR TRAJECTORY NN (this)      : {m_policy/n:.3f}  "
          f"({n_beat_warm}/{n})")
