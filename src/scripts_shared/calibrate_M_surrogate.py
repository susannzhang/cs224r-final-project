"""Calibration test for the M(ε) → P[30] FDFD surrogate, parallelized on Modal.

Tests how well M tracks FDFD on STATES THE PHASE 2 POLICY ACTUALLY VISITS
during a closed-loop rollout. M was trained on Phase 1's ES mean trajectory,
which is a different distribution from "warm-start at ε*(prev) + policy
outputs"; this script quantifies the OOD drift.

All policy rollout planning is done LOCALLY (no FDFD needed — policy + state
transitions are deterministic). All FDFD ground-truth evaluations are
batched and dispatched to Modal in parallel.

Two tests:
  (1) Rollout calibration: M-predicted P vs FDFD-true P at each step of a
      T-step policy rollout. Reports per-step log-corr, Q-error, argmax
      agreement; trajectory total Q error.
  (2) Action-ranking test: at mid-trajectory states, generate K candidate
      δ's, rank by M and by FDFD. Reports top-1 / top-3 agreement and
      Spearman ρ (the metric that matters for planning).

Usage:
    modal run calibrate_M_surrogate.py::main \\
        --surrogate pretrain/M_fdfd_surrogate.pt \\
        --policy pretrain/policy_buffer_traj_h100l2.pt
"""

import os
import pickle
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root (for geometry, simulation)
APP_NAME = os.environ.get("CALIBRATE_APP_NAME", "cs224r-calibrate-M")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install("numpy", "scipy", "scikit-image", "autograd", "ceviche")
    .add_local_dir(
        PROJECT_ROOT, "/root/app", copy=True,
        ignore=["__pycache__", "*.pyc", "phase1_checkpoints",
                "phase1_training_output", "checkpoint_output",
                "phase2_tiny_smoke", "phase2_output", "phase2_parallel_output",
                "tests/visual_output", ".git", ".venv", ".pytest_cache",
                "*.pkl", "wandb"],
    )
)
app = modal.App(APP_NAME)


def _build_pm_env():
    """Same env Phase 1/2 use; mirrors train_phase2_parallel_modal.py."""
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
    receiver_indices = np.array([int(r.index) for r in env.receivers],
                                dtype=np.int64)
    P_list = []
    for eps in eps_batch:
        eps = np.asarray(eps, dtype=np.float32)
        apply_eps_to_canvas(env, eps)
        ez = sum(simulate_ez_fields_per_source(env).values())
        intensity = np.abs(ez) ** 2
        P = np.array([float(np.sum(intensity * r._mask))
                      for r in env.receivers], dtype=np.float64)
        P_list.append(P)
    return pickle.dumps((np.stack(P_list), receiver_indices))


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
    surrogate: str = "pretrain/M_fdfd_surrogate.pt",
    policy: str = "pretrain/policy_buffer_traj_h100l2.pt",
    memory_bank: str = "phase1-uniform-init-output",
    pairs: str = "0:1,3:4,12:13,24:25",
    n_steps: int = 20,
    n_candidates: int = 20,
    batch_size: int = 25,
):
    """Build all FDFD-needed states locally; batch-dispatch to Modal."""
    import numpy as np
    import torch
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from algorithms.policies.es_policy import ESPolicy
    from train_V_network import PNetwork, predict_P

    # ---- Load local artifacts ----
    print(f"Loading M surrogate: {surrogate}")
    ckpt = torch.load(surrogate, weights_only=False)
    M = PNetwork(ckpt["state_shape"], ckpt["n_receivers"],
                 ckpt["hidden_dim"], ckpt["n_hidden_layers"])
    M.load_state_dict(ckpt["model_state_dict"])
    M.eval()
    log_mean = torch.as_tensor(ckpt["log_mean"], dtype=torch.float32)
    log_std = torch.as_tensor(ckpt["log_std"], dtype=torch.float32)

    print(f"Loading policy: {policy}")
    pol = ESPolicy.load(policy)

    print(f"Loading memory bank: {memory_bank}")
    bank = _load_memory_bank(Path(memory_bank))
    print(f"  angles: {sorted(bank.keys())}")

    pair_list = [tuple(int(x) for x in p.split(":")) for p in pairs.split(",")]
    pair_list = [(a, b) for a, b in pair_list if a in bank]
    print(f"  pairs ({len(pair_list)}): {pair_list}")

    # ---- Phase 1: simulate rollouts LOCALLY, collect states needing FDFD ----
    rollout_states = []          # list of ε's
    rollout_meta = []            # list of (pair_idx, step_idx)

    print(f"\nSimulating {len(pair_list)} policy rollouts locally (T={n_steps})...")
    for pi, (prev, target) in enumerate(pair_list):
        eps = bank[prev].astype(np.float32).copy()
        for t in range(n_steps):
            delta = pol.predict(eps, target).astype(np.float32)
            eps_next = np.clip(eps + delta, -1.0, 1.0).astype(np.float32)
            rollout_states.append(eps_next.copy())
            rollout_meta.append((pi, t))
            eps = eps_next
    print(f"  {len(rollout_states)} rollout states collected")

    # ---- Phase 2: action-ranking candidates ----
    rng = np.random.default_rng(0)
    sigma = 0.05
    cand_states = []             # list of ε's
    cand_meta = []               # list of (pair_idx, cand_idx)
    mid_states = []              # the state each candidate batch is anchored at

    print(f"Generating {len(pair_list)}×{n_candidates} action candidates...")
    for pi, (prev, target) in enumerate(pair_list):
        # Roll out to mid-trajectory locally
        eps = bank[prev].astype(np.float32).copy()
        for _ in range(n_steps // 2):
            delta = pol.predict(eps, target).astype(np.float32)
            eps = np.clip(eps + delta, -1.0, 1.0).astype(np.float32)
        mid_states.append(eps.copy())

        base_delta = pol.predict(eps, target).astype(np.float32)
        for k in range(n_candidates):
            noise = rng.normal(0, sigma, size=eps.shape).astype(np.float32)
            d = (base_delta + noise).astype(np.float32)
            cand_eps = np.clip(eps + d, -1.0, 1.0).astype(np.float32)
            cand_states.append(cand_eps)
            cand_meta.append((pi, k))
    print(f"  {len(cand_states)} candidate states collected")

    # ---- Phase 3: batch-dispatch to Modal for FDFD ----
    all_states = rollout_states + cand_states
    n_rollout = len(rollout_states)
    print(f"\nDispatching {len(all_states)} FDFDs to Modal "
          f"({(len(all_states) + batch_size - 1) // batch_size} batches × "
          f"{batch_size}/batch)...")
    batches = [all_states[i:i + batch_size]
               for i in range(0, len(all_states), batch_size)]
    batches_packed = [pickle.dumps(b) for b in batches]

    import time
    t0 = time.time()
    P_chunks = []
    for i, chunk in enumerate(fdfd_batch.map(batches_packed)):
        P_chunk, _ = pickle.loads(chunk)
        P_chunks.append(P_chunk)
    P_all_true = np.concatenate(P_chunks, axis=0)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")
    print(f"  P_all_true shape: {P_all_true.shape}")

    # Split back into rollout vs candidate sets
    P_rollout_true = P_all_true[:n_rollout]
    P_cand_true = P_all_true[n_rollout:]

    # ---- Phase 4: compute M predictions LOCALLY ----
    with torch.no_grad():
        all_states_t = torch.as_tensor(np.stack(all_states),
                                       dtype=torch.float32)
        P_all_pred = predict_P(M, all_states_t, log_mean, log_std).numpy()
    P_rollout_pred = P_all_pred[:n_rollout]
    P_cand_pred = P_all_pred[n_rollout:]

    # ============= TEST 1: ROLLOUT CALIBRATION =============
    print(f"\n{'=' * 78}")
    print(f"  Test 1: Rollout calibration ({len(pair_list)} pairs × T={n_steps})")
    print(f"{'=' * 78}\n")

    def retarget_Q(P, goal):
        return float((P[goal] ** 2) / (P.sum() + 1e-9))

    all_log_corrs, all_rel_errs, all_total_errs = [], [], []
    all_argmax_agree = []
    for pi, (prev, target) in enumerate(pair_list):
        idx = [i for i, m in enumerate(rollout_meta) if m[0] == pi]
        P_pred_p = P_rollout_pred[idx]
        P_true_p = P_rollout_true[idx]
        print(f"--- (prev={prev}, target={target}) ---")
        print(f"  step  Q_M     Q_FDFD  rel_err  arg_true  M_arg→FDFD_arg")
        Q_M_total = 0.0
        Q_T_total = 0.0
        for t in range(n_steps):
            Pp = P_pred_p[t]
            Pt = P_true_p[t]
            Q_M = retarget_Q(Pp, target)
            Q_T = retarget_Q(Pt, target)
            Q_M_total += Q_M
            Q_T_total += Q_T
            rel = abs(Q_M - Q_T) / max(Q_T, 1e-9)
            log_corr = float(np.corrcoef(np.log1p(Pp), np.log1p(Pt))[0, 1])
            agree = int(Pp.argmax()) == int(Pt.argmax())
            all_log_corrs.append(log_corr)
            all_rel_errs.append(rel)
            all_argmax_agree.append(agree)
            print(f"  {t:>3}   {Q_M:>7.2f}  {Q_T:>7.2f}  {rel:>6.1%}  "
                  f"{int(Pt.argmax()):>2}        "
                  f"{int(Pp.argmax()):>2}→{int(Pt.argmax()):<2}  "
                  f"{'✓' if agree else '✗'}")
        total_err = abs(Q_M_total - Q_T_total) / max(Q_T_total, 1e-9)
        all_total_errs.append(total_err)
        print(f"  TOTAL  Q_M={Q_M_total:.2f}  Q_FDFD={Q_T_total:.2f}  "
              f"rel_err={total_err:.1%}\n")

    print(f"Test 1 summary across {len(all_rel_errs)} steps:")
    print(f"  per-step log-corr:    mean={np.mean(all_log_corrs):.4f}  "
          f"min={np.min(all_log_corrs):.4f}")
    print(f"  per-step Q rel err:   median={np.median(all_rel_errs):.1%}  "
          f"mean={np.mean(all_rel_errs):.1%}  max={np.max(all_rel_errs):.1%}")
    print(f"  per-step argmax agreement: "
          f"{sum(all_argmax_agree)}/{len(all_argmax_agree)} "
          f"({100 * np.mean(all_argmax_agree):.0f}%)")
    print(f"  total Q rel err:      mean={np.mean(all_total_errs):.1%}  "
          f"max={np.max(all_total_errs):.1%}")

    # ============= TEST 2: ACTION-RANKING =============
    print(f"\n{'=' * 78}")
    print(f"  Test 2: Action-ranking ({n_candidates} candidates per state)")
    print(f"  Does argmax_a r(M(s+a)) agree with argmax_a r(FDFD(s+a))?")
    print(f"{'=' * 78}\n")

    top1s, top3s, rhos = [], [], []
    for pi, (prev, target) in enumerate(pair_list):
        idx = [i for i, m in enumerate(cand_meta) if m[0] == pi]
        Pc_pred = P_cand_pred[idx]
        Pc_true = P_cand_true[idx]
        Q_M = (Pc_pred[:, target] ** 2) / (Pc_pred.sum(axis=1) + 1e-9)
        Q_T = (Pc_true[:, target] ** 2) / (Pc_true.sum(axis=1) + 1e-9)
        rank_M = np.argsort(-Q_M)
        rank_T = np.argsort(-Q_T)
        top1 = int(rank_M[0] == rank_T[0])
        top3 = int(rank_M[0] in rank_T[:3])
        rho = float(np.corrcoef(
            np.argsort(np.argsort(-Q_M)),
            np.argsort(np.argsort(-Q_T))
        )[0, 1])
        top1s.append(top1)
        top3s.append(top3)
        rhos.append(rho)
        print(f"  (prev={prev}, target={target}):  "
              f"M_best Q={Q_M[rank_M[0]]:>7.2f}  (FDFD says: {Q_T[rank_M[0]]:>7.2f})    "
              f"FDFD_best Q={Q_T[rank_T[0]]:>7.2f}    "
              f"top1={'✓' if top1 else '✗'}  top3={'✓' if top3 else '✗'}  ρ={rho:.3f}")

    print(f"\nTest 2 summary:")
    print(f"  top-1 agreement: {sum(top1s)}/{len(top1s)}")
    print(f"  top-3 agreement: {sum(top3s)}/{len(top3s)}")
    print(f"  Spearman ρ:      mean={np.mean(rhos):.4f}  min={np.min(rhos):.4f}")
