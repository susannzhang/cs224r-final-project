"""Phase 1 ES re-run with far-anchor warm-start (fully parallel, saved checkpoints).

For each Phase 1 target angle θ ∈ {0, 3, 6, …, 27}, initialize ε = ε*(θ_far)
where θ_far is the Phase 1 anchor furthest from θ in receiver index space.
Then run state-space ES (same inner loop as Phase 1) for N iters.

Architecture:
  - Top-level: 10 parallel Modal containers (one per goal)
  - Within each: K=20 candidate FDFDs run in parallel via inner Function.map()
  - Peak concurrency: up to 10 × 20 = 200 simultaneous FDFD containers
  - Periodic checkpointing to a Modal Volume — workers resume from their
    last saved state if Modal preempts them or the local script is rerun.

Every FDFD solve's (ε, P[30]) pair is logged to a master training dataset
(format compatible with mean_states_P.npz). This becomes additional
supervised training data for the M(ε)→P[30] surrogate.

Recovery: rerunning this script picks up where the previous run left off.
Per-goal state lives at /buffer/phase1-reinit/<run_id>/target_NN/state.pkl.

Usage:
    modal run train_phase1_reinit_modal.py::main \\
        --memory-bank phase1-uniform-init-output \\
        --n-iter 250 --pop-size 20 --sigma 0.1 --alpha 0.02 \\
        --run-id phase1-reinit-20260602
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
APP_NAME = os.environ.get("PHASE1_REINIT_APP_NAME", "cs224r-phase1-reinit")
VOLUME_NAME = os.environ.get("PHASE1_REINIT_VOLUME", "cs224r-phase1-reinit-buffer")

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
buffer_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


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


@app.function(image=image, cpu=2, memory=4096, timeout=300)
def fdfd_one(eps_packed: bytes) -> bytes:
    """FDFD one ε, return (P[30], P_loss, receiver_indices) packed.
    P_loss is the PML-absorbed power (used for Phase 1's ΔP_loss term)."""
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")
    import numpy as np
    from algorithms.agents.es_agent import apply_eps_to_canvas, compute_P_loss
    from simulation import simulate_ez_fields_per_source

    eps = np.asarray(pickle.loads(eps_packed), dtype=np.float32)
    env = _build_pm_env()
    apply_eps_to_canvas(env, eps)
    ez = sum(simulate_ez_fields_per_source(env).values())
    intensity = np.abs(ez) ** 2
    P = np.array([float(np.sum(intensity * r._mask))
                  for r in env.receivers], dtype=np.float64)
    P_loss = compute_P_loss(env, intensity)
    rx_idx = np.array([int(r.index) for r in env.receivers], dtype=np.int64)
    return pickle.dumps((P, P_loss, rx_idx))


def _centered_ranks(fitnesses):
    import numpy as np
    K = len(fitnesses)
    ranks = np.argsort(np.argsort(fitnesses))
    return ranks.astype(float) / K - 0.5


@app.function(image=image, cpu=2, memory=8192, timeout=60 * 60 * 6,
              volumes={"/buffer": buffer_volume})
def run_phase1_es_one_goal(payload: dict) -> bytes:
    """Run state-space ES for one Phase 1 target angle, with custom init.
    Checkpoints state to /buffer/<run_id>/target_NN/state.pkl every
    `checkpoint_every` iters so the worker can resume from preemption."""
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")
    import numpy as np

    target = int(payload["target"])
    eps_init = np.asarray(payload["eps_init"], dtype=np.float32)
    n_iter = int(payload["n_iter"])
    K = int(payload["pop_size"])
    sigma = float(payload["sigma"])
    alpha = float(payload["alpha"])
    seed = int(payload["seed"])
    checkpoint_every = int(payload.get("checkpoint_every", 25))
    run_id = str(payload["run_id"])

    # Sigma annealing: σ_t linearly decays from sigma_start to sigma_end over
    # `anneal_iters` iters, then stays at sigma_end. If sigma_start is None
    # (default), σ stays constant at `sigma` throughout.
    sigma_start = payload.get("sigma_start")
    sigma_end = payload.get("sigma_end")
    anneal_iters = int(payload.get("anneal_iters", 0))

    def _sigma_at(it):
        if sigma_start is None or anneal_iters <= 0:
            return sigma
        if it >= anneal_iters:
            return float(sigma_end if sigma_end is not None else sigma)
        s0 = float(sigma_start)
        s1 = float(sigma_end if sigma_end is not None else sigma)
        frac = it / max(anneal_iters, 1)
        return (1 - frac) * s0 + frac * s1

    # Phase 1's exact reward weights (algorithms/agents/es_agent.py:ESAgentConfig)
    w_crosstalk = float(payload.get("w_crosstalk", 0.3))
    w_loss = float(payload.get("w_loss", 1e-3))
    w_energy = float(payload.get("w_energy", 0.1))

    def _compute_E_rods(eps_2d):
        return float(np.sum((1.0 - eps_2d) / 2.0))

    half_K = K // 2
    state_dir = Path("/buffer") / run_id / f"target_{target:02d}"
    state_path = state_dir / "state.pkl"
    state_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume from checkpoint if present ---
    iter_start = 0
    eps = eps_init.copy()
    best_eps = eps.copy()
    best_tf = 0.0
    history = []
    eps_buffer = []
    P_buffer = []
    receiver_indices = None
    rng = np.random.default_rng(seed + target * 1000)

    if state_path.exists():
        try:
            state = pickle.loads(state_path.read_bytes())
            if state.get("complete", False):
                print(f"[goal={target}] checkpoint marks COMPLETE — "
                      f"returning saved result", flush=True)
                return pickle.dumps(state["result"])
            iter_start = int(state["iter"])
            eps = np.asarray(state["eps"], dtype=np.float32)
            best_eps = np.asarray(state["best_eps"], dtype=np.float32)
            best_tf = float(state["best_tf"])
            history = list(state["history"])
            eps_buffer = list(state["eps_buffer"])
            P_buffer = list(state["P_buffer"])
            receiver_indices = state.get("receiver_indices")
            rng.bit_generator.state = state["rng_state"]
            print(f"[goal={target}] RESUMING from iter {iter_start}/{n_iter}  "
                  f"best_tf={best_tf:.4f}  buffer={len(eps_buffer):,}",
                  flush=True)
        except Exception as e:
            print(f"[goal={target}] failed to load checkpoint ({e}); "
                  f"starting fresh", flush=True)
            iter_start = 0

    t0 = time.time()

    # Baseline FDFD at the initial state (only on first iter)
    if iter_start == 0:
        P_init_pkl, _, rx_idx = pickle.loads(list(fdfd_one.map([pickle.dumps(eps)]))[0])
        receiver_indices = rx_idx
        eps_buffer.append(eps.copy()); P_buffer.append(P_init_pkl)
        tf_init = float(P_init_pkl[target] / max(P_init_pkl.sum(), 1e-9))
        print(f"[goal={target}] init |ε|={np.linalg.norm(eps):.3f}  "
              f"init target_frac={tf_init:.4f}", flush=True)
        best_tf = max(best_tf, tf_init)

    def _save_checkpoint(iter_done: int, complete: bool, result=None):
        payload_to_save = {
            "iter": iter_done,
            "eps": eps,
            "best_eps": best_eps,
            "best_tf": best_tf,
            "history": history,
            "eps_buffer": eps_buffer,
            "P_buffer": P_buffer,
            "receiver_indices": receiver_indices,
            "rng_state": rng.bit_generator.state,
            "complete": complete,
        }
        if complete and result is not None:
            payload_to_save["result"] = result
        state_path.write_bytes(pickle.dumps(payload_to_save))
        buffer_volume.commit()

    for it in range(iter_start, n_iter):
        xi_half = rng.standard_normal((half_K, *eps.shape)).astype(np.float32)
        xi_pop = np.concatenate([xi_half, -xi_half], axis=0)
        sigma_t = _sigma_at(it)
        eps_pop = np.clip(eps[None] + sigma_t * xi_pop, -1.0, 1.0)

        # Phase 1 reward uses ΔP_loss / ΔE_rods relative to the current
        # ES mean. Eval baseline (ε_mean) + K candidates as ONE parallel
        # FDFD batch of K+1 items.
        E_rods_baseline = _compute_E_rods(eps)
        batches = [pickle.dumps(eps)] + [pickle.dumps(e) for e in eps_pop]
        results = list(fdfd_one.map(batches))
        P_baseline_pkl, P_loss_baseline, rx_idx = pickle.loads(results[0])
        if receiver_indices is None:
            receiver_indices = rx_idx

        P_pop = []
        P_loss_pop = []
        for r in results[1:]:
            P_k, P_loss_k, _ = pickle.loads(r)
            P_pop.append(P_k)
            P_loss_pop.append(P_loss_k)
        P_pop = np.stack(P_pop)              # (K, 30)
        P_loss_pop = np.array(P_loss_pop)    # (K,)

        # Log baseline + candidates into the (ε, P) training buffer
        eps_buffer.append(eps.copy()); P_buffer.append(P_baseline_pkl)
        for k in range(K):
            eps_buffer.append(eps_pop[k].copy())
            P_buffer.append(P_pop[k].copy())

        P_target = P_pop[:, target]
        P_total = P_pop.sum(axis=1)
        P_others = P_total - P_target
        EPS_NUM = 1e-9
        lam_c = np.where(P_total > 0, 1 - P_target / (P_total + EPS_NUM), 1.0)
        # Phase 1's exact reward formula (es_agent.py get_reward mode="absolute"):
        #   r = P_target − w_c · λ_c · P_others
        #         − w_loss · ΔP_loss − w_energy · ΔE_rods
        d_P_loss = P_loss_pop - P_loss_baseline                     # (K,)
        d_E_rods_pop = np.array([
            _compute_E_rods(eps_pop[k]) - E_rods_baseline for k in range(K)
        ])
        rewards = (P_target
                   - w_crosstalk * lam_c * P_others
                   - w_loss * d_P_loss
                   - w_energy * d_E_rods_pop)

        target_fracs = P_target / np.maximum(P_total, EPS_NUM)
        best_in_pop = int(np.argmax(target_fracs))
        if target_fracs[best_in_pop] > best_tf:
            best_tf = float(target_fracs[best_in_pop])
            best_eps = eps_pop[best_in_pop].copy()

        u = _centered_ranks(rewards)
        grad = np.einsum('k,kij->ij', u, xi_pop) / (K * sigma_t)
        eps = np.clip(eps + alpha * grad, -1.0, 1.0).astype(np.float32)

        history.append({
            "iter": it,
            "pop_reward_mean": float(rewards.mean()),
            "pop_reward_best": float(rewards.max()),
            "pop_target_frac_best": float(target_fracs.max()),
            "best_ever_target_frac": best_tf,
        })

        if (it + 1) % checkpoint_every == 0:
            _save_checkpoint(iter_done=it + 1, complete=False)
            elapsed = time.time() - t0
            print(f"[goal={target}] iter {it+1:>3}/{n_iter}  "
                  f"pop_tf_best={target_fracs.max():.4f}  "
                  f"best_ever_tf={best_tf:.4f}  buffer={len(eps_buffer):,}  "
                  f"({elapsed:.0f}s)  → checkpoint", flush=True)

    elapsed = time.time() - t0
    P_mean_pkl, _, _ = pickle.loads(list(fdfd_one.map([pickle.dumps(eps)]))[0])
    eps_buffer.append(eps.copy()); P_buffer.append(P_mean_pkl)
    tf_mean = float(P_mean_pkl[target] / max(P_mean_pkl.sum(), 1e-9))

    print(f"[goal={target}] FINAL  best_eps_tf={best_tf:.4f}  "
          f"mean_eps_tf={tf_mean:.4f}  total_fdfds={len(eps_buffer):,}  "
          f"({elapsed:.0f}s)", flush=True)

    result = {
        "target": target,
        "best_eps": best_eps,
        "best_target_frac": best_tf,
        "final_mean_eps": eps,
        "final_mean_target_frac": tf_mean,
        "history": history,
        "elapsed_s": elapsed,
        "eps_buffer": np.stack(eps_buffer).astype(np.float32),
        "P_buffer": np.stack(P_buffer).astype(np.float64),
        "receiver_indices": receiver_indices,
    }
    _save_checkpoint(iter_done=n_iter, complete=True, result=result)
    return pickle.dumps(result)


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


@app.local_entrypoint()
def main(
    memory_bank: str = "phase1-uniform-init-output",
    n_iter: int = 250,
    pop_size: int = 20,
    sigma: float = 0.1,
    alpha: float = 0.02,
    out_dir: str = "phase1-reinit-output",
    training_data_out: str = "phase1-reinit-output/all_eps_P.npz",
    render_dir: str = "checkpoint_output/phase1-reinit",
    checkpoint_every: int = 25,
    run_id: str = "",
    seed: int = 0,
    goals: str = "",
    anchors_map: str = "",
    sigma_start: float = -1.0,
    sigma_end: float = -1.0,
    anneal_iters: int = 0,
):
    """Dispatch 10 parallel Phase 1 ES runs, each with far-anchor init."""
    import numpy as np
    from datetime import datetime

    if not run_id:
        run_id = "phase1-reinit-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"run_id: {run_id}  (used as Volume subdirectory for checkpoints)")

    bank = _load_memory_bank(Path(memory_bank))
    anchors = sorted(bank.keys())
    print(f"Memory bank anchors: {anchors}")

    # Optional explicit "target:prev" overrides of the far-anchor default,
    # e.g. --anchors-map "0:3,9:6,12:15,18:24,21:27" to warm-start each target
    # from a NEARBY angle's ε* instead of the far one.
    custom_anchor = {}
    if anchors_map:
        for pair in anchors_map.split(","):
            t, p = pair.split(":")
            custom_anchor[int(t)] = int(p)
        for t, p in custom_anchor.items():
            if p not in bank:
                raise ValueError(f"anchor prev={p} for target {t} not in memory bank {anchors}")
        print(f"Custom anchor overrides (target -> prev): {custom_anchor}")

    def far_anchor(target):
        opp = (target + 15) % 30
        def ring_dist(a):
            d = abs(a - opp)
            return min(d, 30 - d)
        return min(anchors, key=ring_dist)

    # Subset goals if `goals` CLI flag is provided
    if goals:
        wanted = set(int(x) for x in goals.split(","))
        anchors_to_run = [a for a in anchors if a in wanted]
        if not anchors_to_run:
            raise ValueError(f"None of --goals={goals} in anchors {anchors}")
    else:
        anchors_to_run = anchors

    use_anneal = sigma_start > 0 and anneal_iters > 0

    payloads = []
    print(f"\nGoal → far-anchor init mapping:")
    for target in anchors_to_run:
        prev = custom_anchor.get(target, far_anchor(target))
        if prev == target:
            prev = anchors[(anchors.index(target) + 5) % len(anchors)]
        d = (prev - target) % 30
        ring_d = min(d, 30 - d)
        print(f"  target={target:>2}  ←  ε*(prev={prev:>2})  "
              f"(receiver-index distance {ring_d})")
        p = {
            "target": target,
            "eps_init": np.asarray(bank[prev]),
            "n_iter": n_iter,
            "pop_size": pop_size,
            "sigma": sigma,
            "alpha": alpha,
            "checkpoint_every": checkpoint_every,
            "run_id": run_id,
            "seed": seed,
        }
        if use_anneal:
            p["sigma_start"] = float(sigma_start)
            p["sigma_end"] = float(sigma_end if sigma_end > 0 else sigma)
            p["anneal_iters"] = int(anneal_iters)
        payloads.append(p)
    if use_anneal:
        s_end = sigma_end if sigma_end > 0 else sigma
        print(f"\nσ annealing: {sigma_start} → {s_end} over first "
              f"{anneal_iters} iters, then constant.")

    print(f"\nConfig: n_iter={n_iter}  K={pop_size}  σ={sigma}  α={alpha}  "
          f"checkpoint_every={checkpoint_every}")
    print(f"Total FDFDs across all 10 goals: {len(payloads) * n_iter * pop_size:,}")
    print(f"Peak concurrent containers: {len(payloads) * pop_size} (10 outer × K inner)")
    print(f"Checkpoints → modal Volume '{VOLUME_NAME}' at "
          f"/buffer/{run_id}/target_NN/state.pkl")
    print(f"\nDispatching {len(payloads)} parallel ES runs to Modal...\n")

    t0 = time.time()
    results = []
    for chunk_bytes in run_phase1_es_one_goal.map(payloads):
        result = pickle.loads(chunk_bytes)
        results.append(result)
    overall = time.time() - t0
    print(f"\nAll 10 ES runs completed in {overall:.1f}s ({overall/60:.1f} min)")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    receiver_indices = None
    for r in results:
        td = out_path / f"target_{r['target']:02d}"
        td.mkdir(parents=True, exist_ok=True)
        np.save(td / "eps_star.npy", r["best_eps"])
        np.save(td / "eps_mean_final.npy", r["final_mean_eps"])
        (td / "history.json").write_text(json.dumps({
            "target_idx": r["target"],
            "best_target_frac": r["best_target_frac"],
            "final_mean_target_frac": r["final_mean_target_frac"],
            "iterations": len(r["history"]),
            "elapsed_s": r["elapsed_s"],
            "history": r["history"],
        }, indent=2))
        if receiver_indices is None:
            receiver_indices = r["receiver_indices"]

    all_eps = np.concatenate([r["eps_buffer"] for r in results], axis=0)
    all_P = np.concatenate([r["P_buffer"] for r in results], axis=0)
    Path(training_data_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(training_data_out,
             eps=all_eps, P=all_P, receiver_indices=receiver_indices)
    print(f"\nSaved {len(all_eps):,} (ε, P) pairs → {training_data_out}")
    print(f"  eps: {all_eps.shape}  P: {all_P.shape}")

    print()
    print(f"{'target':>6}  {'orig_tf':>7}  {'reinit_tf':>9}  Δ        verdict")
    print("-" * 65)
    n_better = n_worse = n_same = 0
    summary = []
    for r in sorted(results, key=lambda x: x["target"]):
        target = r["target"]
        tf_reinit = r["best_target_frac"]
        meta_path = Path(memory_bank) / f"target_{target:02d}" / "metadata.json"
        tf_orig = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                tf_orig = meta.get("best_ever_target_fraction") \
                          or meta.get("best_target_fraction")
            except Exception:
                pass
        if tf_orig is None:
            verdict = "?"; delta = float("nan")
        else:
            delta = tf_reinit - tf_orig
            if abs(delta) < 0.01:
                verdict = "≈ same"; n_same += 1
            elif delta > 0:
                verdict = "✓ better"; n_better += 1
            else:
                verdict = "✗ worse"; n_worse += 1
        tf_orig_s = f"{tf_orig:.4f}" if tf_orig is not None else "?"
        delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "?"
        print(f"  {target:>2}    {tf_orig_s:>7}   {tf_reinit:.4f}    "
              f"{delta_s}   {verdict}")
        summary.append({
            "target": target, "tf_original": tf_orig, "tf_reinit": tf_reinit,
            "delta": delta if not np.isnan(delta) else None,
            "verdict": verdict,
        })
    print("-" * 65)
    print(f"  better: {n_better}    worse: {n_worse}    same: {n_same}")

    (out_path / "summary.json").write_text(json.dumps({
        "run_id": run_id,
        "config": {"n_iter": n_iter, "pop_size": pop_size,
                   "sigma": sigma, "alpha": alpha, "seed": seed,
                   "checkpoint_every": checkpoint_every},
        "overall_wall_s": overall,
        "total_fdfd_samples": int(len(all_eps)),
        "training_data_path": str(training_data_out),
        "per_goal": summary,
    }, indent=2))
    print(f"\nSaved → {out_path.resolve()}")

    # --- Render each goal's reinit ε* as a 2x2 PNG ---
    print(f"\nRendering PNGs of reinit ε* states → {render_dir}/")
    render_path = Path(render_dir)
    render_path.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from render_phase2_ckpt import build_pm_env, render_2x2_png
    env = build_pm_env()
    bank_orig = _load_memory_bank(Path(memory_bank))

    def far_anchor_local(target):
        opp = (target + 15) % 30
        def ring_dist(a):
            d = abs(a - opp)
            return min(d, 30 - d)
        return min(sorted(bank_orig.keys()), key=ring_dist)

    render_summary = []
    for r in sorted(results, key=lambda x: x["target"]):
        target = r["target"]
        prev = custom_anchor.get(target, far_anchor_local(target))
        eps_init = bank_orig[prev]
        eps_final = r["best_eps"]
        tf_orig = next((s["tf_original"] for s in summary
                        if s["target"] == target), None)
        tf_orig_disp = f"{tf_orig:.3f}" if tf_orig is not None else "?"
        suffix = (f"[reinit] from ε*(prev={prev})  "
                  f"tf orig={tf_orig_disp}  "
                  f"reinit best={r['best_target_frac']:.3f}")
        save_path = render_path / f"target_{target:02d}.png"
        P_i, P_f = render_2x2_png(env, eps_init, eps_final, target,
                                  save_path, title_suffix=suffix)
        ti = float(P_i[target] / max(P_i.sum(), 1e-9))
        tf = float(P_f[target] / max(P_f.sum(), 1e-9))
        print(f"  target={target:>2}  tf: {ti:.4f} → {tf:.4f}  → {save_path.name}")
        render_summary.append({
            "target": target, "prev_init": prev,
            "target_frac_init": ti, "target_frac_final": tf,
            "png": str(save_path),
        })
    (render_path / "viz_summary.json").write_text(json.dumps({
        "method": "phase1_reinit_far_anchor",
        "run_id": run_id,
        "results": render_summary,
    }, indent=2))
    print(f"\nWrote {len(render_summary)} PNGs to {render_path.resolve()}")
