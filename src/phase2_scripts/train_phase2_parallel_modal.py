# =============================================================================
# Phase 2 (parallel) — K-wise parallel rollouts on Modal
# =============================================================================
"""
K-wise parallel version of Phase 2 ES. Each outer iteration dispatches K
rollouts to K separate Modal containers via Function.map(); they run
concurrently. The driver collects fitnesses, applies the centered-rank
ES gradient on φ, and dispatches the next iter.

Wall time per iter:
    K × T × FDFD_time   (sequential, current train_phase2_modal.py)
    →
    T × FDFD_time       (parallel, here)

A ~20× speedup at K=20 — turns ~167 min/iter into ~9 min/iter so a full
24h budget can do 150+ ES iters instead of 8.

Architecture:
    rollout_one(payload):   one T-step trajectory → fitness + transitions
    driver(payload):        ES outer loop, dispatches K rollouts via .map()
    collect(launch_id):     local entrypoint to download results from Volume

Workflow:
    1. modal deploy train_phase2_parallel_modal.py        (one-time)
    2. python spawn_phase2_parallel.py ...                 (queues driver)
    3. modal run train_phase2_parallel_modal.py::collect \\
         --launch-id <id>                                  (download)

The driver lives entirely on Modal, not locally — so local disconnects
can't kill it (we learned that lesson in Phase 1).
"""

import io
import json
import os
import pickle
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root (for geometry, simulation)
APP_NAME = os.environ.get("PHASE2_APP_NAME", "cs224r-phase2-parallel")
VOLUME_NAME = os.environ.get("PHASE2_VOLUME_NAME", "cs224r-phase2-parallel-buffer")

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
                           "phase2_output", "tests/visual_output",
                           ".git", ".venv", ".pytest_cache"])
)

app = modal.App(APP_NAME)
buffer_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

try:
    wandb_secret = modal.Secret.from_name("wandb")
except modal.exception.NotFoundError:
    wandb_secret = None


# =============================================================================
# Env builder — kept in sync with train_phase1_modal.py
# =============================================================================

def _build_pm_env():
    """Build the standard 10×10 pm_setup env with 30 receivers."""
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
# rollout_one — one T-step rollout under a perturbed policy
# =============================================================================

@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60,                # 1 hour per rollout
    # No /buffer volume — rollout returns results, doesn't write to disk.
)
def rollout_one(payload: dict) -> dict:
    """One T-step closed-loop rollout under a perturbed policy.

    Receives perturbed policy weights (flat float32 array) + the starting
    ε_0 + the goal index. Runs T FDFD steps, returns:
        {
            "fitness":         float (Σ_t r_t under the trajectory's goal),
            "rollout_length":  int (≤ T, may be shorter from early termination),
            "transitions_pkl": bytes (pickled list[Transition] — multi-goal
                                      labeled, |logged_reward_indices| per step),
        }
    """
    import sys
    import time
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")

    import numpy as np
    import torch
    import torch.nn.utils as nu
    from types import SimpleNamespace

    # The rollout/reward/termination logic is the SHARED phase2_rollout —
    # identical to ESPolicy.train_phase2, so this worker can never drift.
    from algorithms.policies.es_policy import (
        ESPolicy, ESPolicyConfig, phase2_rollout, phase2_discounted_return,
    )

    # --- Unpack payload ---------------------------------------------
    flat_params_arr = np.asarray(payload["flat_params"], dtype=np.float32)
    policy_config_dict = payload["policy_config"]
    state_shape = tuple(payload["state_shape"])
    goal = int(payload["goal"])
    eps_0 = np.asarray(payload["eps_0"], dtype=np.float32)
    logged_reward_indices = payload["logged_reward_indices"]
    T = int(payload["T"])
    eta = float(payload["eta"])
    w_crosstalk = float(payload["w_crosstalk"])
    w_loss = float(payload["w_loss"])
    w_energy = float(payload["w_energy"])
    reward_mode = str(payload.get("reward_mode", "target_frac"))
    target_frac_scale = float(payload.get("target_frac_scale", 1.0e+5))
    gamma = float(payload.get("gamma", 1.0))
    p_source = float(payload.get("p_source", 1.0))
    hold_threshold = float(payload.get("hold_threshold", 0.9))
    hold_bonus = float(payload.get("hold_bonus", 1.0))

    # --- Rebuild policy + install perturbed params -----------------
    pcfg = ESPolicyConfig(**policy_config_dict)
    policy = ESPolicy(state_shape=state_shape, config=pcfg)
    flat_t = torch.from_numpy(flat_params_arr).to(policy.device)
    nu.vector_to_parameters(flat_t, policy.pi.parameters())

    # --- Build env --------------------------------------------------
    env = _build_pm_env()

    # --- Rollout via the SHARED phase2_rollout ----------------------
    # Single source of truth with ESPolicy.train_phase2: reward modes
    # (incl. reach_hold / source_norm), the γ·Q'−Q telescoping, the γ-discounted
    # fitness, and the early-termination gate all come from one place, so this
    # worker can never silently diverge from the in-process trainer again.
    cfg = SimpleNamespace(
        T=T, eta=eta, reward_mode=reward_mode, gamma=gamma,
        w_crosstalk=w_crosstalk, w_loss=w_loss, w_energy=w_energy,
        target_frac_scale=target_frac_scale,
        hold_threshold=hold_threshold, hold_bonus=hold_bonus, p_source=p_source,
    )
    t0 = time.time()
    traj_eps, _, traj_rewards, transitions, step_P_pairs = phase2_rollout(
        env, policy.predict, eps_0, goal, cfg, logged_reward_indices)
    rollout_length = len(traj_eps)
    elapsed = time.time() - t0
    fitness = phase2_discounted_return(traj_rewards, gamma)

    transitions_pkl = pickle.dumps(transitions)
    # per_step_data: list of (eps_next, P_next) tuples for online M training.
    # Each rollout contributes ≤T pairs (terminated early if eta hit).
    return {
        "fitness": fitness,
        "rollout_length": rollout_length,
        "transitions_pkl": transitions_pkl,
        "per_step_data_pkl": pickle.dumps(step_P_pairs),
        "elapsed_seconds": elapsed,
    }


# =============================================================================
# driver — ES outer loop, dispatches K rollouts in parallel per iter
# =============================================================================

@app.function(
    image=image,
    cpu=2,
    memory=32768,                   # 32 GB — driver aggregates all rollout
                                    # transitions in memory. At K=100, T=50,
                                    # union=20: ~250 MB/iter × N_iter.
    timeout=60 * 60 * 24,           # 24 hours max — Modal cap
    volumes={"/buffer": buffer_volume},
    secrets=[wandb_secret] if wandb_secret is not None else [],
)
def driver(payload: dict) -> dict:
    """ES outer loop. Per iter:
        1. Sample K mirrored noise vectors ξ on policy params
        2. Sample K (goal, ε_0) pairs
        3. Build K payloads with perturbed φ + ε_0 + goal
        4. rollout_one.map(payloads) — dispatch K parallel rollouts
        5. Collect fitnesses, do centered-rank ES update on φ
        6. Append transitions, log to wandb, repeat
    """
    import sys
    import time
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")

    import numpy as np
    import torch
    import torch.nn.utils as nu

    from algorithms.agents.es_agent import centered_ranks
    from algorithms.infrastructure.utils import ReplayBuffer
    from algorithms.policies.es_policy import ESPolicy, ESPolicyConfig, Phase2Config

    # --- Unpack payload ------------------------------------------
    launch_id = payload["launch_id"]
    goal_indices = list(payload["goal_indices"])
    converged_indices = payload.get("converged_indices")
    config_kwargs = payload["config_kwargs"]
    policy_bytes = payload.get("policy_bytes")
    policy_config_kwargs = payload.get("policy_config_kwargs")
    memory_bank_arrays = payload["memory_bank"]
    wandb_cfg = payload.get("wandb")
    # M-filter (optional): path to FDFD surrogate, K to save, max attempts
    m_surrogate_path = payload.get("m_surrogate_path", None)
    m_filter_K = int(payload.get("m_filter_K", 10))
    m_filter_max_attempts = int(payload.get("m_filter_max_attempts", 50))

    # --- Wandb init ----------------------------------------------
    wandb_run = None
    if wandb_cfg is not None and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_run = wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity") or None,
            name=f"phase2-parallel-{launch_id}",
            group=launch_id,
            tags=["parallel",
                  f"K={config_kwargs['K']}",
                  f"T={config_kwargs['T']}",
                  f"N_iter={config_kwargs['N_iter']}"],
            config={**config_kwargs,
                    "goal_indices": goal_indices,
                    "converged_indices": converged_indices,
                    "architecture": "K-parallel via rollout_one.map()"},
            reinit=True,
        )

    # --- Build policy (warm-start or fresh) ---------------------
    if policy_bytes is not None:
        ckpt = torch.load(io.BytesIO(policy_bytes), map_location="cpu",
                          weights_only=False)
        state_shape = ckpt["state_shape"]
        policy = ESPolicy(state_shape=state_shape, config=ckpt["config"])
        policy.pi.load_state_dict(ckpt["pi_state_dict"])
        print(f"[driver] loaded warm-start policy; state_shape={state_shape}",
              flush=True)
    else:
        if policy_config_kwargs is None:
            raise ValueError("Need policy_bytes OR policy_config_kwargs.")
        state_shape = tuple(policy_config_kwargs["state_shape"])
        pcfg = ESPolicyConfig(
            policy_arch=policy_config_kwargs.get("policy_arch", "cnn"),
            hidden_dim=policy_config_kwargs["hidden_dim"],
            n_hidden_layers=policy_config_kwargs["n_hidden_layers"],
            tanh_output=policy_config_kwargs["tanh_output"],
            tanh_output_scale=policy_config_kwargs.get("tanh_output_scale", 1.0),
            n_goals=policy_config_kwargs["n_goals"],
            seed=config_kwargs.get("seed", 0),
        )
        policy = ESPolicy(state_shape=state_shape, config=pcfg)
        print(f"[driver] built fresh policy; state_shape={state_shape}  "
              f"n_goals={pcfg.n_goals}  "
              f"tanh_scale={pcfg.tanh_output_scale}", flush=True)

    # The dict to ship to each rollout worker (architecture only — weights
    # are sent separately per call).
    policy_config_for_workers = {
        # Resolve arch the SAME way ESPolicy.__init__ does (vars() — not getattr,
        # which the dataclass class-default would shadow). Workers must rebuild
        # the EXACT arch the driver holds, or vector_to_parameters mismatches.
        "policy_arch": vars(policy.config).get("policy_arch", "mlp"),
        "hidden_dim": policy.config.hidden_dim,
        "n_hidden_layers": policy.config.n_hidden_layers,
        "tanh_output": policy.config.tanh_output,
        "tanh_output_scale": getattr(policy.config, "tanh_output_scale", 1.0),
        "n_goals": policy.config.n_goals,
        "seed": policy.config.seed,
    }

    # --- M (FDFD surrogate) for pre-filter + online training -----
    M = None
    M_log_mean = None
    M_log_std = None
    M_optimizer = None
    M_buffer_states = []        # numpy ε arrays, shape (10, 10) each
    M_buffer_P = []             # numpy P arrays, shape (30,) each
    M_train_batch_size = 256
    M_train_epochs_per_iter = 5
    if m_surrogate_path is not None:
        import torch.nn.functional as F
        from train_V_network import PNetwork
        M_ckpt = torch.load(m_surrogate_path, map_location="cpu",
                            weights_only=False)
        M = PNetwork(M_ckpt["state_shape"], M_ckpt["n_receivers"],
                     M_ckpt["hidden_dim"], M_ckpt["n_hidden_layers"])
        M.load_state_dict(M_ckpt["model_state_dict"])
        M.eval()
        M_log_mean = torch.as_tensor(M_ckpt["log_mean"], dtype=torch.float32)
        M_log_std = torch.as_tensor(M_ckpt["log_std"], dtype=torch.float32)
        M_optimizer = torch.optim.Adam(M.parameters(), lr=1e-3,
                                       weight_decay=1e-5)
        # Seed M's training buffer from spawn-shipped bytes (the 2,461 Phase 1
        # mean states with P[30] from compute_P_per_state_sampled.py). Shipped
        # over the wire because phase1-uniform-init-output/ is excluded from
        # the Modal image build.
        init_data_bytes = payload.get("m_training_data_bytes")
        if init_data_bytes:
            init_data = np.load(io.BytesIO(init_data_bytes))
            M_buffer_states.extend(list(init_data["eps"]))
            M_buffer_P.extend(list(init_data["P"]))
            print(f"[driver] M training buffer seeded from payload: "
                  f"{len(M_buffer_states):,} pairs", flush=True)
        else:
            print(f"[driver] no seed data shipped; starting from empty M "
                  f"training buffer (will fill from FDFD rollouts)",
                  flush=True)
        print(f"[driver] M surrogate loaded: {m_surrogate_path}  "
              f"K_save={m_filter_K}  max_attempts={m_filter_max_attempts}",
              flush=True)

    # --- Memory bank --------------------------------------------
    memory_bank = {int(k): np.asarray(v, dtype=np.float32)
                   for k, v in memory_bank_arrays.items()}
    print(f"[driver] memory bank: {sorted(memory_bank.keys())}", flush=True)

    if converged_indices is None:
        converged_indices = list(memory_bank.keys())
    N_RECEIVERS = 30
    logged_reward_indices = list(range(N_RECEIVERS))
    print(f"[driver] goal_indices: {goal_indices}", flush=True)
    print(f"[driver] converged_indices: {converged_indices}", flush=True)
    print(f"[driver] logged_reward_indices: all {N_RECEIVERS} receivers",
          flush=True)

    # --- source_norm: auto-calibrate the fixed denominator ON MODAL --------
    # No p_source in config_kwargs ⇒ launcher used the auto sentinel. Compute
    # the free-space total power once here (one FDFD solve) so the run stays
    # fully self-contained on Modal — nothing calibrated locally.
    if config_kwargs.get("reward_mode") == "source_norm" and "p_source" not in config_kwargs:
        from algorithms.agents.es_agent import compute_source_power
        config_kwargs["p_source"] = compute_source_power(_build_pm_env())
        print(f"[driver] source_norm: auto-calibrated p_source="
              f"{config_kwargs['p_source']:.4g} (free-space total)", flush=True)

    # --- Phase 2 config -----------------------------------------
    cfg = Phase2Config(**config_kwargs)
    K = cfg.K
    half_K = K // 2

    # Snapshot starting params; we perturb around `flat_params` each iter.
    flat_params = nu.parameters_to_vector(
        policy.pi.parameters()).detach().clone().cpu()
    d = flat_params.numel()
    print(f"[driver] policy has {d:,} params  K={K}  σ={cfg.sigma}  "
          f"α_2={cfg.alpha_2}  T={cfg.T}  N_iter={cfg.N_iter}", flush=True)

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    buffer = ReplayBuffer()
    history = []
    overall_t0 = time.time()

    # Early-stop & checkpoint config (passed through payload from spawn).
    patience = int(payload.get("early_stop_patience", 0))   # 0 = disabled
    min_delta = float(payload.get("early_stop_min_delta", 0.0))
    checkpoint_every = int(payload.get("checkpoint_every", 10))
    print(f"[driver] early_stop: patience={patience}  min_delta={min_delta}  "
          f"checkpoint_every={checkpoint_every}", flush=True)

    best_ever_fitness = float("-inf")
    last_improvement_iter = -1
    plateau_count = 0

    # Volume layout for in-flight saves (overwritten each checkpoint).
    out_dir = Path("/buffer") / launch_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(n, label="checkpoint"):
        """Save current policy + history (and online-trained M if active)
        to Volume so we don't lose progress if the driver dies mid-run."""
        policy_out = io.BytesIO()
        torch.save({
            "pi_state_dict": policy.pi.state_dict(),
            "config": policy.config,
            "state_shape": policy.state_shape,
        }, policy_out)
        (out_dir / "policy_phase2.pt").write_bytes(policy_out.getvalue())
        (out_dir / "history.json").write_text(json.dumps({
            "config": config_kwargs,
            "goal_indices": goal_indices,
            "converged_indices": converged_indices,
            "logged_reward_indices": logged_reward_indices,
            "history": history,
            "iter_at_save": n,
            "best_ever_fitness": best_ever_fitness,
            "label": label,
        }, indent=2))
        if M is not None:
            m_out = io.BytesIO()
            torch.save({
                "model_state_dict": M.state_dict(),
                "state_shape": M_ckpt["state_shape"],
                "n_receivers": M_ckpt["n_receivers"],
                "hidden_dim": M_ckpt["hidden_dim"],
                "n_hidden_layers": M_ckpt["n_hidden_layers"],
                "log_mean": M_ckpt["log_mean"],
                "log_std": M_ckpt["log_std"],
                "receiver_indices": M_ckpt.get("receiver_indices"),
                "buffer_size": len(M_buffer_states),
            }, m_out)
            (out_dir / "M_online.pt").write_bytes(m_out.getvalue())
        buffer_volume.commit()
        msg = (f"[driver] {label} @ iter {n}: policy_phase2.pt + history.json"
               + (" + M_online.pt" if M is not None else "")
               + " saved")
        print(msg, flush=True)

    # --- M-ghost rollout helper (driver-side, no FDFD) ----------
    def _m_ghost_final_P(perturbed_params: torch.Tensor, eps_0: np.ndarray,
                        goal: int) -> np.ndarray:
        """Install perturbed params into policy, roll out T steps locally
        (no FDFD), return M's predicted P[30] at the trajectory's final
        state. Restores original params before returning."""
        nu.vector_to_parameters(perturbed_params, policy.pi.parameters())
        try:
            eps = eps_0.astype(np.float32, copy=True)
            for _ in range(cfg.T):
                delta = policy.predict(eps, goal).astype(np.float32)
                eps = np.clip(eps + delta, -1.0, 1.0).astype(np.float32)
            with torch.no_grad():
                eps_t = torch.as_tensor(eps[None], dtype=torch.float32)
                normed = M(eps_t)
                P_log = normed * M_log_std + M_log_mean
                P = torch.expm1(P_log).clamp(min=0.0).numpy()[0]
            return P
        finally:
            nu.vector_to_parameters(flat_params, policy.pi.parameters())

    def _sample_task():
        """Sample (goal, ε_0). Independent per saved candidate (no mirroring
        when M filter is active — saved ξ's already encode 'good direction')."""
        goal_k = int(rng.choice(goal_indices))
        cands = [g for g in converged_indices
                 if g != goal_k and g in memory_bank]
        goal_prev = int(rng.choice(cands)) if cands else None
        use_random = (rng.random() < cfg.p_rand) or (goal_prev is None)
        if use_random:
            eps_0 = rng.uniform(-1.0, 1.0, size=state_shape).astype(np.float32)
        else:
            eps_0 = memory_bank[goal_prev].astype(np.float32)
        return goal_k, eps_0

    # --- ES outer loop ------------------------------------------
    for n in range(cfg.N_iter):

        if M is not None:
            # ====== M-filtered candidate sampling ======
            # Resample ξ until we have m_filter_K saved (signed) perturbations
            # whose M-ghost trajectory ends with argmax == goal. Each saved
            # candidate gets its own (goal, ε_0) — mirrored-pair structure is
            # lost, but every candidate is "M-approved good direction" so
            # FDFD compute is concentrated on plausibly-useful rollouts.
            saved_xi = []
            saved_goals = []
            saved_eps_0 = []
            n_attempts = 0
            n_both_good = 0
            n_one_good = 0
            n_neither = 0
            while (len(saved_xi) < m_filter_K
                   and n_attempts < m_filter_max_attempts):
                xi = torch.randn(d)
                goal_k, eps_0_k = _sample_task()

                P_plus = _m_ghost_final_P(flat_params + cfg.sigma * xi,
                                          eps_0_k, goal_k)
                P_minus = _m_ghost_final_P(flat_params - cfg.sigma * xi,
                                           eps_0_k, goal_k)
                plus_good = int(P_plus.argmax()) == goal_k
                minus_good = int(P_minus.argmax()) == goal_k

                if plus_good and minus_good:
                    n_both_good += 1
                    Q_plus = (P_plus[goal_k] ** 2) / (P_plus.sum() + 1e-9)
                    Q_minus = (P_minus[goal_k] ** 2) / (P_minus.sum() + 1e-9)
                    pick = +1 if Q_plus >= Q_minus else -1
                    saved_xi.append(pick * xi)
                    saved_goals.append(goal_k)
                    saved_eps_0.append(eps_0_k)
                elif plus_good:
                    n_one_good += 1
                    saved_xi.append(xi)
                    saved_goals.append(goal_k)
                    saved_eps_0.append(eps_0_k)
                elif minus_good:
                    n_one_good += 1
                    saved_xi.append(-xi)
                    saved_goals.append(goal_k)
                    saved_eps_0.append(eps_0_k)
                else:
                    n_neither += 1
                n_attempts += 1

            K_actual = len(saved_xi)
            print(f"[driver iter {n:>3}] M-filter: {n_attempts} attempts → "
                  f"{K_actual} saved  (both_good={n_both_good}, "
                  f"one_good={n_one_good}, neither={n_neither})", flush=True)

            if K_actual == 0:
                print(f"[driver iter {n:>3}] M filter rejected ALL attempts; "
                      f"falling back to a single unfiltered candidate.",
                      flush=True)
                xi = torch.randn(d)
                goal_k, eps_0_k = _sample_task()
                saved_xi = [xi]
                saved_goals = [goal_k]
                saved_eps_0 = [eps_0_k]
                K_actual = 1

            xi_pop = torch.stack(saved_xi)         # (K_actual, d)
            goals_k = np.array(saved_goals)
            eps_0_list = saved_eps_0
            K_this_iter = K_actual

        else:
            # ====== Original mirrored-pair sampling ======
            xi_half = torch.randn(half_K, d)
            xi_pop = torch.cat([xi_half, -xi_half], dim=0)  # (K, d)

            # Task per mirrored pair (the variance-reduction fix from earlier).
            goals_half = rng.choice(goal_indices, size=half_K, replace=True)
            goals_k = np.concatenate([goals_half, goals_half])  # (K,)
            eps_0_list_half = []
            for k in range(half_K):
                candidates = [g for g in converged_indices
                              if g != int(goals_half[k]) and g in memory_bank]
                goal_prev = int(rng.choice(candidates)) if candidates else None
                use_random = (rng.random() < cfg.p_rand) or (goal_prev is None)
                if use_random:
                    eps_0 = rng.uniform(-1.0, 1.0,
                                        size=state_shape).astype(np.float32)
                else:
                    eps_0 = memory_bank[goal_prev].astype(np.float32)
                eps_0_list_half.append(eps_0)
            eps_0_list = eps_0_list_half + eps_0_list_half
            K_this_iter = K

        # 3. Build rollout payloads (K_this_iter candidates with perturbed params).
        payloads = []
        for k in range(K_this_iter):
            perturbed = (flat_params + cfg.sigma * xi_pop[k]).numpy().astype(np.float32)
            payloads.append({
                "flat_params": perturbed,
                "policy_config": policy_config_for_workers,
                "state_shape": list(state_shape),
                "goal": int(goals_k[k]),
                "eps_0": eps_0_list[k],
                "logged_reward_indices": logged_reward_indices,
                "T": cfg.T,
                "eta": cfg.eta,
                "w_crosstalk": cfg.w_crosstalk,
                "w_loss": cfg.w_loss,
                "w_energy": cfg.w_energy,
                "reward_mode": cfg.reward_mode,
                "target_frac_scale": cfg.target_frac_scale,
                "gamma": cfg.gamma,
                "p_source": cfg.p_source,
                "hold_threshold": cfg.hold_threshold,
                "hold_bonus": cfg.hold_bonus,
            })

        # 4. Dispatch K rollouts in parallel. .map() returns results in order.
        t_iter = time.time()
        results = list(rollout_one.map(payloads))
        iter_wall = time.time() - t_iter

        # 5. Collect fitnesses, lengths, transitions.
        fitnesses = np.array([r["fitness"] for r in results], dtype=np.float64)
        rollout_lengths = np.array([r["rollout_length"] for r in results])
        for r in results:
            transitions_k = pickle.loads(r["transitions_pkl"])
            buffer.extend(transitions_k)

        # 5b. Online M training: every FDFD rollout step is fresh labeled
        # data for M. Append to M's training buffer, then run a few SGD
        # passes. Closes the OOD gap between Phase 1 mean-trajectory states
        # (M's seed data) and the Phase 2 rollout states it will be asked
        # to filter next iter.
        if M is not None:
            new_pairs = 0
            for r in results:
                pairs = pickle.loads(r["per_step_data_pkl"])
                for eps_next, P_next in pairs:
                    M_buffer_states.append(np.asarray(eps_next, dtype=np.float32))
                    M_buffer_P.append(np.asarray(P_next, dtype=np.float64))
                    new_pairs += 1

            if len(M_buffer_states) >= M_train_batch_size:
                M.train()
                m_train_losses = []
                buf_n = len(M_buffer_states)
                for epoch in range(M_train_epochs_per_iter):
                    idx = rng.choice(buf_n, size=M_train_batch_size, replace=False)
                    batch_eps = np.stack([M_buffer_states[i] for i in idx]).astype(np.float32)
                    batch_P = np.stack([M_buffer_P[i] for i in idx]).astype(np.float32)
                    batch_log_P = np.log1p(batch_P)
                    log_mean_np = M_log_mean.numpy()
                    log_std_np = M_log_std.numpy()
                    batch_target = ((batch_log_P - log_mean_np) / log_std_np
                                    ).astype(np.float32)

                    batch_eps_t = torch.as_tensor(batch_eps, dtype=torch.float32)
                    batch_target_t = torch.as_tensor(batch_target, dtype=torch.float32)
                    pred = M(batch_eps_t)
                    loss = F.mse_loss(pred, batch_target_t)
                    M_optimizer.zero_grad()
                    loss.backward()
                    M_optimizer.step()
                    m_train_losses.append(float(loss))
                M.eval()
                print(f"[driver iter {n:>3}] M trained: +{new_pairs} pairs "
                      f"(buffer={len(M_buffer_states):,})  "
                      f"loss {m_train_losses[0]:.4e}→{m_train_losses[-1]:.4e}",
                      flush=True)

        # 6. Centered-rank ES update.
        u = torch.as_tensor(centered_ranks(fitnesses), dtype=torch.float32)
        grad = (cfg.alpha_2 / (K_this_iter * cfg.sigma)) * (xi_pop.T @ u)
        flat_params = flat_params + grad

        # Sync into the policy network (for the next iter's noise base).
        nu.vector_to_parameters(flat_params, policy.pi.parameters())

        # 7. Log.
        entry = {
            "iteration": n,
            "fitness_mean": float(fitnesses.mean()),
            "fitness_best": float(fitnesses.max()),
            "fitness_std": float(fitnesses.std()),
            "mean_rollout_length": float(rollout_lengths.mean()),
            "iter_wall_seconds": iter_wall,
            "buffer_size": len(buffer),
        }
        history.append(entry)
        if wandb_run is not None:
            import wandb
            wandb.log({
                "iteration": entry["iteration"],
                "fitness/mean": entry["fitness_mean"],
                "fitness/best": entry["fitness_best"],
                "fitness/std": entry["fitness_std"],
                "mean_rollout_length": entry["mean_rollout_length"],
                "iter_wall_minutes": entry["iter_wall_seconds"] / 60,
                "buffer_size": entry["buffer_size"],
            }, step=entry["iteration"])
        print(f"[driver iter {n:>3}/{cfg.N_iter}]  "
              f"fitness mean={entry['fitness_mean']:+.3e}  "
              f"best={entry['fitness_best']:+.3e}  "
              f"rollout_len={entry['mean_rollout_length']:.1f}  "
              f"wall={iter_wall/60:.1f} min  buffer={len(buffer):,}",
              flush=True)

        # --- Early-stop tracking ----------------------------------
        # Track running max of fitness_MEAN (not max). The mean reflects the
        # policy's average behavior across the K candidates; tracking the
        # max-single-candidate is brittle when an early lucky outlier
        # locks out future progress (the failed run plateaued for 25 iters
        # on a single iter-10 outlier while the mean was still improving).
        iter_mean = float(fitnesses.mean())
        if iter_mean > best_ever_fitness + min_delta:
            best_ever_fitness = iter_mean
            last_improvement_iter = n
            plateau_count = 0
        else:
            plateau_count += 1

        # --- Periodic checkpoint to Volume ------------------------
        if checkpoint_every > 0 and (n + 1) % checkpoint_every == 0:
            _save_checkpoint(n, label=f"checkpoint_iter_{n:04d}")

        # --- Early stop on plateau --------------------------------
        if patience > 0 and plateau_count >= patience:
            print(f"[driver] EARLY STOP @ iter {n}: best mean fitness "
                  f"({best_ever_fitness:+.3e}) hasn't improved by ≥{min_delta} "
                  f"for {plateau_count} iters (last improvement @ iter "
                  f"{last_improvement_iter}).", flush=True)
            if wandb_run is not None:
                import wandb
                wandb.log({"early_stop_iter": n,
                          "early_stop_best_mean_fitness": best_ever_fitness},
                          step=n)
            _save_checkpoint(n, label=f"early_stop_iter_{n:04d}")
            break

    total_elapsed = time.time() - overall_t0
    print(f"[driver] done in {total_elapsed/60:.1f} min "
          f"({cfg.N_iter} iters)", flush=True)

    # --- Persist to Volume ---------------------------------------
    out_dir = Path("/buffer") / launch_id
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_out = io.BytesIO()
    torch.save({
        "pi_state_dict": policy.pi.state_dict(),
        "config": policy.config,
        "state_shape": policy.state_shape,
    }, policy_out)
    (out_dir / "policy_phase2.pt").write_bytes(policy_out.getvalue())

    (out_dir / "history.json").write_text(json.dumps({
        "config": config_kwargs,
        "goal_indices": goal_indices,
        "converged_indices": converged_indices,
        "logged_reward_indices": logged_reward_indices,
        "history": history,
        "elapsed_seconds": total_elapsed,
        "n_transitions_end": len(buffer),
        "architecture": "K-parallel via rollout_one.map()",
    }, indent=2))

    with open(out_dir / "buffer_appended.pkl", "wb") as fh:
        pickle.dump(list(buffer.transitions), fh)

    (out_dir / "meta.json").write_text(json.dumps({
        "launch_id": launch_id,
        "goal_indices": goal_indices,
        "converged_indices": converged_indices,
        "n_history_entries": len(history),
        "n_transitions_end": len(buffer),
        "elapsed_seconds": total_elapsed,
    }, indent=2))
    buffer_volume.commit()

    if wandb_run is not None:
        import wandb
        wandb.log({
            "final/elapsed_minutes": total_elapsed / 60,
            "final/n_transitions_end": len(buffer),
            "final/n_history_entries": len(history),
        })
        wandb_run.finish()

    return {
        "launch_id": launch_id,
        "n_iters_completed": len(history),
        "elapsed_seconds": total_elapsed,
        "n_transitions_end": len(buffer),
    }


# =============================================================================
# Collect entrypoint
# =============================================================================

@app.local_entrypoint()
def collect(
    launch_id: str = None,
    out_dir: str = "phase2_parallel_output",
):
    """Pull Phase 2 parallel-run outputs from the Volume."""
    if not launch_id:
        raise ValueError("--launch-id required")
    out = Path(out_dir) / launch_id
    out.mkdir(parents=True, exist_ok=True)

    files = ["policy_phase2.pt", "history.json", "meta.json",
             "buffer_appended.pkl"]
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
