# =============================================================================
# ES STATE-SPACE POLICY: Per-Goal ε-ES with M-Surrogate Pre-Filter
# =============================================================================
"""
State-space Evolution Strategies on the permittivity ε directly (Phase-1
architecture), with an M(ε) → P[30] surrogate used to pre-rank a large
candidate pool before paying the FDFD cost on the top survivors. Every FDFD
call also feeds the (ε, P) pair back into M's training buffer for DAGGER-
style online refinement.

This replaces the closed-loop ESPolicy (algorithms/policies/es_policy.py)
for Phase 2 deployment. Empirically, the static θ → ε mapping is the right
shape for this design problem — closed-loop rollouts added trajectory
variance with no benefit, and parameter-space ES on π_φ never beat plain
linear interpolation. State-space ES per-goal, with M as a candidate
pre-filter, is the descendant of:

    Phase 1 ES (es_agent.py)          — state-space ES, all K candidates FDFD'd
    deploy_phase2_fdfd_inner.py       — same per-goal at deploy time, K=20 FDFD/iter
    deploy_phase2_online.py           — M as the *only* fitness scorer (biased)

Here M acts as a *filter*, not the scorer: K_cand mass-evaluated by M (free),
K_real << K_cand survivors evaluated by FDFD (ground truth), ES update on ε
uses only the FDFD fitnesses. M's role is to spend FDFD budget on candidates
likely to be good, not to replace ground truth. As M improves online, the
filter gets sharper and FDFD compute concentrates on better and better
directions.

Reward defaults to Phase 1's full absolute formula
    r = P_θ − w_c·λ_c·P_others − w_loss·ΔP_loss − w_energy·ΔE_rods
with (w_c, w_loss, w_energy) = (0.3, 1e-3, 0.1) matching the original
Phase 1 run. The `reward_mode="retarget"` alternative (Q = P_θ² / P_total)
is also supported per get_reward() in algorithms/agents/es_agent.py.

Per-iter dataflow (one goal):
    eps_curr ─┬─► xi_pop ~ N(0, I), K_cand mirrored pairs
              │     │
              │     └─► eps_pop = clip(eps_curr + σ·xi_pop, [-1, 1])
              │             │
              │             ▼
              │   M.predict(eps_pop) → P_pred ∈ R^{K_cand × 30}
              │             │
              │   top K_real by Q_pred = P_pred[:, goal]² / P_pred.sum(axis=1)
              │             │
              │             ▼
              │   fdfd_fn(eps_pop[top_idx]) → P_true ∈ R^{K_real × 30}
              │             │           (parallel on Modal — caller supplies)
              │             ▼
              │   fitnesses_real = get_reward(P_true, ...)
              │             │
              │             ▼
              │   M ← SGD on (eps_pop[top_idx], P_true) (online DAGGER)
              │             │
              │             ▼
              └─► eps_curr ← clip(eps_curr + α/(K_real·σ)·Σ u_k·xi_top[k], [-1, 1])

The class is callback-driven so it doesn't depend on Modal: the caller
supplies `fdfd_batch_fn(eps_batch) → (P_batch, P_loss_batch)` and the policy
calls it once per outer iteration. The Modal driver in
train_phase2_state_space_modal.py wraps that callback in a
`fdfd_one.map([...])` call so K_real FDFDs run concurrently.

Checkpointing: snapshot via `state_dict()` / `load_state_dict()` mirrors the
Phase 1 ESCheckpoint shape — the runner is responsible for writing it to
persistent storage (Modal Volume).
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]   # cs153 root → geometry, simulation
_DBS_ROOT = Path(__file__).resolve().parents[2]       # dynamic_beam_steering/ → algorithms.*
for _p in (_PROJECT_ROOT, _DBS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from algorithms.agents.es_agent import (  # noqa: E402
    centered_ranks, compute_E_rods, get_reward,
)
from algorithms.infrastructure.utils import Transition  # noqa: E402


# =============================================================================
# Config
# =============================================================================

@dataclass
class ESStateSpaceConfig:
    """Per-goal state-space ES with M pre-filter."""

    # --- ES outer loop ----------------------------------------------------
    # K_cand: number of M-evaluated candidates per iter (free — just a forward
    #         pass). Larger = wider exploration, no FDFD cost.
    # K_real: number of those candidates that get FDFD'd (the actual cost
    #         driver). Must be even for the mirrored-pair logic — survivors
    #         aren't necessarily paired (filter breaks symmetry), but we keep
    #         even for parity with Phase 1 and to make K_real // 2 = half_K.
    K_cand: int = 200
    K_real: int = 20
    sigma: float = 0.05            # σ in ε-space; small — refinement, not search
    alpha: float = 0.05            # state-space LR (Phase 1 used α_1 = 0.02)
    N_iter: int = 50               # outer iters per goal
    eta: float = 1e-2              # early-term when best target_frac ≥ 1 − η
    seed: int = 0
    log_every: int = 1

    # --- M-filter ---------------------------------------------------------
    # filter_mode: how to score & pick the top K_real:
    #   "q"        — sort by Q = P_pred[goal]² / (P_pred.sum() + EPS), descending.
    #                Matches the retarget reward; recommended default.
    #   "argmax"   — keep only candidates whose argmax(P_pred) == goal, then
    #                top-K_real by Q. Stricter filter — passes fewer candidates
    #                when M is uncertain. If fewer than K_real pass, falls
    #                back to top-K_real by Q overall.
    #   "off"      — no filter; K_real == K_cand FDFDs per iter (sanity baseline).
    filter_mode: str = "q"

    # Online M training (DAGGER on the FDFD-derived (ε, P) pairs).
    # Disabled by default (online_m=False). The driver typically owns the
    # M buffer/optimizer and trains M centrally between iters; setting
    # online_m=True makes the policy take ownership instead.
    online_m: bool = False
    m_train_epochs_per_iter: int = 5
    m_train_batch_size: int = 256
    m_train_lr: float = 1e-3
    m_train_weight_decay: float = 1e-5
    m_warmup_buffer: int = 256     # don't train M until buffer ≥ this size

    # --- Reward shaping (matches Phase 1 defaults exactly) ----------------
    # absolute mode — Phase 1's full reward formula. Used here because the
    # state-space-ES inner loop has no telescoping (each candidate is scored
    # in isolation, not as a step), so the absolute formulation is what
    # actually corresponds to "what Phase 1 was optimizing per FDFD call".
    reward_mode: str = "absolute"
    w_crosstalk: float = 0.3
    w_loss: float = 1e-3
    w_energy: float = 0.1
    target_frac_scale: float = 1.0e+5  # only when reward_mode == "target_frac"

    def __post_init__(self):
        if self.K_real <= 0 or self.K_cand < self.K_real:
            raise ValueError(
                f"K_real ({self.K_real}) must be ≤ K_cand ({self.K_cand}) "
                f"and > 0."
            )
        if self.K_cand % 2 != 0:
            raise ValueError(
                f"K_cand must be even for mirrored sampling; got {self.K_cand}."
            )
        if self.filter_mode not in ("q", "argmax", "off"):
            raise ValueError(
                f"filter_mode={self.filter_mode!r}; expected 'q' | 'argmax' | 'off'."
            )
        if self.reward_mode not in ("absolute", "retarget", "target_frac"):
            raise ValueError(
                f"reward_mode={self.reward_mode!r}; expected "
                f"'absolute' | 'retarget' | 'target_frac'."
            )


@dataclass
class ESStateSpaceResult:
    """Per-goal output of run_one_goal."""
    goal: int                                     # 0-indexed receiver
    eps_star: np.ndarray                          # best ε by FDFD-true Q
    best_target_frac: float                       # best P_target / P_total seen
    best_Q: float                                 # best Q = P_target² / P_total
    iterations: int                               # iters actually run
    converged: bool                               # early-term hit?
    history: List[dict] = field(default_factory=list)
    # All (ε, P) pairs FDFD'd over the course of the run — used to grow
    # M's training set offline (the caller writes these to disk).
    fdfd_eps: Optional[np.ndarray] = None         # (N_total_fdfd, N_x, N_y)
    fdfd_P: Optional[np.ndarray] = None           # (N_total_fdfd, n_receivers)
    eps_initial: Optional[np.ndarray] = None      # warm-start ε for viz
    # Per-iter eps_curr sequence — the closed-loop trajectory. Shape
    # (iterations + 1, N_x, N_y), with eps_traj[0] == eps_initial and
    # eps_traj[t+1] = clip(eps_traj[t] + α·grad_t, [-1, 1]). Each consecutive
    # pair is one (state, next_state) supervision example for the closed-loop
    # distillation in train_phase2_distill_closed_loop.py:
    #     δ_t := eps_traj[t+1] − eps_traj[t]  is the target action for π_φ(ε_t, θ).
    eps_traj: Optional[np.ndarray] = None         # (iterations + 1, N_x, N_y)


# =============================================================================
# M-surrogate wrapper (M is a duck-typed object exposing `predict_P`)
# =============================================================================

class MSurrogate:
    """Thin wrapper around the trained M(ε) → P[30] network.

    Loads the train_V_network.py checkpoint format. Exposes:
      - predict_P(eps_batch) -> P_pred  (numpy in, numpy out)
      - train_step(eps_batch, P_batch) -> loss  (one SGD step)
      - state_dict() / load_state_dict()  for checkpoint passthrough
    """

    def __init__(self, ckpt_path, device: str = "cpu",
                 lr: float = 1e-3, weight_decay: float = 1e-5):
        import torch
        # Imported lazily so the policy can be imported in pure-numpy contexts.
        from train_V_network import PNetwork

        self._torch = torch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.state_shape = tuple(ckpt["state_shape"])
        self.n_receivers = int(ckpt["n_receivers"])
        self.hidden_dim = int(ckpt["hidden_dim"])
        self.n_hidden_layers = int(ckpt["n_hidden_layers"])
        self.log_mean = np.asarray(ckpt["log_mean"], dtype=np.float32)
        self.log_std = np.asarray(ckpt["log_std"], dtype=np.float32)
        self.receiver_indices = ckpt.get("receiver_indices")

        self.model = PNetwork(self.state_shape, self.n_receivers,
                              self.hidden_dim, self.n_hidden_layers).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.device = device
        self._log_mean_t = torch.as_tensor(self.log_mean, device=device)
        self._log_std_t = torch.as_tensor(self.log_std, device=device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def predict_P(self, eps_batch: np.ndarray) -> np.ndarray:
        """ε ∈ R^{B × N_x × N_y} → P ∈ R^{B × n_receivers} (de-normalized)."""
        torch = self._torch
        with torch.no_grad():
            eps_t = torch.as_tensor(eps_batch, dtype=torch.float32,
                                    device=self.device)
            normed = self.model(eps_t)
            log_P = normed * self._log_std_t + self._log_mean_t
            return torch.expm1(log_P).clamp(min=0.0).cpu().numpy()

    def train_step(self, eps_batch: np.ndarray, P_batch: np.ndarray) -> float:
        """One Adam step on a minibatch. Returns the MSE loss (normalized space)."""
        torch = self._torch
        import torch.nn.functional as F
        self.model.train()
        try:
            log_P = np.log1p(P_batch).astype(np.float32)
            target = (log_P - self.log_mean) / self.log_std
            eps_t = torch.as_tensor(eps_batch, dtype=torch.float32,
                                    device=self.device)
            target_t = torch.as_tensor(target, dtype=torch.float32,
                                       device=self.device)
            pred = self.model(eps_t)
            loss = F.mse_loss(pred, target_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        finally:
            self.model.eval()

    def state_dict(self) -> dict:
        """Round-trippable snapshot — for in-flight checkpointing."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_shape": self.state_shape,
            "n_receivers": self.n_receivers,
            "hidden_dim": self.hidden_dim,
            "n_hidden_layers": self.n_hidden_layers,
            "log_mean": self.log_mean,
            "log_std": self.log_std,
            "receiver_indices": self.receiver_indices,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd["model_state_dict"])
        if "optimizer_state_dict" in sd:
            self.optimizer.load_state_dict(sd["optimizer_state_dict"])


# =============================================================================
# ESStateSpacePolicy — per-goal ES on ε with M filter
# =============================================================================

# Callback supplied by the runner. Takes a batched ε tensor and returns
#   (P_batch, P_loss_batch)  -- both numpy arrays of shape (B, n_receivers)
# and (B,) respectively. The Modal pipeline implements this via
# fdfd_one.map([...]) so the B candidates evaluate in parallel.
FDFDBatchFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


@dataclass
class ESStateSpaceCheckpoint:
    """Snapshot of run_one_goal state for resume."""
    eps_curr: np.ndarray
    eps_initial: np.ndarray
    best_eps: np.ndarray
    best_Q: float
    best_target_frac: float
    history: List[dict]
    next_iteration: int
    rng_state: Any
    fdfd_eps: List[np.ndarray]
    fdfd_P: List[np.ndarray]
    # eps_traj is the per-iter eps_curr sequence, length = next_iteration + 1
    # (includes the initial state at index 0). Used by closed-loop distillation
    # to extract (ε_t, ε_{t+1}, goal) supervision tuples.
    eps_traj: List[np.ndarray] = field(default_factory=list)
    m_state_dict: Optional[dict] = None    # when policy.online_m=True


class ESStateSpacePolicy:
    """Per-goal state-space ES with M-surrogate pre-filter.

    Stateless across goals — call `run_one_goal` once per (goal, eps_init);
    pass the same instance to share M across goals (recommended). The class
    holds the M reference and the reward config; per-call state (ε, best,
    history, RNG) is local to run_one_goal so concurrent calls don't alias.

    M is OPTIONAL: pass `m_surrogate=None` to fall back to no-filter ES
    (every candidate gets FDFD'd — equivalent to deploy_phase2_fdfd_inner).
    """

    def __init__(
        self,
        state_shape: Tuple[int, int],
        m_surrogate: Optional[MSurrogate] = None,
        config: Optional[ESStateSpaceConfig] = None,
    ):
        self.state_shape = tuple(state_shape)
        self.config = config if config is not None else ESStateSpaceConfig()
        self.M = m_surrogate
        if self.config.filter_mode != "off" and self.M is None:
            raise ValueError(
                f"filter_mode={self.config.filter_mode!r} requires an "
                f"m_surrogate; pass filter_mode='off' to skip filtering."
            )

        # Online-M training buffer (only used when cfg.online_m=True; otherwise
        # the runner manages the buffer externally and the policy just returns
        # the per-iter FDFD pairs).
        self._m_buffer_eps: List[np.ndarray] = []
        self._m_buffer_P: List[np.ndarray] = []

    # ----- M filter -------------------------------------------------------
    def _filter_candidates(
        self,
        eps_pop: np.ndarray,
        goal: int,
    ) -> np.ndarray:
        """Return indices of the K_real candidates to FDFD this iter.

        Pure numpy — `eps_pop` shape (K_cand, N_x, N_y), returns (K_real,)
        int indices into eps_pop.
        """
        cfg = self.config
        K_cand = eps_pop.shape[0]

        if cfg.filter_mode == "off" or self.M is None:
            return np.arange(min(cfg.K_real, K_cand))

        P_pred = self.M.predict_P(eps_pop)        # (K_cand, n_receivers)
        EPS = 1e-9
        P_target = P_pred[:, goal]
        Q_pred = (P_target ** 2) / (P_pred.sum(axis=1) + EPS)

        if cfg.filter_mode == "argmax":
            # Only keep candidates M says peak AT the goal; tiebreak by Q.
            argmax_match = (P_pred.argmax(axis=1) == goal)
            n_match = int(argmax_match.sum())
            if n_match >= cfg.K_real:
                match_idx = np.where(argmax_match)[0]
                top_within = np.argsort(-Q_pred[match_idx])[:cfg.K_real]
                return match_idx[top_within]
            # Not enough peakers — fall back to global top-K by Q.

        # "q" mode (default) or argmax-fallback.
        return np.argsort(-Q_pred)[:cfg.K_real]

    # ----- Per-iter fitness from FDFD-true P -----------------------------
    def _score_fitnesses(
        self,
        P_true: np.ndarray,        # (K_real, n_receivers)
        P_loss_true: np.ndarray,   # (K_real,)
        eps_pop_top: np.ndarray,   # (K_real, N_x, N_y) — for E_rods
        P_loss_baseline: float,
        E_rods_baseline: float,
        goal: int,
        training_indices: List[int],
    ) -> np.ndarray:
        """Compute per-candidate fitness using the chosen reward mode."""
        cfg = self.config
        K_real = P_true.shape[0]
        fitnesses = np.zeros(K_real, dtype=np.float64)
        for k in range(K_real):
            E_rods_k = compute_E_rods(eps_pop_top[k])
            d_P_loss = float(P_loss_true[k]) - P_loss_baseline
            d_E_rods = E_rods_k - E_rods_baseline

            # In retarget mode, lambda_loss/energy carry the ABSOLUTE per-state
            # values (P_loss(s), E_rods(s)); in absolute mode they carry the
            # ΔP_loss / ΔE_rods step deltas. get_reward applies the right
            # interpretation based on `mode`.
            if cfg.reward_mode == "retarget":
                ll, le = float(P_loss_true[k]), E_rods_k
            else:
                ll, le = d_P_loss, d_E_rods

            fitnesses[k] = get_reward(
                P_true[k], goal, training_indices,
                lambda_loss=ll, lambda_energy=le,
                w_crosstalk=cfg.w_crosstalk,
                w_loss=cfg.w_loss, w_energy=cfg.w_energy,
                mode=cfg.reward_mode,
                target_frac_scale=cfg.target_frac_scale,
            )
        return fitnesses

    # ----- Online M training ---------------------------------------------
    def _train_M_step(self, rng: np.random.Generator) -> List[float]:
        """One SGD pass of cfg.m_train_epochs_per_iter steps on the M buffer."""
        cfg = self.config
        if (self.M is None or not cfg.online_m
                or len(self._m_buffer_eps) < cfg.m_warmup_buffer):
            return []

        losses = []
        buf_n = len(self._m_buffer_eps)
        bs = min(cfg.m_train_batch_size, buf_n)
        for _ in range(cfg.m_train_epochs_per_iter):
            idx = rng.choice(buf_n, size=bs, replace=False)
            eps_batch = np.stack([self._m_buffer_eps[i] for i in idx]
                                 ).astype(np.float32)
            P_batch = np.stack([self._m_buffer_P[i] for i in idx]
                               ).astype(np.float32)
            losses.append(self.M.train_step(eps_batch, P_batch))
        return losses

    # ----- One outer ES loop for one goal --------------------------------
    def run_one_goal(
        self,
        eps_init: np.ndarray,
        goal: int,
        training_indices: List[int],
        fdfd_batch_fn: FDFDBatchFn,
        baseline_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, float]]] = None,
        buffer=None,                           # optional ReplayBuffer for multi-goal logging
        on_iteration: Optional[Callable[[dict], None]] = None,
        on_checkpoint: Optional[Callable[["ESStateSpaceCheckpoint"], None]] = None,
        checkpoint_every: int = 0,
        resume_state: Optional["ESStateSpaceCheckpoint"] = None,
    ) -> ESStateSpaceResult:
        """Run state-space ES + M-filter for ONE (goal, eps_init).

        Args:
            eps_init: starting ε, shape (N_x, N_y). Usually the linear
                interpolation between the two nearest Phase 1 anchors, or
                the nearest anchor verbatim. Memory-bank lookup is the
                caller's job (the policy doesn't know about Phase 1
                outputs).
            goal: 0-indexed target receiver (e.g. 2 for the third receiver).
            training_indices: union of receiver indices used in the reward's
                "P_others" sum and in multi-goal logging (typically all 30).
            fdfd_batch_fn: callback that runs FDFD on a batch of ε's.
                Signature: eps_batch (B, N_x, N_y) → (P (B, n_recv),
                                                      P_loss (B,))
                The Modal pipeline implements this as a fan-out via
                fdfd_one.map(); local testing uses a serial loop.
            baseline_fn: optional callback for the baseline FDFD at eps_curr
                each iter. If omitted, the policy adds eps_curr as a
                (K_real+1)-th call to fdfd_batch_fn. Provide separately if
                you want different batching for the baseline.
            buffer: optional ReplayBuffer; if provided, each FDFD candidate
                is logged as len(training_indices) transitions per the Phase 1
                multi-goal scheme.
            on_iteration: optional callback called after each iter with the
                logged history entry. Used by the runner for wandb streaming.
            on_checkpoint: optional callback called every `checkpoint_every`
                iters with an ESStateSpaceCheckpoint. Runner persists it.
            checkpoint_every: 0 = disabled.
            resume_state: if provided, skip init and pick up from there.

        Returns:
            ESStateSpaceResult with the converged ε, history, and the full
            (ε, P) cohort collected over the run (for M retraining offline).
        """
        cfg = self.config
        N_x, N_y = self.state_shape
        K_cand, K_real = cfg.K_cand, cfg.K_real
        half_K_cand = K_cand // 2

        # --- Init or resume ---------------------------------------------
        if resume_state is not None:
            eps_curr = resume_state.eps_curr.astype(np.float32, copy=True)
            eps_initial = resume_state.eps_initial.astype(np.float32, copy=True)
            best_eps = resume_state.best_eps.astype(np.float32, copy=True)
            best_Q = float(resume_state.best_Q)
            best_target_frac = float(resume_state.best_target_frac)
            history = list(resume_state.history)
            start_iter = int(resume_state.next_iteration)
            fdfd_eps_acc = list(resume_state.fdfd_eps)
            fdfd_P_acc = list(resume_state.fdfd_P)
            # Old checkpoints (pre-distill) won't have eps_traj — fall back
            # to a single-element list seeded at eps_curr; distillation will
            # just miss the early steps from that goal.
            eps_traj_acc = list(getattr(resume_state, "eps_traj", []) or [])
            if not eps_traj_acc:
                eps_traj_acc = [eps_curr.copy()]
            rng = np.random.default_rng(cfg.seed)
            rng.bit_generator.state = resume_state.rng_state
            if (resume_state.m_state_dict is not None
                    and self.M is not None and cfg.online_m):
                self.M.load_state_dict(resume_state.m_state_dict)
        else:
            eps_curr = np.asarray(eps_init, dtype=np.float32).copy()
            eps_initial = eps_curr.copy()
            best_eps = eps_curr.copy()
            best_Q = -np.inf
            best_target_frac = 0.0
            history = []
            start_iter = 0
            fdfd_eps_acc = []
            fdfd_P_acc = []
            eps_traj_acc = [eps_curr.copy()]    # closed-loop trajectory, step 0
            rng = np.random.default_rng(cfg.seed + int(goal))

        for n in range(start_iter, cfg.N_iter):

            # 0. Baseline FDFD at eps_curr — needed for ΔP_loss / ΔE_rods.
            if baseline_fn is not None:
                P_baseline_arr, P_loss_baseline = baseline_fn(eps_curr)
            else:
                P_b, P_l_b = fdfd_batch_fn(eps_curr[None])  # (1, n_recv), (1,)
                P_baseline_arr = P_b[0]
                P_loss_baseline = float(P_l_b[0])
            E_rods_baseline = compute_E_rods(eps_curr)
            fdfd_eps_acc.append(eps_curr.copy())
            fdfd_P_acc.append(P_baseline_arr.copy())

            # 1. Mirrored Gaussian noise in ε-space (Phase 1 default).
            xi_half = rng.standard_normal((half_K_cand, N_x, N_y)
                                          ).astype(np.float32)
            xi_pop = np.concatenate([xi_half, -xi_half], axis=0)   # (K_cand,)
            eps_pop = np.clip(eps_curr[None] + cfg.sigma * xi_pop,
                              -1.0, 1.0).astype(np.float32)

            # 2. M-filter: choose K_real survivors to FDFD this iter.
            top_idx = self._filter_candidates(eps_pop, goal)
            eps_top = eps_pop[top_idx]
            xi_top = xi_pop[top_idx]

            # 3. FDFD on the survivors (parallel on Modal via the callback).
            P_true, P_loss_true = fdfd_batch_fn(eps_top)  # (K_real, n), (K_real,)
            fdfd_eps_acc.extend(list(eps_top))
            fdfd_P_acc.extend(list(P_true))

            # 4. Fitness from FDFD-true P (NOT from M — M is filter-only).
            fitnesses = self._score_fitnesses(
                P_true, P_loss_true, eps_top,
                P_loss_baseline, E_rods_baseline,
                goal, training_indices,
            )

            # 5. Centered-rank ES gradient on the survivor cohort.
            # Note: with the filter active the survivors aren't necessarily
            # mirrored pairs — centered-rank shaping handles unpaired
            # samples fine (it's invariant to monotone reward transforms),
            # at the cost of slightly higher gradient variance than a
            # fully-mirrored cohort. Worth it for the FDFD savings.
            denom = max(len(xi_top), 1)
            u = centered_ranks(fitnesses)
            grad = np.einsum('k,kij->ij', u, xi_top) / (denom * cfg.sigma)
            eps_curr = np.clip(eps_curr + cfg.alpha * grad, -1.0, 1.0
                               ).astype(np.float32)
            # Snapshot the post-update state as the (t+1)-th trajectory point.
            # Each consecutive (eps_traj[t], eps_traj[t+1]) is one closed-loop
            # supervision example for π_φ(ε_t, goal) → δ_t.
            eps_traj_acc.append(eps_curr.copy())

            # 6. Track best by FDFD-true Q (concentration); independent of
            #    the chosen reward_mode, this is the deployment-time metric.
            EPS = 1e-9
            Q_true_pop = ((P_true[:, goal] ** 2)
                          / (P_true.sum(axis=1) + EPS))
            tf_pop = P_true[:, goal] / np.maximum(P_true.sum(axis=1), EPS)
            k_best = int(Q_true_pop.argmax())
            if float(Q_true_pop[k_best]) > best_Q:
                best_Q = float(Q_true_pop[k_best])
                best_eps = eps_top[k_best].copy()
            if float(tf_pop.max()) > best_target_frac:
                best_target_frac = float(tf_pop.max())

            # 7. Multi-goal transition logging — same Phase 1 scheme: one
            #    transition per training angle per FDFD'd candidate.
            if buffer is not None:
                for k in range(len(top_idx)):
                    E_rods_k = compute_E_rods(eps_top[k])
                    d_P_loss = float(P_loss_true[k]) - P_loss_baseline
                    d_E_rods = E_rods_k - E_rods_baseline
                    for goal_j in training_indices:
                        if cfg.reward_mode == "retarget":
                            ll, le = float(P_loss_true[k]), E_rods_k
                        else:
                            ll, le = d_P_loss, d_E_rods
                        r_j = get_reward(
                            P_true[k], int(goal_j), training_indices,
                            lambda_loss=ll, lambda_energy=le,
                            w_crosstalk=cfg.w_crosstalk,
                            w_loss=cfg.w_loss, w_energy=cfg.w_energy,
                            mode=cfg.reward_mode,
                            target_frac_scale=cfg.target_frac_scale,
                        )
                        buffer.append(Transition(
                            state=eps_curr.copy(),
                            action=(eps_top[k] - eps_curr).astype(np.float32),
                            reward=float(r_j),
                            next_state=eps_top[k].copy(),
                            goal=int(goal_j),
                        ))

            # 8. Online M training (if enabled) — DAGGER on fresh FDFD pairs.
            for k in range(len(top_idx)):
                self._m_buffer_eps.append(eps_top[k].copy())
                self._m_buffer_P.append(P_true[k].copy())
            m_train_losses = self._train_M_step(rng)

            # 9. Log + callbacks.
            if n % cfg.log_every == 0:
                entry = {
                    "iteration": n,
                    "goal": int(goal),
                    "fitness_mean": float(fitnesses.mean()),
                    "fitness_best": float(fitnesses.max()),
                    "Q_true_mean": float(Q_true_pop.mean()),
                    "Q_true_best": float(Q_true_pop.max()),
                    "target_frac_best_iter": float(tf_pop.max()),
                    "best_ever_Q": float(best_Q),
                    "best_ever_target_frac": float(best_target_frac),
                    "n_fdfd_total": len(fdfd_eps_acc),
                    "m_train_loss_first": (float(m_train_losses[0])
                                           if m_train_losses else None),
                    "m_train_loss_last": (float(m_train_losses[-1])
                                          if m_train_losses else None),
                }
                history.append(entry)
                if on_iteration is not None:
                    on_iteration(entry)

            # 10. Checkpoint (after state is fully updated for the iter).
            if (on_checkpoint is not None and checkpoint_every > 0
                    and (n + 1) % checkpoint_every == 0):
                ckpt = ESStateSpaceCheckpoint(
                    eps_curr=eps_curr.copy(),
                    eps_initial=eps_initial.copy(),
                    best_eps=best_eps.copy(),
                    best_Q=best_Q,
                    best_target_frac=best_target_frac,
                    history=list(history),
                    next_iteration=n + 1,
                    rng_state=rng.bit_generator.state,
                    fdfd_eps=list(fdfd_eps_acc),
                    fdfd_P=list(fdfd_P_acc),
                    eps_traj=list(eps_traj_acc),
                    m_state_dict=(self.M.state_dict()
                                  if (self.M is not None and cfg.online_m)
                                  else None),
                )
                on_checkpoint(ckpt)

            # 11. Early termination.
            if best_target_frac >= 1.0 - cfg.eta:
                return ESStateSpaceResult(
                    goal=int(goal), eps_star=best_eps,
                    best_target_frac=best_target_frac,
                    best_Q=best_Q,
                    iterations=n + 1, converged=True,
                    history=history,
                    fdfd_eps=np.stack(fdfd_eps_acc).astype(np.float32),
                    fdfd_P=np.stack(fdfd_P_acc).astype(np.float64),
                    eps_initial=eps_initial,
                    eps_traj=np.stack(eps_traj_acc).astype(np.float32),
                )

        return ESStateSpaceResult(
            goal=int(goal), eps_star=best_eps,
            best_target_frac=best_target_frac, best_Q=best_Q,
            iterations=cfg.N_iter, converged=False,
            history=history,
            fdfd_eps=np.stack(fdfd_eps_acc).astype(np.float32),
            fdfd_P=np.stack(fdfd_P_acc).astype(np.float64),
            eps_initial=eps_initial,
            eps_traj=np.stack(eps_traj_acc).astype(np.float32),
        )


# =============================================================================
# Memory-bank helpers (warm-start ε for run_one_goal)
# =============================================================================

def interpolate_anchors(memory_bank: Dict[int, np.ndarray],
                        goal: int) -> np.ndarray:
    """Linear interpolation between the two nearest Phase 1 anchors.

    If `goal` is itself in the bank, returns that anchor unchanged. Mirrors
    the deploy_phase2_online.py interpolation baseline (0.143 mean tf at
    deployment); state-space ES then refines from there.
    """
    if goal in memory_bank:
        return memory_bank[goal].astype(np.float32).copy()
    known = sorted(memory_bank.keys())
    lower = max((a for a in known if a < goal), default=None)
    upper = min((a for a in known if a > goal), default=None)
    if lower is None:
        return memory_bank[upper].astype(np.float32).copy()
    if upper is None:
        return memory_bank[lower].astype(np.float32).copy()
    alpha = (goal - lower) / (upper - lower)
    return ((1 - alpha) * memory_bank[lower]
            + alpha * memory_bank[upper]).astype(np.float32)


def nearest_anchor(memory_bank: Dict[int, np.ndarray], goal: int) -> np.ndarray:
    """ε* of the Phase 1 angle closest to `goal` (ring distance)."""
    keys = sorted(memory_bank.keys())
    n = max(max(keys), goal) + 1
    nearest = min(keys, key=lambda a: min(abs(a - goal), n - abs(a - goal)))
    return memory_bank[nearest].astype(np.float32).copy()
