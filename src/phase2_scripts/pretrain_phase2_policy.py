# =============================================================================
# Phase 2 policy imitation pretraining from Phase 1 ε* configs
# =============================================================================
"""
Supervised pretraining of the Phase 2 policy π_φ(ε, θ) — bypass the Phase 1
buffer entirely (it has no usable directional signal; we verified that
empirically with the AWR sweep). Instead, treat Phase 1's converged ε*(θ)
configs as the "right answers" and supervise the policy to move toward them.

For each (state, goal) pair the target action is:
    target_action = clip(ε*_synthetic(goal) - state, -1, 1)

ε*_synthetic(goal) is:
  - The exact memory_bank entry when goal ∈ Phase 1 angles
  - Linear interpolation between the two nearest Phase 1 angles otherwise

State distribution covers (state generators are summed in the dataset):
  - exact Phase 1 ε* configs (10 anchors)
  - Phase 1 ε* + small Gaussian noise (≈ "near a converged design")
  - Uniform random ε (any state the policy might encounter)
  - Linear mixtures of two random Phase 1 ε* configs (in-between states)

The resulting policy outputs "go-toward-target-ε*" — a strong Phase 2 warm
start. Phase 2 ES can then refine from there.

Usage:
    python pretrain_phase2_policy.py \\
        --memory-bank phase1-uniform-init-output \\
        --out pretrain/policy_imitation.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import sys as _sys
from pathlib import Path as _Path
_DBS = _Path(__file__).resolve().parent           # dynamic_beam_steering/
_PROJ = _DBS.parent                                # cs153 repo root
for _p in (_PROJ, _DBS):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
from algorithms.policies.es_policy import ESPolicy, ESPolicyConfig


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Imitation pretraining for Phase 2 policy")
    p.add_argument("--memory-bank", type=Path, required=True,
                   help="Phase 1 output dir with target_<NN>/eps_star.npy.")
    p.add_argument("--out", type=Path, default=Path("pretrain/policy_imitation.pt"),
                   help="Where to save the pretrained policy.")

    # Goal space
    p.add_argument("--n-goals", type=int, default=30,
                   help="Total receivers (for sin/cos modulus). Default 30.")
    p.add_argument("--goal-indices", type=str, default=None,
                   help='Restrict training goals to this set. Default = all 30.')

    # Dataset
    p.add_argument("--n-anchor-per-pair", type=int, default=1,
                   help="Number of exact-ε* anchor samples per (state, goal) pair.")
    p.add_argument("--n-near-per-pair", type=int, default=20,
                   help="Number of 'ε* + small noise' samples per (state_angle, goal).")
    p.add_argument("--n-random-per-goal", type=int, default=100,
                   help="Number of uniform-random ε state samples per goal.")
    p.add_argument("--n-mix-per-goal", type=int, default=100,
                   help="Number of 'mix of two ε* configs' state samples per goal.")
    p.add_argument("--noise-sigma", type=float, default=0.2,
                   help="σ for the Gaussian noise around ε* anchors.")

    # Architecture (must match Phase 2 spawn defaults)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--no-tanh", action="store_true")
    p.add_argument("--tanh-output-scale", type=float, default=1.0,
                   dest="tanh_output_scale",
                   help="Cap on |δ| per element (= scale·tanh). Default 1.0 "
                        "for backward-compat. Phase 2 retarget runs use 0.25 to "
                        "produce graduated multi-step trajectories instead of "
                        "saturating jumps; pass --tanh-output-scale 0.25.")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# =============================================================================
# Memory bank load + synthetic-target interpolation
# =============================================================================

def load_memory_bank(memory_bank_dir: Path) -> dict:
    memory_bank = {}
    for target_dir in sorted(memory_bank_dir.glob("target_*")):
        if not target_dir.is_dir():
            continue
        try:
            idx = int(target_dir.name.split("_")[-1])
        except ValueError:
            continue
        path = target_dir / "eps_star.npy"
        if path.exists():
            memory_bank[idx] = np.load(path).astype(np.float32)
    if not memory_bank:
        raise FileNotFoundError(f"No eps_star.npy under {memory_bank_dir}")
    return memory_bank


def synthesize_target_eps(goal: int, memory_bank: dict) -> np.ndarray:
    """Return ε*(goal): exact when goal ∈ memory_bank keys, else linear
    interpolation between the two nearest Phase 1 angles."""
    if goal in memory_bank:
        return memory_bank[goal].copy()
    known = sorted(memory_bank.keys())
    lower = max((a for a in known if a < goal), default=None)
    upper = min((a for a in known if a > goal), default=None)
    if lower is None:
        return memory_bank[upper].copy()
    if upper is None:
        return memory_bank[lower].copy()
    alpha = (goal - lower) / (upper - lower)
    return ((1 - alpha) * memory_bank[lower] + alpha * memory_bank[upper]).astype(np.float32)


# =============================================================================
# Dataset construction
# =============================================================================

def build_dataset(memory_bank: dict, goal_indices: list, args, rng):
    """Build (states, goals, target_actions) triples covering several state regimes."""
    known_angles = sorted(memory_bank.keys())
    state_shape = next(iter(memory_bank.values())).shape

    states, goals, actions = [], [], []

    # Pre-compute synthetic target ε for each goal (used many times).
    target_eps_by_goal = {g: synthesize_target_eps(g, memory_bank) for g in goal_indices}

    # Clip target action to the policy's output range [-scale, scale]. Without
    # this, when tanh_output_scale < 1.0 the policy can never match large
    # targets and the MSE loss has a non-zero floor.
    scale = getattr(args, "tanh_output_scale", 1.0)

    def _append(state, goal, target_eps):
        state = np.asarray(state, dtype=np.float32)
        target_action = np.clip(target_eps - state, -scale, scale).astype(np.float32)
        states.append(state.copy())
        goals.append(int(goal))
        actions.append(target_action)

    # 1. Anchor states (exact ε* configs) × all goals
    for state_angle in known_angles:
        state_ref = memory_bank[state_angle]
        for goal in goal_indices:
            target_eps = target_eps_by_goal[goal]
            for _ in range(args.n_anchor_per_pair):
                _append(state_ref, goal, target_eps)

    # 2. Near-anchor: ε* + small Gaussian noise (state still resembles a converged design)
    for state_angle in known_angles:
        state_ref = memory_bank[state_angle]
        for goal in goal_indices:
            target_eps = target_eps_by_goal[goal]
            for _ in range(args.n_near_per_pair):
                noise = rng.normal(0, args.noise_sigma,
                                   size=state_shape).astype(np.float32)
                noisy_state = np.clip(state_ref + noise, -1.0, 1.0).astype(np.float32)
                _append(noisy_state, goal, target_eps)

    # 3. Uniform random ε states × goals
    for goal in goal_indices:
        target_eps = target_eps_by_goal[goal]
        for _ in range(args.n_random_per_goal):
            random_state = rng.uniform(-1, 1, size=state_shape).astype(np.float32)
            _append(random_state, goal, target_eps)

    # 4. Mixtures of two ε* configs × goals (in-between trajectory states)
    for goal in goal_indices:
        target_eps = target_eps_by_goal[goal]
        for _ in range(args.n_mix_per_goal):
            a, b = rng.choice(known_angles, size=2, replace=False)
            alpha = float(rng.uniform(0, 1))
            mix_state = (alpha * memory_bank[a] + (1 - alpha) * memory_bank[b]).astype(np.float32)
            _append(mix_state, goal, target_eps)

    states_arr = np.stack(states)
    goals_arr = np.array(goals, dtype=np.int64)
    actions_arr = np.stack(actions)
    return states_arr, goals_arr, actions_arr


# =============================================================================
# Training loop
# =============================================================================

def train(policy: ESPolicy, states, goals, actions, args, rng):
    N = len(states)
    idx = rng.permutation(N)
    n_val = max(1, int(N * args.val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    states_t = torch.as_tensor(states, dtype=torch.float32, device=policy.device)
    goals_t = torch.as_tensor(goals, dtype=torch.long, device=policy.device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=policy.device)

    opt = optim.Adam(policy.pi.parameters(), lr=args.lr)
    train_hist, val_hist = [], []

    for epoch in range(args.epochs):
        # Train
        policy.pi.train()
        perm = rng.permutation(len(train_idx))
        shuffled = train_idx[perm]
        epoch_loss = 0.0
        n_batches = 0
        for batch_start in range(0, len(shuffled), args.batch_size):
            bi = shuffled[batch_start:batch_start + args.batch_size]
            bi_t = torch.as_tensor(bi, dtype=torch.long, device=policy.device)
            pred = policy.pi(states_t[bi_t], goals_t[bi_t])
            loss = F.mse_loss(pred, actions_t[bi_t])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss)
            n_batches += 1
        train_hist.append(epoch_loss / max(n_batches, 1))

        # Val (plain MSE)
        policy.pi.eval()
        with torch.no_grad():
            vi_t = torch.as_tensor(val_idx, dtype=torch.long, device=policy.device)
            pred_val = policy.pi(states_t[vi_t], goals_t[vi_t])
            v = float(F.mse_loss(pred_val, actions_t[vi_t]))
        val_hist.append(v)

        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"  epoch {epoch + 1:>3}/{args.epochs}:  "
                  f"train={train_hist[-1]:.4e}  val={v:.4e}")

    return train_hist, val_hist, len(train_idx), n_val


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Memory bank
    memory_bank = load_memory_bank(args.memory_bank)
    print(f"Memory bank: {sorted(memory_bank.keys())}  "
          f"(state_shape={next(iter(memory_bank.values())).shape})")

    # Goal indices
    if args.goal_indices is not None:
        goal_indices = [int(x) for x in args.goal_indices.split(",")]
    else:
        goal_indices = list(range(args.n_goals))
    print(f"Training over goals: {goal_indices}  ({len(goal_indices)} angles)")

    # Build dataset
    print("Building dataset...")
    t0 = time.time()
    states, goals, actions = build_dataset(memory_bank, goal_indices, args, rng)
    print(f"  {len(states):,} (state, goal, target_action) samples "
          f"({time.time() - t0:.1f}s)")
    print(f"  action stats: |target| mean={np.linalg.norm(actions, axis=(1, 2)).mean():.3f}  "
          f"max={np.abs(actions).max():.3f}")

    # Policy
    state_shape = states.shape[1:]
    pcfg = ESPolicyConfig(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        tanh_output=not args.no_tanh,
        tanh_output_scale=args.tanh_output_scale,
        n_goals=args.n_goals,
        device=args.device,
        seed=args.seed,
    )
    policy = ESPolicy(state_shape=state_shape, config=pcfg)
    n_params = sum(p.numel() for p in policy.pi.parameters())
    print(f"\nPolicy: state_shape={state_shape}  n_goals={args.n_goals}  "
          f"hidden={args.hidden_dim}×{args.n_hidden_layers}  params={n_params:,}  "
          f"tanh_output_scale={args.tanh_output_scale}")

    # Train
    print(f"\nTraining {args.epochs} epochs over {len(states):,} samples...")
    t0 = time.time()
    train_hist, val_hist, n_train, n_val = train(policy, states, goals, actions, args, rng)
    elapsed = time.time() - t0
    print(f"  finished in {elapsed:.1f}s")
    print(f"  train_loss: first→last  {train_hist[0]:.4e} → {train_hist[-1]:.4e}")
    print(f"  val_loss:   first→last  {val_hist[0]:.4e} → {val_hist[-1]:.4e}")
    print(f"  n_train={n_train:,}  n_val={n_val:,}")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.out)
    print(f"\nPolicy → {args.out.resolve()}")

    np.savez(args.out.with_suffix(".loss.npz"),
             train_loss=np.array(train_hist, dtype=np.float32),
             val_loss=np.array(val_hist, dtype=np.float32),
             n_train=n_train, n_val=n_val)


if __name__ == "__main__":
    main()
