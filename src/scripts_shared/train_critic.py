# =============================================================================
# Post-hoc Critic Trainer — TD(0) on a Phase 1 replay buffer
# =============================================================================
"""
Train a goal-conditioned Q_ψ(ε, δ ; θ) offline using a replay buffer saved
by Phase 1. Decoupled from the actor loop (no FDFD calls; this is pure
GPU/CPU bound on a fixed buffer).

When to use this instead of `train_phase1.py --use-critic`:
  - You want ONE critic trained on ALL targets' transitions (cross-goal
    coverage), not a separate per-worker critic.
  - You want to iterate on critic hyperparameters without re-running Phase 1.
  - You ran Phase 1 cheaper (no critic) on Modal and want to do the critic
    work locally.

Usage:
    python train_critic.py --buffer phase1_training_output/replay_buffer.pkl

    # Specific hyperparameters
    python train_critic.py --buffer phase1_training_output/replay_buffer.pkl \\
        --out phase1_training_output/critic.pt \\
        --total-steps 50000 --batch-size 256 --lr 1e-3 \\
        --hidden-dim 256 --n-hidden-layers 3 \\
        --n-goals 30 --state-shape 10,10

Outputs:
    A single critic.pt at --out (default: phase1_training_output/critic.pt)
    and prints a training-loss curve.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path
_DBS = _Path(__file__).resolve().parent           # dynamic_beam_steering/
_PROJ = _DBS.parent                                # cs153 repo root
for _p in (_PROJ, _DBS):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
from algorithms.critics.dqn_critic import DQNCritic, DQNCriticConfig
from algorithms.infrastructure.utils import ReplayBuffer


def parse_args():
    p = argparse.ArgumentParser(description="Post-hoc critic trainer")
    p.add_argument("--buffer", type=Path, required=True,
                   help="Path to replay_buffer.pkl produced by a Phase 1 runner.")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to save the critic. "
                        "Default: <buffer-dir>/critic.pt")
    # Architecture
    p.add_argument("--state-shape", type=str, default=None,
                   help="\"N_x,N_y\". Default: inferred from the buffer.")
    p.add_argument("--n-goals", type=int, default=None,
                   help="One-hot dimension. Default: max(goal) + 1 from the buffer.")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    # Training
    p.add_argument("--total-steps", type=int, default=20000,
                   help="Total TD(0) gradient steps.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.995, help="Polyak averaging factor.")
    p.add_argument("--bootstrap-sigma", type=float, default=0.1,
                   help="σ for the Phase 1 Gaussian bootstrap action distribution.")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--device", type=str, default="cpu",
                   help="'cpu' or 'cuda'.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _infer_buffer_meta(transitions):
    """Read state_shape and a default n_goals from the loaded transitions."""
    state_shape = transitions[0].state.shape
    max_goal = max(t.goal for t in transitions)
    return state_shape, int(max_goal) + 1


def main():
    args = parse_args()

    # ----- load buffer ------------------------------------------------
    if not args.buffer.exists():
        raise FileNotFoundError(f"Replay buffer not found: {args.buffer}")
    print(f"Loading transitions from {args.buffer} ...")
    t0 = time.time()
    with open(args.buffer, "rb") as fh:
        transitions = pickle.load(fh)
    print(f"  {len(transitions):,} transitions loaded in {time.time() - t0:.1f}s")

    if len(transitions) == 0:
        raise ValueError("Buffer is empty.")

    # ----- infer shapes / config --------------------------------------
    inferred_state_shape, inferred_n_goals = _infer_buffer_meta(transitions)
    if args.state_shape is None:
        state_shape = inferred_state_shape
    else:
        state_shape = tuple(int(x) for x in args.state_shape.split(","))
    n_goals = args.n_goals if args.n_goals is not None else inferred_n_goals

    buffer = ReplayBuffer()
    buffer.extend(transitions)

    # ----- build critic -----------------------------------------------
    cfg = DQNCriticConfig(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        batch_size=args.batch_size,
        G=1,                            # we'll drive the loop manually
        n_goals=n_goals,
        bootstrap_sigma=args.bootstrap_sigma,
        device=args.device,
        seed=args.seed,
    )
    critic = DQNCritic(state_shape=state_shape, config=cfg)

    print()
    print(f"Critic config:")
    print(f"  state_shape={state_shape}  n_goals={n_goals}  "
          f"device={args.device}")
    print(f"  hidden_dim={args.hidden_dim} x {args.n_hidden_layers} layers")
    print(f"  γ={args.gamma}  τ={args.tau}  lr={args.lr}  "
          f"batch={args.batch_size}  σ_bootstrap={args.bootstrap_sigma}")
    print(f"  total_steps={args.total_steps:,}  log every {args.log_every}")
    print()

    # ----- training loop ----------------------------------------------
    losses = []
    t0 = time.time()
    log_chunk = args.log_every
    print(f"Training... ({args.total_steps:,} TD(0) steps)")
    for step_start in range(0, args.total_steps, log_chunk):
        chunk = min(log_chunk, args.total_steps - step_start)
        loss = critic.update(buffer, n_steps=chunk)   # returns mean loss over chunk
        if loss is None:
            print(f"  step {step_start:>6}: buffer too small for batch — abort")
            break
        losses.append(loss)
        elapsed = time.time() - t0
        steps_done = step_start + chunk
        rate = steps_done / elapsed if elapsed > 0 else 0.0
        eta = (args.total_steps - steps_done) / rate if rate > 0 else 0.0
        print(f"  step {steps_done:>6}/{args.total_steps:,}: "
              f"loss = {loss:.4e}   "
              f"({rate:.0f} steps/s, ~{eta:.0f}s remaining)")

    print()
    print(f"Done. {len(losses)} loss snapshots, "
          f"first→last: {losses[0]:.4e} → {losses[-1]:.4e}")

    # ----- save -------------------------------------------------------
    out = args.out if args.out is not None else args.buffer.parent / "critic.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    critic.save(out)
    print(f"Critic checkpoint → {out.resolve()}")

    # Also dump a tiny loss-curve sidecar so you can plot it later.
    np.save(out.with_suffix(".loss_curve.npy"), np.array(losses, dtype=np.float32))


if __name__ == "__main__":
    main()
