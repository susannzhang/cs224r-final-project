# =============================================================================
# AWR trainer — Phase 2 policy π_φ warm-start from Phase 1 buffer
# =============================================================================
"""
Advantage-weighted regression warm-start. Replaces the old BC init.

    φ ← arg min_φ  E_{(ε, δ, r, θ) ∼ B}  w(r, θ) · ‖π_φ(ε; θ) - δ‖²

w(r, θ) = exp((r - baseline_θ) / (std(A) · β)), clipped, mean-normalized.
High-reward transitions dominate the gradient; low-reward ones contribute
but at exponentially-decayed weight.

Usage:
    # Default: uniform-init buffer, all defaults
    python train_awr.py --buffer phase1-uniform-init-output/replay_buffer.pkl

    # Tuning knobs
    python train_awr.py --buffer phase1-uniform-init-output/replay_buffer.pkl \\
        --out phase1-uniform-init-output/policy_awr.pt \\
        --epochs 50 --batch-size 256 --lr 1e-3 \\
        --beta 1.0 --baseline per_goal_median --clip 20.0 \\
        --hidden-dim 256 --n-hidden-layers 3 \\
        --n-goals 30 --state-shape 10,10

Outputs:
    <out>             — torch checkpoint with policy weights + config
    <out>.loss.npz    — train_loss / val_loss / weight_stats arrays
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_DBS = _Path(__file__).resolve().parent           # dynamic_beam_steering/
_PROJ = _DBS.parent                                # cs153 repo root
for _p in (_PROJ, _DBS):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
from algorithms.infrastructure.utils import ReplayBuffer
from algorithms.policies.es_policy import ESPolicy, ESPolicyConfig


def parse_args():
    p = argparse.ArgumentParser(description="AWR warm-start for the Phase 2 policy")
    p.add_argument("--buffer", type=Path, required=True,
                   help="Path to replay_buffer.pkl produced by a Phase 1 runner.")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to save the policy. Default: <buffer-dir>/policy_awr.pt")
    # Architecture
    p.add_argument("--state-shape", type=str, default=None,
                   help='"N_x,N_y". Default: inferred from the buffer.')
    p.add_argument("--n-goals", type=int, default=None,
                   help="One-hot dimension. Default: max(goal) + 1 in the buffer.")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--no-tanh", action="store_true",
                   help="Disable tanh output squash (raw linear δ).")
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    # AWR-specific
    p.add_argument("--beta", type=float, default=1.0,
                   help="Smaller β → sharper preference for high-reward actions.")
    p.add_argument("--baseline", type=str, default="per_goal_median",
                   choices=["per_goal_median", "per_goal_mean", "global_median"])
    p.add_argument("--clip", type=float, default=20.0,
                   help="Cap per-sample weight to prevent single-sample dominance.")
    # Platform
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _infer_buffer_meta(transitions):
    state_shape = transitions[0].state.shape
    max_goal = max(t.goal for t in transitions)
    return state_shape, int(max_goal) + 1


def main():
    args = parse_args()

    if not args.buffer.exists():
        raise FileNotFoundError(f"Replay buffer not found: {args.buffer}")
    print(f"Loading transitions from {args.buffer} ...")
    t0 = time.time()
    with open(args.buffer, "rb") as fh:
        transitions = pickle.load(fh)
    print(f"  {len(transitions):,} transitions loaded in {time.time() - t0:.1f}s")
    if len(transitions) == 0:
        raise ValueError("Buffer is empty.")

    inferred_state_shape, inferred_n_goals = _infer_buffer_meta(transitions)
    state_shape = (tuple(int(x) for x in args.state_shape.split(","))
                   if args.state_shape else inferred_state_shape)
    n_goals = args.n_goals if args.n_goals is not None else inferred_n_goals

    buffer = ReplayBuffer()
    buffer.extend(transitions)

    cfg = ESPolicyConfig(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        tanh_output=not args.no_tanh,
        awr_epochs=args.epochs,
        awr_batch_size=args.batch_size,
        awr_lr=args.lr,
        awr_validation_split=args.val_split,
        awr_beta=args.beta,
        awr_baseline=args.baseline,
        awr_clip=args.clip,
        n_goals=n_goals,
        device=args.device,
        seed=args.seed,
    )
    policy = ESPolicy(state_shape=state_shape, config=cfg)

    print()
    print("Policy config:")
    print(f"  state_shape={state_shape}  n_goals={n_goals}  device={args.device}")
    print(f"  hidden_dim={args.hidden_dim} x {args.n_hidden_layers} layers"
          f"   tanh_output={cfg.tanh_output}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}"
          f"   val_split={args.val_split}")
    print(f"  β={args.beta}  baseline={args.baseline}  clip={args.clip}")
    print()

    print(f"AWR training... ({args.epochs} epochs over {len(buffer):,} transitions)")
    t0 = time.time()
    history = policy.awr_init(buffer)
    elapsed = time.time() - t0
    print(f"  finished in {elapsed:.1f}s")
    train = history["train_loss"]
    val = history["val_loss"]
    ws = history["weight_stats"]
    print(f"  train_loss: first→last  {train[0]:.4e} → {train[-1]:.4e}")
    print(f"  val_loss:   first→last  {val[0]:.4e} → {val[-1]:.4e}")
    print(f"  weight_stats: min={ws['min']:.3e}  max={ws['max']:.3e}  "
          f"mean={ws['mean']:.3f}  median={ws['median']:.3e}  "
          f"frac_at_clip={ws['frac_at_clip']:.3%}")
    print(f"  n_train={history['n_train']:,}  n_val={history['n_val']:,}")

    out = args.out if args.out is not None else args.buffer.parent / "policy_awr.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(out)
    print()
    print(f"Policy → {out.resolve()}")

    np.savez(
        out.with_suffix(".loss.npz"),
        train_loss=np.array(train, dtype=np.float32),
        val_loss=np.array(val, dtype=np.float32),
        weight_min=ws["min"], weight_max=ws["max"],
        weight_mean=ws["mean"], weight_median=ws["median"],
        weight_frac_at_clip=ws["frac_at_clip"],
        n_train=history["n_train"], n_val=history["n_val"],
    )


if __name__ == "__main__":
    main()
