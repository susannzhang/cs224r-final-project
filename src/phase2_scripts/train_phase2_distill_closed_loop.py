"""Distill ESStateSpacePolicy trajectories into a deployable closed-loop π_φ.

Consumes per-goal `eps_traj.npy` files produced by the state-space-ES Modal
pipeline (`train_phase2_state_space_modal.py`) and trains a small MLP
π_φ(ε_t, θ) → δ_t via plain MSE regression on the (ε_t, ε_{t+1} − ε_t, goal)
tuples extracted from those trajectories.

The output is a `policy_distilled.pt` checkpoint compatible with
`algorithms.policies.es_policy.ESPolicy.load()` — so the deployment side
(closed-loop rollout, evaluation harness) doesn't need to change. At
inference:

    ε_0 ← anchor warm-start (interp / nearest / uniform — caller's choice)
    for t = 0, ..., T-1:
        δ_t = π_φ(ε_t, goal)
        ε_{t+1} = clip(ε_t + δ_t, [-1, 1])
    return ε_T

Why this works where the original ESPolicy parameter-space-ES warm-start
didn't: the supervision here is "what one ES iter step produced from this
state" — i.e., FDFD-verified descent directions, not noisy random
perturbations from a Phase 1 single-step buffer. The distilled π_φ imitates
the algorithmic solver, then runs at sub-ms inference cost.

Usage:
    python train_phase2_distill_closed_loop.py \\
        --runs-dir phase2_state_space_output/phase2-ss-20260603-090000 \\
        --hidden-dim 256 --n-hidden-layers 3 --tanh-output-scale 0.5 \\
        --epochs 200 --batch-size 128 --lr 1e-3 \\
        --out pretrain/policy_distilled.pt

Multiple runs can be combined via repeated --runs-dir:
    python train_phase2_distill_closed_loop.py \\
        --runs-dir phase2_state_space_output/run-A \\
        --runs-dir phase2_state_space_output/run-B \\
        --out pretrain/policy_distilled_combined.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent          # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                       # cs153 repo root (geometry, simulation)
for _p in (PROJECT_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from algorithms.policies.es_policy import ESPolicy, ESPolicyConfig  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Distill ESStateSpacePolicy trajectories → closed-loop π_φ")

    # --- Data sources ---------------------------------------------------
    p.add_argument("--runs-dir", action="append", required=True,
                   dest="runs_dirs", type=Path,
                   help="Directory containing goal_<NN>/eps_traj.npy files. "
                        "Pass multiple times to combine runs.")
    p.add_argument("--include-goals", type=str, default=None,
                   help="Comma-separated whitelist of goal indices to include "
                        "(default: all found).")
    p.add_argument("--exclude-goals", type=str, default=None,
                   help="Comma-separated blacklist of goal indices to exclude.")

    # --- Output ---------------------------------------------------------
    p.add_argument("--out", type=Path,
                   default=Path("pretrain/policy_distilled.pt"))

    # --- Architecture (matches ESPolicyConfig) -------------------------
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-hidden-layers", type=int, default=3)
    p.add_argument("--n-goals", type=int, default=30,
                   help="Total receiver count for sin/cos goal encoding.")
    p.add_argument("--tanh-output", action="store_true", default=True)
    p.add_argument("--no-tanh-output", action="store_false", dest="tanh_output")
    p.add_argument("--tanh-output-scale", type=float, default=0.5,
                   help="Cap on |δ| per element (scale·tanh). ES per-step "
                        "δ = α·grad ≈ O(0.05); scale=0.5 leaves >5σ headroom "
                        "without tanh saturating at the distilled targets.")

    # --- Training -------------------------------------------------------
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--val-by-goal", action="store_true",
                   help="Hold out ENTIRE goals (random subset) for validation "
                        "instead of mixing transitions. Tests cross-goal "
                        "generalization rather than within-trajectory fit.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=10)

    # --- Target formulation --------------------------------------------
    # The supervision target δ_t for π_φ(ε_t, θ). Three options, each
    # trading variance vs. fidelity-to-ES differently:
    #
    #   "step" (the original): δ_t = ε_{t+1} − ε_t. Faithful to the ES
    #       descent procedure but per-iter ES gradient is high-variance —
    #       a single (ε_t, δ_t) pair is essentially noise around the true
    #       descent direction. Cos(δ_pred, δ_true) on val typically ~0.02.
    #       Use if you want π_φ to imitate ES *steps* exactly.
    #
    #   "horizon": δ_t = (ε_{t+H} − ε_t) / H. Averages the noisy single-step
    #       gradient over H iters, exposing the underlying descent direction.
    #       Choose H ~ 5–20; larger H = smoother target but loses the late-
    #       trajectory fine-tuning regime. Recommended when the ES procedure
    #       itself is the imitation target but per-step noise is a problem.
    #
    #   "eps_star" (recommended for the stated objective): δ_t = α·(ε* − ε_t)
    #       where ε* is the trajectory's final state (or eps_star.npy if
    #       available). The policy learns a smooth vector field that points
    #       to the goal's converged ε*, independent of how ES happened to
    #       reach it. At deployment, iterating π_φ gives geometric approach
    #       to ε*(θ). Highest signal-to-noise per supervision tuple.
    p.add_argument("--target-mode", type=str, default="eps_star",
                   choices=["step", "horizon", "eps_star"],
                   help='Supervision target. "step" = ε_{t+1}-ε_t (noisy); '
                        '"horizon" = (ε_{t+H}-ε_t)/H (averaged); '
                        '"eps_star" = α·(ε*-ε_t) (vector field to converged ε*).')
    p.add_argument("--target-horizon", type=int, default=10,
                   help="H for --target-mode=horizon. Default 10.")
    p.add_argument("--target-eps-star-alpha", type=float, default=0.1,
                   help="α scaling for --target-mode=eps_star. δ = α·(ε*-ε_t). "
                        "α=0.1 implies ~10 deployment steps to converge "
                        "(linear gap-closing approximation). Default 0.1.")
    p.add_argument("--clip-target-delta", type=float, default=None,
                   help="If set, clip |δ_target| to this magnitude before "
                        "loss. Defensive against outlier ES updates "
                        "(early iters with σ-annealing schedules can produce "
                        "large δ_t that tanh wouldn't fit). Default: off.")
    p.add_argument("--drop-trailing-steps", type=int, default=0,
                   help="Skip the last N transitions of each trajectory (the "
                        "ones near ε* where δ→0 dominate the loss). Default 0.")
    return p.parse_args()


def discover_trajectories(runs_dirs, include, exclude):
    """Walk all `goal_<NN>/eps_traj.npy` under the given run dirs.

    Returns a list of dicts: {goal, eps_traj, run_dir}. Skips dirs without
    an eps_traj.npy (older runs, in-flight ones).
    """
    found = []
    for run_dir in runs_dirs:
        if not run_dir.exists():
            print(f"  WARN: {run_dir} does not exist; skipping")
            continue
        for goal_dir in sorted(run_dir.glob("goal_*")):
            if not goal_dir.is_dir():
                continue
            try:
                goal = int(goal_dir.name.split("_")[-1])
            except ValueError:
                continue
            if include is not None and goal not in include:
                continue
            if exclude is not None and goal in exclude:
                continue
            traj_path = goal_dir / "eps_traj.npy"
            if not traj_path.exists():
                print(f"  skip {goal_dir.name}: no eps_traj.npy")
                continue
            eps_traj = np.load(traj_path)
            if eps_traj.ndim != 3 or eps_traj.shape[0] < 2:
                print(f"  skip {goal_dir.name}: trajectory too short "
                      f"({eps_traj.shape})")
                continue
            eps_star_path = goal_dir / "eps_star.npy"
            found.append({
                "goal": goal,
                "eps_traj": eps_traj,
                "run_dir": str(run_dir),
                "eps_star_path": (str(eps_star_path)
                                  if eps_star_path.exists() else None),
            })
            print(f"  + goal {goal:>2} ({run_dir.name}): "
                  f"{eps_traj.shape[0]} states → "
                  f"{eps_traj.shape[0] - 1} transitions")
    return found


def _resolve_eps_star(traj_entry):
    """Pick ε* for a trajectory.

    Prefer an explicit `eps_star.npy` in the same goal_<NN>/ directory (the
    Modal pipeline's best-by-Q final state, slightly more accurate than the
    trajectory's mean endpoint); fall back to the trajectory's final state.
    """
    explicit = traj_entry.get("eps_star_path")
    if explicit is not None and Path(explicit).exists():
        return np.load(explicit).astype(np.float32)
    return traj_entry["eps_traj"][-1].astype(np.float32)


def build_supervision(trajectories, target_mode, target_horizon,
                      target_eps_star_alpha, drop_trailing_steps,
                      clip_target_delta):
    """Stack (ε_t, δ_t, goal) supervision tuples per --target-mode.

    Returns concatenated arrays (eps, delta, goal, goal_idx_for_grouping).
    """
    eps_all, delta_all, goal_all, goal_idx_all = [], [], [], []
    for i, t in enumerate(trajectories):
        eps_traj = t["eps_traj"]
        L = eps_traj.shape[0]

        if target_mode == "step":
            T = L - 1 - drop_trailing_steps
            if T <= 0:
                continue
            eps_t = eps_traj[:T]
            delta_t = (eps_traj[1:T + 1] - eps_t).astype(np.float32)

        elif target_mode == "horizon":
            H = target_horizon
            T = L - H - drop_trailing_steps
            if T <= 0:
                continue
            eps_t = eps_traj[:T]
            # Average over H iters → smoother descent direction than the
            # noisy per-iter ES gradient. Scale by 1/H so the per-step
            # magnitude lives in the same range as the "step" target.
            delta_t = ((eps_traj[H:H + T] - eps_t) / H).astype(np.float32)

        elif target_mode == "eps_star":
            T = L - drop_trailing_steps
            if T <= 0:
                continue
            eps_star = _resolve_eps_star(t)
            eps_t = eps_traj[:T]
            # Smooth vector field pointing toward ε*. Same target direction
            # for all (ε_t, goal) pairs of the same trajectory — high SNR.
            delta_t = (target_eps_star_alpha
                       * (eps_star[None] - eps_t)).astype(np.float32)

        else:
            raise ValueError(f"Unknown target_mode={target_mode!r}")

        if clip_target_delta is not None and clip_target_delta > 0:
            delta_t = np.clip(delta_t, -clip_target_delta, clip_target_delta)
        eps_all.append(eps_t.astype(np.float32))
        delta_all.append(delta_t)
        goal_all.append(np.full(len(eps_t), t["goal"], dtype=np.int64))
        goal_idx_all.append(np.full(len(eps_t), i, dtype=np.int64))
    return (np.concatenate(eps_all, axis=0),
            np.concatenate(delta_all, axis=0),
            np.concatenate(goal_all, axis=0),
            np.concatenate(goal_idx_all, axis=0))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    include = (set(int(x) for x in args.include_goals.split(","))
               if args.include_goals else None)
    exclude = (set(int(x) for x in args.exclude_goals.split(","))
               if args.exclude_goals else None)

    print(f"Discovering trajectories under: "
          f"{[str(p) for p in args.runs_dirs]}")
    trajectories = discover_trajectories(args.runs_dirs, include, exclude)
    if not trajectories:
        raise RuntimeError("No trajectories found. Check --runs-dir.")
    print(f"\nFound {len(trajectories)} trajectories "
          f"({sorted(set(t['goal'] for t in trajectories))})")

    eps, delta, goal, goal_idx = build_supervision(
        trajectories,
        target_mode=args.target_mode,
        target_horizon=args.target_horizon,
        target_eps_star_alpha=args.target_eps_star_alpha,
        drop_trailing_steps=args.drop_trailing_steps,
        clip_target_delta=args.clip_target_delta,
    )
    state_shape = tuple(eps.shape[1:])
    N = len(eps)
    print(f"\nTarget mode: {args.target_mode}", end="")
    if args.target_mode == "horizon":
        print(f"  (H={args.target_horizon})")
    elif args.target_mode == "eps_star":
        print(f"  (α={args.target_eps_star_alpha})")
    else:
        print()
    print(f"Supervision: {N:,} transitions  "
          f"state_shape={state_shape}  goals={len(set(goal.tolist()))}")
    print(f"  δ stats:  mean|δ|={np.abs(delta).mean():.4e}  "
          f"max|δ|={np.abs(delta).max():.4e}  "
          f"std={delta.std():.4e}")
    print(f"  ε stats:  mean={eps.mean():.3f}  std={eps.std():.3f}  "
          f"min={eps.min():.3f}  max={eps.max():.3f}")

    # --- Train/val split -------------------------------------------------
    if args.val_by_goal:
        n_traj = len(trajectories)
        n_val_traj = max(1, int(round(n_traj * args.val_split)))
        perm = rng.permutation(n_traj)
        val_traj = set(int(i) for i in perm[:n_val_traj])
        is_val = np.isin(goal_idx, list(val_traj))
        val_idx = np.where(is_val)[0]
        train_idx = np.where(~is_val)[0]
        print(f"  val-by-goal: held-out trajs={sorted(val_traj)}  "
              f"({len(val_idx)} transitions val, {len(train_idx)} train)")
    else:
        perm = rng.permutation(N)
        n_val = max(1, int(N * args.val_split))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        print(f"  random-transition split: train={len(train_idx)} "
              f"val={len(val_idx)}")

    # --- Build ESPolicy --------------------------------------------------
    pcfg = ESPolicyConfig(
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        n_goals=args.n_goals,
        tanh_output=args.tanh_output,
        tanh_output_scale=args.tanh_output_scale,
        device=args.device,
        seed=args.seed,
    )
    policy = ESPolicy(state_shape=state_shape, config=pcfg)
    n_params = sum(p.numel() for p in policy.pi.parameters())
    print(f"\nπ_φ: state_shape={state_shape}  n_goals={args.n_goals}  "
          f"hidden={args.hidden_dim}×{args.n_hidden_layers}  "
          f"tanh_scale={args.tanh_output_scale}  params={n_params:,}")

    # --- Tensors ---------------------------------------------------------
    device = torch.device(args.device)
    eps_t = torch.as_tensor(eps, dtype=torch.float32, device=device)
    delta_t = torch.as_tensor(delta, dtype=torch.float32, device=device)
    goal_t = torch.as_tensor(goal, dtype=torch.long, device=device)

    optimizer = optim.Adam(policy.pi.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    train_hist, val_hist = [], []
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device)

    print(f"\nTraining {args.epochs} epochs  "
          f"(lr={args.lr}, batch={args.batch_size}, wd={args.weight_decay})")
    t0 = time.time()
    for epoch in range(args.epochs):
        policy.pi.train()
        shuffled = train_idx_t[torch.randperm(len(train_idx_t), device=device)]
        ep_loss, n_batches = 0.0, 0
        for bs in range(0, len(shuffled), args.batch_size):
            bi = shuffled[bs:bs + args.batch_size]
            pred = policy.pi(eps_t[bi], goal_t[bi])
            loss = F.mse_loss(pred, delta_t[bi])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        train_hist.append(ep_loss / max(n_batches, 1))

        policy.pi.eval()
        with torch.no_grad():
            pred_val = policy.pi(eps_t[val_idx_t], goal_t[val_idx_t])
            val_loss = F.mse_loss(pred_val, delta_t[val_idx_t]).item()
            # Cosine sim between predicted and target δ — direction quality,
            # which matters for closed-loop descent more than the per-element MSE.
            cos = F.cosine_similarity(
                pred_val.reshape(len(val_idx_t), -1),
                delta_t[val_idx_t].reshape(len(val_idx_t), -1),
                dim=-1,
            )
            val_cos_mean = float(cos.mean())
        val_hist.append(val_loss)

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>4}/{args.epochs}:  "
                  f"train={train_hist[-1]:.4e}  "
                  f"val={val_loss:.4e}  "
                  f"val_cos(δ)={val_cos_mean:+.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.  final train={train_hist[-1]:.4e}  "
          f"val={val_hist[-1]:.4e}")

    # --- Save -----------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.out)
    np.savez(args.out.with_suffix(".loss.npz"),
             train_loss=np.array(train_hist, dtype=np.float32),
             val_loss=np.array(val_hist, dtype=np.float32))
    (args.out.with_suffix(".meta.json")).write_text(json.dumps({
        "runs_dirs": [str(p) for p in args.runs_dirs],
        "n_trajectories": len(trajectories),
        "n_transitions": int(N),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "val_by_goal": bool(args.val_by_goal),
        "trajectory_goals": sorted(set(int(t["goal"]) for t in trajectories)),
        "config": {
            "hidden_dim": args.hidden_dim,
            "n_hidden_layers": args.n_hidden_layers,
            "tanh_output_scale": args.tanh_output_scale,
            "n_goals": args.n_goals,
        },
        "target": {
            "mode": args.target_mode,
            "horizon": args.target_horizon,
            "eps_star_alpha": args.target_eps_star_alpha,
            "drop_trailing_steps": args.drop_trailing_steps,
            "clip_target_delta": args.clip_target_delta,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "final_train_loss": float(train_hist[-1]),
        "final_val_loss": float(val_hist[-1]),
        "delta_stats": {
            "mean_abs": float(np.abs(delta).mean()),
            "max_abs": float(np.abs(delta).max()),
            "std": float(delta.std()),
        },
        "elapsed_seconds": elapsed,
    }, indent=2))
    print(f"\nSaved π_distilled → {args.out.resolve()}")
    print(f"Losses → {args.out.with_suffix('.loss.npz').resolve()}")
    print(f"Meta   → {args.out.with_suffix('.meta.json').resolve()}")
    print(f"\nLoad and deploy with:")
    print(f"  from algorithms.policies.es_policy import ESPolicy")
    print(f"  policy = ESPolicy.load('{args.out}')")
    print(f"  delta = policy.predict(eps, goal)")


if __name__ == "__main__":
    main()
