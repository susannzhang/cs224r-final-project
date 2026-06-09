"""Phase 2 imitation pretraining from Phase 1's actual ES mean trajectories.

Self-contained alternative to pretrain_phase2_policy.py. Where the synthetic
script builds imitation targets as "clip(ε*(goal) - state, ±scale)" (a
jump-to-target heuristic), this script extracts the actual sequence of ES
mean ε values Phase 1 visited en route to each ε*(θ_j), and uses the
consecutive differences (state → next_state actions) as the imitation
targets. The result is a policy that mimics Phase 1's gradient dynamics
rather than its endpoint.

Distills into any policy architecture via --policy-arch {cnn,mlp,pinn}
(default: pinn). For the PINN, --physics-loss-weight > 0 adds the Helmholtz
residual to the imitation MSE so the physics prior stays active during
distillation. The output is an ESPolicy checkpoint (config carries the arch),
so it loads straight into the grad-refine driver
(train_phase2_grad_learned_angles.py).

Usage:
    python pretrain_phase2_from_buffer.py \\
        --buffer phase1-uniform-init-output/replay_buffer.pkl \\
        --memory-bank phase1-uniform-init-output \\
        --policy-arch pinn --physics-loss-weight 0.1 \\
        --hidden-dim 100 --n-hidden-layers 2 \\
        --tanh-output-scale 1.0 \\
        --out pretrain/policy_buffer_traj_pinn.pt
"""

import argparse
import pickle
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


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 pretrain from Phase 1 buffer trajectories")
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--memory-bank", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("pretrain/policy_buffer_traj.pt"))

    # Augmentation
    p.add_argument("--n-noise-per-state", type=int, default=5,
                   help="Noisy copies of each trajectory state (same action target).")
    p.add_argument("--noise-sigma", type=float, default=0.05)

    # Goal-conditioning generalization to Phase 2 angles
    p.add_argument("--n-goals", type=int, default=30,
                   help="Goal encoding modulus (sin/cos). Default 30.")

    # Architecture
    p.add_argument("--policy-arch", choices=["cnn", "mlp", "pinn"], default="pinn",
                   dest="policy_arch",
                   help="Architecture to distill into (default: pinn).")
    p.add_argument("--hidden-dim", type=int, default=100)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--no-tanh", action="store_true")
    p.add_argument("--tanh-output-scale", type=float, default=1.0,
                   dest="tanh_output_scale")
    p.add_argument("--physics-loss-weight", type=float, default=0.0,
                   dest="physics_loss_weight",
                   help="PINN only: weight on the Helmholtz residual added to the "
                        "imitation MSE (keeps the field-consistency prior active "
                        "during distillation). No-op for cnn/mlp.")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_memory_bank(d: Path) -> dict:
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
    if not bank:
        raise FileNotFoundError(f"No eps_star.npy under {d}")
    return bank


def extract_trajectories(buffer: list, memory_bank: dict,
                         training_order: list) -> list:
    """Recover per-angle ES mean trajectories from the buffer.

    Phase 1 ran ES sequentially per angle in `training_order`. Each iter
    appends K candidates × |training_indices| transitions, all sharing the
    same `state` field (the ES mean ε at that iter). Globally dedupping by
    state (preserving order) gives the concatenated trajectory across all
    angles; the first state is ε=0 (shared init), followed by ~M unique
    mean states per angle, in the same order Phase 1 trained them.

    L2 matching to memory_bank ε* is unreliable here because Phase 1 didn't
    converge (250 iters, summary.json shows `converged: false`), so the ES
    mean's final position is offset from the best candidate ε* by mean-walk
    + σ·ξ_k_best. We use the explicit training_order instead.

    Returns: [(angle_idx, [state_0, ..., state_M]), ...]
    """
    seen, ordered = set(), []
    for t in buffer:
        h = t.state.tobytes()
        if h not in seen:
            seen.add(h)
            ordered.append(np.asarray(t.state, dtype=np.float32))
    print(f"Unique mean states in buffer: {len(ordered)}")

    if not np.allclose(ordered[0], 0):
        raise ValueError(
            f"First unique state is not ε=0 "
            f"(norm={np.linalg.norm(ordered[0]):.3f}); buffer order is unexpected.")
    eps_zero = ordered[0]

    remaining = ordered[1:]
    n_angles = len(training_order)
    states_per = len(remaining) // n_angles
    print(f"Splitting {len(remaining)} states into {n_angles} groups × ~{states_per}, "
          f"assigned in training_order={training_order}")

    trajectories = []
    for i, angle in enumerate(training_order):
        start = i * states_per
        end = (i + 1) * states_per if i < n_angles - 1 else len(remaining)
        group = remaining[start:end]
        # Prepend ε=0 (shared init).
        traj = [eps_zero] + group
        final = group[-1]
        l2_to_bank = float(np.linalg.norm(final - memory_bank[angle]))
        trajectories.append((angle, traj))
        print(f"  group {i}: angle {angle:>2}  {len(traj)} states  "
              f"|ε|_final={np.linalg.norm(final):.2f}  "
              f"L2(final → ε*({angle}))={l2_to_bank:.2f}")

    return trajectories


def build_dataset(trajectories: list, args, rng) -> tuple:
    """Build (states, goals, target_actions) from trajectories.

    For each consecutive pair (ε_n, ε_{n+1}) in a trajectory for angle A,
    the imitation target is action = ε_{n+1} - ε_n (the ES mean update).
    Augment with near-trajectory Gaussian noise — same target action,
    perturbed input state (helps the policy generalize to the off-trajectory
    states it'll see during Phase 2 rollouts).
    """
    states, goals, actions = [], [], []
    for angle, traj in trajectories:
        for n in range(len(traj) - 1):
            s = traj[n]
            action = (traj[n + 1] - s).astype(np.float32)

            # Anchor (exact trajectory state, 1 copy)
            states.append(s.copy())
            goals.append(int(angle))
            actions.append(action.copy())

            # Noisy copies (target action unchanged)
            for _ in range(args.n_noise_per_state):
                noise = rng.normal(0, args.noise_sigma,
                                   size=s.shape).astype(np.float32)
                noisy_state = np.clip(s + noise, -1.0, 1.0)
                states.append(noisy_state)
                goals.append(int(angle))
                actions.append(action.copy())

    return (np.stack(states),
            np.array(goals, dtype=np.int64),
            np.stack(actions))


def train(policy, states, goals, actions, args, rng):
    N = len(states)
    idx = rng.permutation(N)
    n_val = max(1, int(N * args.val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    states_t = torch.as_tensor(states, dtype=torch.float32, device=policy.device)
    goals_t = torch.as_tensor(goals, dtype=torch.long, device=policy.device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=policy.device)

    opt = optim.Adam(policy.pi.parameters(), lr=args.lr)
    train_hist, val_hist = [], []

    # PINN physics regularizer: add the Helmholtz residual to the imitation MSE
    # so the field-consistency prior stays active. No-op for cnn/mlp (no method)
    # or when the weight is 0.
    has_phys = hasattr(policy.pi, "helmholtz_residual_loss")
    phys_w = float(getattr(args, "physics_loss_weight", 0.0))

    for epoch in range(args.epochs):
        policy.pi.train()
        perm = rng.permutation(len(train_idx))
        shuffled = train_idx[perm]
        epoch_loss, n_batches = 0.0, 0
        for bs in range(0, len(shuffled), args.batch_size):
            bi = shuffled[bs:bs + args.batch_size]
            bi_t = torch.as_tensor(bi, dtype=torch.long, device=policy.device)
            pred = policy.pi(states_t[bi_t], goals_t[bi_t])
            loss = F.mse_loss(pred, actions_t[bi_t])
            if has_phys and phys_w > 0.0:
                loss = loss + phys_w * policy.pi.helmholtz_residual_loss(
                    states_t[bi_t], goals_t[bi_t])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_hist.append(epoch_loss / max(n_batches, 1))

        policy.pi.eval()
        with torch.no_grad():
            vi_t = torch.as_tensor(val_idx, dtype=torch.long, device=policy.device)
            v = float(F.mse_loss(policy.pi(states_t[vi_t], goals_t[vi_t]),
                                 actions_t[vi_t]))
        val_hist.append(v)

        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"  epoch {epoch + 1:>3}/{args.epochs}:  "
                  f"train={train_hist[-1]:.4e}  val={v:.4e}")

    return train_hist, val_hist, len(train_idx), n_val


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading buffer: {args.buffer}")
    with open(args.buffer, "rb") as f:
        buf = pickle.load(f)
    print(f"  {len(buf):,} transitions")

    memory_bank = load_memory_bank(args.memory_bank)
    print(f"Memory bank angles: {sorted(memory_bank.keys())}")

    # Phase 1's training order: read from summary.json's results sequence
    # (the order train_all_angles processed angles, which is also the order
    # they were appended to the buffer).
    summary_path = args.memory_bank / "summary.json"
    if summary_path.exists():
        import json
        summary = json.loads(summary_path.read_text())
        training_order = [int(r["target_idx"]) for r in summary["results"]]
        print(f"Training order from summary.json: {training_order}")
    else:
        training_order = sorted(memory_bank.keys())
        print(f"No summary.json; assuming sorted order: {training_order}")

    print("\nExtracting trajectories...")
    trajectories = extract_trajectories(buf, memory_bank, training_order)

    print("\nBuilding dataset...")
    t0 = time.time()
    states, goals, actions = build_dataset(trajectories, args, rng)
    print(f"  {len(states):,} (state, goal, target_action) samples ({time.time() - t0:.1f}s)")
    print(f"  action stats: |target|_L2 mean={np.linalg.norm(actions.reshape(len(actions), -1), axis=1).mean():.3f}  "
          f"per-elem max={np.abs(actions).max():.3f}  per-elem mean={np.abs(actions).mean():.4f}")

    # Build policy
    state_shape = states.shape[1:]
    pcfg = ESPolicyConfig(
        policy_arch=args.policy_arch,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        tanh_output=not args.no_tanh,
        tanh_output_scale=args.tanh_output_scale,
        n_goals=args.n_goals,
        physics_loss_weight=args.physics_loss_weight,
        device=args.device,
        seed=args.seed,
    )
    policy = ESPolicy(state_shape=state_shape, config=pcfg)
    n_params = sum(p.numel() for p in policy.pi.parameters())
    phys_note = (f"  physics_w={args.physics_loss_weight}"
                 if hasattr(policy.pi, "helmholtz_residual_loss") and args.physics_loss_weight > 0
                 else "")
    print(f"\nPolicy[{args.policy_arch}]: state_shape={state_shape}  "
          f"hidden={args.hidden_dim}×{args.n_hidden_layers}  params={n_params:,}  "
          f"tanh_scale={args.tanh_output_scale}{phys_note}")

    print(f"\nTraining {args.epochs} epochs over {len(states):,} samples...")
    t0 = time.time()
    train_hist, val_hist, n_train, n_val = train(policy, states, goals, actions, args, rng)
    print(f"  finished in {time.time() - t0:.1f}s")
    print(f"  train_loss: first→last  {train_hist[0]:.4e} → {train_hist[-1]:.4e}")
    print(f"  val_loss:   first→last  {val_hist[0]:.4e} → {val_hist[-1]:.4e}")
    print(f"  n_train={n_train:,}  n_val={n_val:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.out)
    np.savez(args.out.with_suffix(".loss.npz"),
             train_loss=np.array(train_hist, dtype=np.float32),
             val_loss=np.array(val_hist, dtype=np.float32),
             n_train=n_train, n_val=n_val)
    print(f"\nPolicy → {args.out.resolve()}")


if __name__ == "__main__":
    main()
