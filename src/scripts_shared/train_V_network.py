"""Train an FDFD surrogate M(state) → P[30] on Phase 1 mean-state data.

Consumes the output of compute_P_per_state_sampled.py (an .npz with
{eps: (N, 10, 10), P: (N, 30), receiver_indices: (30,)}) and trains a small
MLP M_φ(ε) → P[30] that predicts per-receiver power without an FDFD call.

Once trained, M can be used for:
  - Cheap planning at runtime: argmax_a r(M(ε + a), goal)
  - Critic in Phase 2 ES rollouts (variance-reduction signal)
  - Model-based bootstrap (your earlier proposal)

We use log1p(P) targets internally to tame the ~8-order-of-magnitude dynamic
range of P (0 → 4e5). The saved checkpoint includes the per-receiver mean/std
of log1p(P) so inference can de-normalize back to P.

Usage:
    python train_V_network.py \\
        --data phase1-uniform-init-output/mean_states_P.npz \\
        --hidden-dim 128 --n-hidden-layers 3 \\
        --out pretrain/M_fdfd_surrogate.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def parse_args():
    p = argparse.ArgumentParser(description="Train M(ε) → P[30] FDFD surrogate")
    p.add_argument("--data", type=Path,
                   default=Path("phase1-uniform-init-output/mean_states_P.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("pretrain/M_fdfd_surrogate.pt"))

    # Architecture
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-hidden-layers", type=int, default=3)

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


class PNetwork(nn.Module):
    """M_φ(ε) → log1p(P) ∈ R^{n_receivers}, then de-normalized at inference.

    Output is in *normalized log1p(P) space* during training; the caller
    converts back to raw P via `expm1(out * std + mean)`.
    """

    def __init__(self, state_shape, n_receivers=30,
                 hidden_dim=128, n_hidden_layers=3):
        super().__init__()
        N_x, N_y = state_shape
        layers, prev = [], N_x * N_y
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, n_receivers))
        self.net = nn.Sequential(*layers)
        self.state_shape = state_shape
        self.n_receivers = n_receivers

    def forward(self, eps):
        B = eps.shape[0]
        return self.net(eps.reshape(B, -1))


def predict_P(model: PNetwork, eps: torch.Tensor,
              log_mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Forward + de-normalize → raw P[30]. eps: (B, 10, 10) → out: (B, 30)."""
    with torch.no_grad():
        normed = model(eps)
        log_P = normed * log_std + log_mean
        return torch.expm1(log_P).clamp(min=0.0)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.data}")
    data = np.load(args.data)
    eps, P = data["eps"], data["P"]
    rx_idx = data["receiver_indices"]
    n_receivers = int(P.shape[1])
    print(f"  eps: {eps.shape}  P: {P.shape}  n_receivers: {n_receivers}")
    print(f"  P stats (raw): mean={P.mean():.2e}  max={P.max():.2e}  "
          f"min={P.min():.2e}")

    # log1p compression. P can span 8 orders of magnitude; log keeps the
    # per-receiver MSE balanced across magnitudes.
    log_P = np.log1p(P).astype(np.float32)
    log_mean = log_P.mean(axis=0)         # (30,) per-receiver mean
    log_std = log_P.std(axis=0) + 1e-9    # (30,) per-receiver std
    log_P_normed = (log_P - log_mean) / log_std
    print(f"  log1p(P) per-receiver mean range: [{log_mean.min():.3f}, "
          f"{log_mean.max():.3f}]")
    print(f"  log1p(P) per-receiver std  range: [{log_std.min():.3f}, "
          f"{log_std.max():.3f}]")

    # Train / val split
    N = len(eps)
    idx = rng.permutation(N)
    n_val = max(1, int(N * args.val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    eps_t = torch.as_tensor(eps, dtype=torch.float32, device=args.device)
    target_t = torch.as_tensor(log_P_normed, dtype=torch.float32, device=args.device)

    state_shape = tuple(eps.shape[1:])
    model = PNetwork(state_shape, n_receivers,
                     hidden_dim=args.hidden_dim,
                     n_hidden_layers=args.n_hidden_layers).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nP network: state_shape={state_shape}  out={n_receivers}  "
          f"hidden={args.hidden_dim}×{args.n_hidden_layers}  "
          f"params={n_params:,}")

    opt = optim.Adam(model.parameters(),
                     lr=args.lr, weight_decay=args.weight_decay)

    # Pre-tensors for normalization at val time
    log_mean_t = torch.as_tensor(log_mean, dtype=torch.float32, device=args.device)
    log_std_t = torch.as_tensor(log_std, dtype=torch.float32, device=args.device)
    P_raw_t = torch.as_tensor(P.astype(np.float32), dtype=torch.float32,
                              device=args.device)

    train_hist, val_hist = [], []
    print(f"\nTraining {args.epochs} epochs (lr={args.lr}, batch={args.batch_size}, "
          f"wd={args.weight_decay})...")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(train_idx))
        shuffled = train_idx[perm]
        ep_loss, n_batches = 0.0, 0
        for bs in range(0, len(shuffled), args.batch_size):
            bi = torch.as_tensor(shuffled[bs:bs + args.batch_size],
                                 dtype=torch.long, device=args.device)
            pred = model(eps_t[bi])
            loss = F.mse_loss(pred, target_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss)
            n_batches += 1
        train_hist.append(ep_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            vi = torch.as_tensor(val_idx, dtype=torch.long, device=args.device)
            val_pred_normed = model(eps_t[vi])
            val_loss_normed = float(F.mse_loss(val_pred_normed, target_t[vi]))

            # Also report on raw-P scale: MAE and R²
            val_pred_log = val_pred_normed * log_std_t + log_mean_t
            val_pred_raw = torch.expm1(val_pred_log).clamp(min=0.0)
            val_true_raw = P_raw_t[vi]
            mae_raw = float((val_pred_raw - val_true_raw).abs().mean())
            ss_res = float(((val_pred_raw - val_true_raw) ** 2).sum())
            ss_tot = float(((val_true_raw - val_true_raw.mean()) ** 2).sum())
            r2_raw = 1.0 - ss_res / max(ss_tot, 1e-9)
        val_hist.append(val_loss_normed)

        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"  epoch {epoch + 1:>4}/{args.epochs}:  "
                  f"train(norm)={train_hist[-1]:.4e}  "
                  f"val(norm)={val_loss_normed:.4e}  "
                  f"val_MAE(raw)={mae_raw:.2e}  val_R²(raw)={r2_raw:.4f}")

    elapsed = time.time() - t0
    print(f"  finished in {elapsed:.1f}s")

    # Final detailed metrics
    model.eval()
    with torch.no_grad():
        vi = torch.as_tensor(val_idx, dtype=torch.long, device=args.device)
        val_pred_normed = model(eps_t[vi])
        val_pred_log = val_pred_normed * log_std_t + log_mean_t
        val_pred_raw = torch.expm1(val_pred_log).clamp(min=0.0).cpu().numpy()
    val_true_raw = P[val_idx]

    # Per-receiver Pearson correlations
    corrs = np.array([
        float(np.corrcoef(val_pred_raw[:, j], val_true_raw[:, j])[0, 1])
        for j in range(n_receivers)
    ])
    log_corrs = np.array([
        float(np.corrcoef(np.log1p(val_pred_raw[:, j]),
                          np.log1p(val_true_raw[:, j]))[0, 1])
        for j in range(n_receivers)
    ])

    print(f"\nPer-receiver validation correlation (raw P, log1p(P)):")
    print(f"  Pearson r (raw):  mean={corrs.mean():.4f}  "
          f"min={corrs.min():.4f}  max={corrs.max():.4f}")
    print(f"  Pearson r (log):  mean={log_corrs.mean():.4f}  "
          f"min={log_corrs.min():.4f}  max={log_corrs.max():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_shape": state_shape,
        "n_receivers": n_receivers,
        "hidden_dim": args.hidden_dim,
        "n_hidden_layers": args.n_hidden_layers,
        "log_mean": log_mean,         # (30,) per-receiver
        "log_std": log_std,           # (30,)
        "receiver_indices": rx_idx,   # (30,) for reference
    }, args.out)
    np.savez(args.out.with_suffix(".loss.npz"),
             train_loss=np.array(train_hist, dtype=np.float32),
             val_loss=np.array(val_hist, dtype=np.float32))

    print(f"\nSaved M(ε) → P[30] surrogate → {args.out.resolve()}")
    print(f"\nInference recipe:")
    print(f"  ckpt = torch.load('{args.out}', weights_only=False)")
    print(f"  model = PNetwork(ckpt['state_shape'], ckpt['n_receivers'],")
    print(f"                   ckpt['hidden_dim'], ckpt['n_hidden_layers'])")
    print(f"  model.load_state_dict(ckpt['model_state_dict'])")
    print(f"  log_mean = torch.as_tensor(ckpt['log_mean'])")
    print(f"  log_std  = torch.as_tensor(ckpt['log_std'])")
    print(f"  P_pred = predict_P(model, state_batch, log_mean, log_std)")


if __name__ == "__main__":
    main()
