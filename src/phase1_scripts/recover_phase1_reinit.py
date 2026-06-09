"""Recover Phase 1 reinit results from Modal Volume checkpoints.

The local `modal run` entrypoint died with a Modal client `UnimplementedError`
when fetching large per-goal results back. Each worker's full state — including
eps_buffer, P_buffer, best_eps, history — was checkpointed to the Modal Volume
at /buffer/<run_id>/target_NN/state.pkl. This script downloads those, does the
post-processing (aggregate to .npz, write per-goal eps_star.npy, render PNGs)
that the entrypoint would have done.

Usage:
    python recover_phase1_reinit.py --run-id phase1-reinit-20260602-052948
"""

import argparse
import io
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

VOLUME_NAME = os.environ.get("PHASE1_REINIT_VOLUME",
                             "cs224r-phase1-reinit-buffer")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True, type=str)
    p.add_argument("--memory-bank", type=Path,
                   default=Path("phase1-uniform-init-output"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("phase1-reinit-output"))
    p.add_argument("--training-data-out", type=Path,
                   default=Path("phase1-reinit-output/all_eps_P.npz"))
    p.add_argument("--render-dir", type=Path,
                   default=Path("checkpoint_output/phase1-reinit"))
    p.add_argument("--anchors-map", type=str, default="",
                   help="Explicit 'target:prev,...' anchors used by the run "
                        "(must match train_phase1_reinit_modal --anchors-map), "
                        "e.g. '0:3,9:6,12:15,18:24,21:27'. Overrides far-anchor "
                        "for the rendered 'before' config and prev_init label.")
    p.add_argument("--tmp-dir", type=Path, default=Path("/tmp/phase1_reinit_dl"))
    return p.parse_args()


def download_state(run_id, target, tmp_dir):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / f"state_target_{target:02d}.pkl"
    src = f"{run_id}/target_{target:02d}/state.pkl"
    print(f"  downloading {src} → {dst}")
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, src, str(dst), "--force"],
        check=True, capture_output=True,
    )
    return dst


def load_memory_bank(d):
    bank = {}
    for td in sorted(d.glob("target_*")):
        if not td.is_dir(): continue
        try:
            idx = int(td.name.split("_")[-1])
        except ValueError: continue
        p = td / "eps_star.npy"
        if p.exists():
            bank[idx] = np.load(p).astype(np.float32)
    return bank


def main():
    args = parse_args()
    print(f"Recovering run: {args.run_id}")
    print(f"From volume: {VOLUME_NAME}")

    bank_orig = load_memory_bank(args.memory_bank)
    anchors = sorted(bank_orig.keys())
    print(f"Memory bank anchors: {anchors}\n")

    custom_anchor = {}
    if args.anchors_map:
        for pair in args.anchors_map.split(","):
            t, pr = pair.split(":")
            custom_anchor[int(t)] = int(pr)
        print(f"Custom anchor overrides (target -> prev): {custom_anchor}\n")

    # Download state.pkl per target; skip ones missing on the Volume
    # (Run A only ran the 8 failed goals; full-bank runs have all 10).
    states = {}
    for target in anchors:
        try:
            dst = download_state(args.run_id, target, args.tmp_dir)
        except subprocess.CalledProcessError as e:
            print(f"  target {target:>2}: NOT ON VOLUME (skipping)  [{e}]")
            continue
        state = pickle.loads(dst.read_bytes())
        states[target] = state
        complete = state.get("complete", False)
        n_buffer = len(state.get("eps_buffer", []))
        n_iter = state.get("iter", 0)
        result_present = "result" in state
        best_tf = state.get("best_tf", 0.0)
        print(f"  target {target:>2}: complete={complete}  iter={n_iter}  "
              f"buffer={n_buffer}  best_tf={best_tf:.4f}  result_in_state={result_present}")

    print()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receiver_indices = None
    all_eps_chunks = []
    all_P_chunks = []
    summary_rows = []

    # Only process targets actually present (Run A is partial).
    targets_present = sorted(states.keys())
    for target in targets_present:
        s = states[target]
        if not s.get("complete", False):
            print(f"  WARN: target {target} not marked complete; using partial state.")

        # Extract the result. The state dict has fields directly (and possibly
        # a nested "result" copy if complete=True).
        if "result" in s:
            r = s["result"]
        else:
            # Reconstruct from raw state fields
            r = {
                "target": target,
                "best_eps": s["best_eps"],
                "best_target_frac": s["best_tf"],
                "final_mean_eps": s["eps"],
                "final_mean_target_frac": None,
                "history": s["history"],
                "eps_buffer": np.stack(s["eps_buffer"]).astype(np.float32),
                "P_buffer": np.stack(s["P_buffer"]).astype(np.float64),
                "receiver_indices": s.get("receiver_indices"),
                "elapsed_s": None,
            }

        td = args.out_dir / f"target_{target:02d}"
        td.mkdir(parents=True, exist_ok=True)
        np.save(td / "eps_star.npy", r["best_eps"])
        np.save(td / "eps_mean_final.npy", r["final_mean_eps"])
        (td / "history.json").write_text(json.dumps({
            "target_idx": target,
            "best_target_frac": float(r["best_target_frac"]),
            "final_mean_target_frac": (
                float(r["final_mean_target_frac"])
                if r["final_mean_target_frac"] is not None else None),
            "iterations": len(r["history"]),
            "elapsed_s": r.get("elapsed_s"),
            "history": r["history"],
        }, indent=2))

        # Aggregate buffers
        eps_b = np.asarray(r["eps_buffer"])
        P_b = np.asarray(r["P_buffer"])
        all_eps_chunks.append(eps_b)
        all_P_chunks.append(P_b)
        if receiver_indices is None and r.get("receiver_indices") is not None:
            receiver_indices = np.asarray(r["receiver_indices"])

        # Original Phase 1 best_target_frac
        meta_path = args.memory_bank / f"target_{target:02d}" / "metadata.json"
        tf_orig = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                tf_orig = meta.get("best_ever_target_fraction") \
                          or meta.get("best_target_fraction")
            except Exception:
                pass
        delta = (float(r["best_target_frac"]) - tf_orig) if tf_orig is not None else None
        verdict = "?" if delta is None else (
            "≈ same" if abs(delta) < 0.01 else
            ("✓ better" if delta > 0 else "✗ worse"))
        summary_rows.append({
            "target": target, "tf_original": tf_orig,
            "tf_reinit": float(r["best_target_frac"]),
            "delta": delta, "verdict": verdict,
        })

    all_eps = np.concatenate(all_eps_chunks, axis=0)
    all_P = np.concatenate(all_P_chunks, axis=0)
    args.training_data_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.training_data_out,
             eps=all_eps, P=all_P, receiver_indices=receiver_indices)
    print(f"\nSaved {len(all_eps):,} (ε, P) pairs → {args.training_data_out}")
    print(f"  eps: {all_eps.shape}  P: {all_P.shape}")

    # Comparison table
    print()
    print(f"{'target':>6}  {'orig_tf':>7}  {'reinit_tf':>9}  {'Δ':>7}    verdict")
    print("-" * 60)
    n_better = n_worse = n_same = 0
    for row in summary_rows:
        orig = f"{row['tf_original']:.4f}" if row['tf_original'] is not None else "?"
        delta = f"{row['delta']:+.4f}" if row['delta'] is not None else "?"
        v = row["verdict"]
        if v == "✓ better": n_better += 1
        elif v == "✗ worse": n_worse += 1
        elif v == "≈ same": n_same += 1
        print(f"  {row['target']:>2}    {orig:>7}   {row['tf_reinit']:.4f}    "
              f"{delta:>7}    {v}")
    print("-" * 60)
    print(f"  better: {n_better}    worse: {n_worse}    same: {n_same}")

    (args.out_dir / "summary.json").write_text(json.dumps({
        "run_id": args.run_id,
        "total_fdfd_samples": int(len(all_eps)),
        "per_goal": summary_rows,
    }, indent=2))

    # Render PNGs
    print(f"\nRendering PNGs → {args.render_dir}")
    args.render_dir.mkdir(parents=True, exist_ok=True)
    from render_phase2_ckpt import build_pm_env, render_2x2_png
    env = build_pm_env()

    def far_anchor(target):
        opp = (target + 15) % 30
        def ring_dist(a):
            d = abs(a - opp)
            return min(d, 30 - d)
        return min(anchors, key=ring_dist)

    render_summary = []
    _pick_anchor = lambda tgt: custom_anchor.get(tgt, far_anchor(tgt))
    for row in summary_rows:
        target = row["target"]
        prev = _pick_anchor(target)
        eps_init = bank_orig[prev]
        eps_final = np.load(args.out_dir / f"target_{target:02d}" / "eps_star.npy")
        tf_orig_disp = (f"{row['tf_original']:.3f}"
                        if row['tf_original'] is not None else "?")
        suffix = (f"[reinit] from ε*(prev={prev})  "
                  f"tf orig={tf_orig_disp}  "
                  f"reinit best={row['tf_reinit']:.3f}")
        save_path = args.render_dir / f"target_{target:02d}.png"
        P_i, P_f = render_2x2_png(env, eps_init, eps_final, target,
                                  save_path, title_suffix=suffix)
        ti = float(P_i[target] / max(P_i.sum(), 1e-9))
        tf = float(P_f[target] / max(P_f.sum(), 1e-9))
        print(f"  target={target:>2}  tf {ti:.4f} → {tf:.4f}  → {save_path.name}")
        render_summary.append({
            "target": target, "prev_init": prev,
            "target_frac_init": ti, "target_frac_final": tf,
            "png": str(save_path),
        })
    (args.render_dir / "viz_summary.json").write_text(json.dumps({
        "method": "phase1_reinit_far_anchor",
        "run_id": args.run_id,
        "results": render_summary,
    }, indent=2))
    print(f"\nWrote {len(render_summary)} PNGs to {args.render_dir.resolve()}")


if __name__ == "__main__":
    main()
