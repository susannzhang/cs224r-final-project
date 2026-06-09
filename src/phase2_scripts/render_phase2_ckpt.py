"""Render Phase 2 retargeting before/after PNGs from a trained policy.

Usage:
    python render_phase2_ckpt.py --launch-id phase2-parallel-20260601-123456 \\
        --memory-bank phase1-uniform-init-output

    # Override the policy source (skip Modal Volume download)
    python render_phase2_ckpt.py --launch-id ... \\
        --policy-path phase2_parallel_output/phase2-.../policy_phase2.pt \\
        --memory-bank phase1-uniform-init-output

For each angle in the run's goal_indices, picks a θ_prev from converged_indices
(the Phase 1 angles), rolls the trained policy out for T steps starting at
ε*(θ_prev), and renders a 2×2 figure (initial ε / initial field / final ε /
final field) — same layout as Phase 1's before_after.png.

Output:
    checkpoint_output/phase2/<launch_id>/target_<NN>.png   (one per goal)
"""

import argparse
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent          # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                       # cs153 repo root (geometry, simulation)
for _p in (PROJECT_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from algorithms.agents.es_agent import apply_eps_to_canvas
from algorithms.policies.es_policy import ESPolicy
from geometry import (create_design_region, create_environment, create_grid,
                      create_receiver, create_source)
from simulation import initialize_environment, simulate_ez_fields_per_source

VOLUME_NAME = os.environ.get(
    "PHASE2_VOLUME_NAME", "cs224r-phase2-parallel-buffer")


def parse_args():
    p = argparse.ArgumentParser(description="Render Phase 2 retargeting PNGs")
    p.add_argument("--launch-id", type=str, required=True,
                   help="Phase 2 run launch_id (e.g. phase2-parallel-20260601-123456).")
    p.add_argument("--memory-bank", type=Path, required=True,
                   help="Phase 1 output dir with target_<NN>/eps_star.npy.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: checkpoint_output/phase2/<launch_id>/")
    p.add_argument("--policy-path", type=Path, default=None,
                   help="Local policy_phase2.pt path. If omitted, downloads "
                        "from Modal Volume.")
    p.add_argument("--goal-indices", type=str, default=None,
                   help='Goals to render, comma-separated. If omitted, '
                        'reads from <out_dir>/history.json.')
    p.add_argument("--T", type=int, default=20,
                   help="Max policy rollout steps for the 'final' state. "
                        "Use the same T as the training run (or larger).")
    p.add_argument("--eta", type=float, default=1e-2,
                   help="Early-terminate rollout when target_frac ≥ 1−η.")
    p.add_argument("--theta-prev", type=str, default="nearest",
                   choices=["nearest", "fixed"],
                   help="How to pick θ_prev: 'nearest' Phase 1 angle by index, "
                        "or 'fixed' to use --theta-prev-fixed.")
    p.add_argument("--theta-prev-fixed", type=int, default=0,
                   help="θ_prev when --theta-prev=fixed.")
    return p.parse_args()


def download_from_volume(launch_id: str, filename: str, dest: Path) -> bool:
    """`modal volume get <vol> <launch_id>/<filename> <dest> --force`.
    Returns True on success, False if the file isn't in the Volume."""
    src = f"{launch_id}/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, src, str(dest), "--force"],
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def load_memory_bank(memory_bank_dir: Path) -> dict:
    """Read every target_<NN>/eps_star.npy in the directory."""
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
            memory_bank[idx] = np.load(path)
    if not memory_bank:
        raise FileNotFoundError(f"No memory bank found under {memory_bank_dir}")
    return memory_bank


def build_pm_env():
    """Standard pm_setup env — must mirror what training used."""
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


def rollout_policy(policy, env, eps_0, goal, T, eta):
    """Roll out the trained policy from eps_0 for up to T steps under `goal`.

    Fast version: no in-loop FDFD (the early-termination check is dropped
    since training logs show mean_rollout_length=T always, i.e. early-term
    never fires). Just run T policy.predict + clip steps; render_2x2_png
    does the FDFD on the resulting state.
    """
    del env, eta  # kept in signature for API compatibility
    eps = eps_0.astype(np.float32, copy=True)
    for _ in range(T):
        delta = policy.predict(eps, int(goal)).astype(np.float32)
        eps = np.clip(eps + delta, -1.0, 1.0).astype(np.float32)
    return eps


def render_2x2_png(env, eps_initial, eps_final, target_idx, save_path,
                   title_suffix=""):
    """2×2: initial permittivity / initial intensity / final ε / final intensity.
    Layout mirrors Phase 1's _render_before_after_png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Initial
    apply_eps_to_canvas(env, eps_initial)
    canvas_initial = env.design_region._canvas.copy()
    ez_initial = sum(simulate_ez_fields_per_source(env).values())
    intensity_initial = np.abs(ez_initial) ** 2
    P_initial = np.array([
        float(np.sum(intensity_initial * r._mask)) for r in env.receivers
    ])

    # Final
    apply_eps_to_canvas(env, eps_final)
    canvas_final = env.design_region._canvas.copy()
    ez_final = sum(simulate_ez_fields_per_source(env).values())
    intensity_final = np.abs(ez_final) ** 2
    P_final = np.array([
        float(np.sum(intensity_final * r._mask)) for r in env.receivers
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    def _draw_perm(ax, canvas, title):
        clipped = np.clip(canvas, 0, 10)
        im = ax.imshow(clipped, cmap="plasma", origin="lower", vmin=0, vmax=10)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, label="ε")

    def _draw_int(ax, intensity, canvas, title, target_receiver):
        vmax = np.percentile(intensity, 98)
        im = ax.imshow(intensity, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
        ax.contour(canvas, [3.0, 5e5], colors="white", alpha=0.5, linewidths=0.6)
        ax.contour(target_receiver._mask, [0.5], colors="cyan", linewidths=1.5)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, label="|E_z|²")

    target_receiver = env.receivers[target_idx]
    _draw_perm(axes[0, 0], canvas_initial,
               "Initial permittivity (warm-start ε*)")
    _draw_int(axes[0, 1], intensity_initial, canvas_initial,
              "Initial field intensity", target_receiver)
    _draw_perm(axes[1, 0], canvas_final,
               f"Final permittivity after policy rollout (target={target_idx})")
    _draw_int(axes[1, 1], intensity_final, canvas_final,
              "Final field intensity", target_receiver)
    fig.suptitle(
        f"Phase 2 retargeting → target {target_idx} {title_suffix}",
        fontsize=13)

    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return P_initial, P_final


def pick_theta_prev(target_idx, converged_indices, strategy, fixed_idx):
    converged = [g for g in converged_indices if g != target_idx]
    if not converged:
        return None
    if strategy == "fixed":
        return fixed_idx if fixed_idx in converged else converged[0]
    # nearest by index distance
    return min(converged, key=lambda g: abs(g - target_idx))


def main():
    args = parse_args()

    # ----- Resolve output dir + policy path -----
    out_dir = (args.out_dir if args.out_dir is not None
               else REPO_ROOT / "checkpoint_output" / "phase2" / args.launch_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}")

    policy_path = args.policy_path or (out_dir / "policy_phase2.pt")
    if not policy_path.exists():
        print(f"Downloading {args.launch_id}/policy_phase2.pt from "
              f"Modal Volume {VOLUME_NAME}...")
        if not download_from_volume(args.launch_id, "policy_phase2.pt", policy_path):
            raise FileNotFoundError(
                f"policy_phase2.pt not on Volume {VOLUME_NAME} under "
                f"{args.launch_id}/. Did the run finish?")

    # ----- Memory bank -----
    memory_bank = load_memory_bank(args.memory_bank)
    converged_indices = sorted(memory_bank.keys())
    print(f"Memory bank: {converged_indices}")

    # ----- Goal indices: explicit CLI > history.json on Volume -----
    history_path = out_dir / "history.json"
    iter_at_save = None
    if args.goal_indices is not None:
        goal_indices = [int(x) for x in args.goal_indices.split(",")]
        if history_path.exists():
            try:
                iter_at_save = json.loads(history_path.read_text()).get("iter_at_save")
            except Exception:
                iter_at_save = None
    else:
        if not history_path.exists():
            print(f"Downloading {args.launch_id}/history.json...")
            if not download_from_volume(args.launch_id, "history.json",
                                        history_path):
                raise FileNotFoundError(
                    "Need --goal-indices, OR history.json on the Volume.")
        meta = json.loads(history_path.read_text())
        goal_indices = meta.get("goal_indices")
        iter_at_save = meta.get("iter_at_save")
        if goal_indices is None:
            raise ValueError("history.json missing goal_indices field.")
    print(f"goal_indices: {goal_indices}")
    if iter_at_save is not None:
        print(f"iter_at_save: {iter_at_save}")

    # ----- Load policy -----
    policy = ESPolicy.load(policy_path)
    print(f"Loaded policy: state_shape={policy.state_shape}  "
          f"n_goals={policy.config.n_goals}  hidden={policy.config.hidden_dim}")

    # ----- Build env once -----
    print("Building pm_setup env (10×10 grid, 30 receivers)...")
    env = build_pm_env()

    # ----- For each goal, rollout + render -----
    iter_tag = f"iter {iter_at_save}" if iter_at_save is not None else "iter ?"
    title_suffix = f"[{iter_tag}] (T≤{args.T}, η={args.eta})"
    summary = []
    for target_idx in goal_indices:
        theta_prev = pick_theta_prev(target_idx, converged_indices,
                                     args.theta_prev, args.theta_prev_fixed)
        if theta_prev is None:
            print(f"  ✗ target {target_idx:02d}: no theta_prev candidate, skipping")
            continue

        eps_0 = memory_bank[theta_prev]
        print(f"  target {target_idx:02d}: warm-start ε*(θ_prev={theta_prev}) → "
              f"rolling out T≤{args.T} ...")
        eps_final = rollout_policy(policy, env, eps_0, target_idx,
                                   args.T, args.eta)
        save_path = out_dir / f"target_{target_idx:02d}.png"
        P_initial, P_final = render_2x2_png(
            env, eps_0, eps_final, target_idx, save_path,
            title_suffix=f"{title_suffix} from ε*(θ_prev={theta_prev})")
        ti = float(P_initial.sum())
        tf = float(P_final.sum())
        fi = P_initial[target_idx] / ti if ti > 0 else 0.0
        ff = P_final[target_idx] / tf if tf > 0 else 0.0
        print(f"    target_frac: {fi:.3f} → {ff:.3f}    → {save_path}")
        summary.append({
            "target_idx": target_idx,
            "theta_prev": theta_prev,
            "target_frac_initial": float(fi),
            "target_frac_final": float(ff),
            "png": str(save_path),
        })

    # ----- Aggregate summary -----
    summary_path = out_dir / "viz_summary.json"
    summary_path.write_text(json.dumps({
        "launch_id": args.launch_id,
        "T": args.T,
        "eta": args.eta,
        "theta_prev_strategy": args.theta_prev,
        "results": summary,
    }, indent=2))
    print()
    print(f"Wrote {len(summary)} PNGs + viz_summary.json to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
