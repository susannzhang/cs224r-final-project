"""Deploy the distilled closed-loop π_φ on heldout angles and measure how
quickly it reaches target_frac thresholds.

Per goal, runs the closed-loop rollout
    ε_0 = InterpolateAnchors(M, θ)
    ε_{t+1} = clip(ε_t + π_φ(ε_t, θ), [-1, 1])   for t = 0, ..., T-1
and FDFDs every step (via Modal's deployed fdfd_one) to compute target_frac
at each step. Reports the first step at which target_frac crosses given
thresholds, the full tf curve, and renders before/after PNGs per goal.

This is a *pure deployment* eval — no ES, no M filter, no FDFD in the
policy's inner loop. The only Modal work is the per-step FDFD verification
so we can measure the policy's convergence speed.

Usage:
    python deploy_phase2_distilled.py \\
        --policy pretrain/policy_distilled_v2.pt \\
        --memory-bank phase1-uniform-init-output \\
        --goals 2,5,8,11,14,17,20,23,26,29 \\
        --T 50 --thresholds 0.2,0.6 \\
        --out-dir checkpoint_output/distill_deploy
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import modal
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PHASE2_SS_APP = os.environ.get("PHASE2_SS_APP_NAME",
                               "cs224r-phase2-state-space")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, type=Path,
                   help="Distilled ESPolicy checkpoint "
                        "(e.g. pretrain/policy_distilled_v2.pt).")
    p.add_argument("--memory-bank", required=True, action="append", type=Path,
                   dest="memory_banks",
                   help="Directory with target_<NN>/eps_star.npy and/or "
                        "goal_<NN>/eps_star.npy. Pass multiple times to "
                        "union into a single 'seen ε*' bank.")
    p.add_argument("--goals", required=True, type=str,
                   help='Comma-separated goal indices, e.g. "2,5,8,11,...".')
    p.add_argument("--init-mode", default="interp",
                   choices=["interp", "nearest", "random_seen"],
                   help='Warm-start ε per goal. "interp" = linear interp '
                        'between two nearest seen anchors; "nearest" = ε* of '
                        'the closest seen anchor verbatim; "random_seen" = '
                        'ε* of a RANDOM seen angle (the policy must retarget '
                        'from there). Default: interp.')
    p.add_argument("--init-seed", type=int, default=0,
                   help="Seed for --init-mode=random_seen.")
    p.add_argument("--T", type=int, default=50,
                   help="Max closed-loop steps per goal.")
    p.add_argument("--thresholds", type=str, default="0.2,0.6",
                   help="Comma-separated tf thresholds to track.")
    p.add_argument("--early-stop-all", type=float, default=None,
                   help="Stop entire run once every goal hits this tf "
                        "threshold. Default: never.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("checkpoint_output/distill_deploy"))
    # --- wandb ----------------------------------------------------------
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str,
                   default="cs224r-phase2-state-space",
                   help="Reuse the state-space project so distill-deploy runs "
                        "cluster with the solver runs they correspond to.")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-group", type=str, default="",
                   help="Group tag (default: auto from --policy stem + "
                        "current timestamp).")
    return p.parse_args()


def load_memory_bank(dirs) -> dict:
    """Union ε* files from one or more directories.

    Accepts both `target_<NN>/eps_star.npy` (Phase 1 layout) and
    `goal_<NN>/eps_star.npy` (state-space-ES Modal layout). On collisions,
    later directories override earlier ones.
    """
    bank = {}
    if not isinstance(dirs, (list, tuple)):
        dirs = [dirs]
    for d in dirs:
        d = Path(d)
        for prefix in ("target_", "goal_"):
            for td in sorted(d.glob(f"{prefix}*")):
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
        raise FileNotFoundError(f"No eps_star.npy found under {dirs}")
    return bank


def first_hit(curve, threshold):
    """First index at which curve[i] >= threshold, or None."""
    for i, v in enumerate(curve):
        if v >= threshold:
            return i
    return None


def main():
    args = parse_args()
    from algorithms.policies.es_policy import ESPolicy
    from algorithms.policies.es_state_space_policy import (
        interpolate_anchors, nearest_anchor,
    )

    fdfd_one = modal.Function.from_name(PHASE2_SS_APP, "fdfd_one")

    goals = [int(g) for g in args.goals.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    # --- wandb setup: single run; per-goal scalars logged as namespaced keys
    # `goal_<NN>/target_frac`. Multi-run-per-goal hits wandb's `reinit=True`
    # finishing behavior, which silently drops late .log() calls on
    # earlier-init'd runs.
    wandb = None
    run = None
    if not args.no_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            print("wandb not installed; running without logging.")
    if wandb is not None:
        from datetime import datetime
        launch_id = (args.wandb_group
                     or f"{args.policy.stem}-{datetime.now():%Y%m%d-%H%M%S}")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=f"deploy-{launch_id}",
            group=launch_id,
            tags=["deploy", "distilled-policy",
                  f"init_mode={args.init_mode}"],
            config={
                "policy": str(args.policy),
                "memory_banks": [str(p) for p in args.memory_banks],
                "goals": goals,
                "init_mode": args.init_mode,
                "init_seed": args.init_seed,
                "T": args.T,
                "thresholds": thresholds,
            },
        )

    policy = ESPolicy.load(args.policy)
    memory_bank = load_memory_bank(args.memory_banks)
    print(f"Loaded π_φ from {args.policy}  "
          f"(state_shape={policy.state_shape})")
    print(f"Seen-ε* bank: {len(memory_bank)} entries at "
          f"angles {sorted(memory_bank.keys())}")
    print(f"Eval goals: {goals}")
    print(f"Init mode: {args.init_mode}"
          + (f" (seed={args.init_seed})"
             if args.init_mode == "random_seen" else ""))
    print(f"Thresholds: {thresholds}  T={args.T}\n")

    # Pick warm-start ε per goal.
    rng_init = np.random.default_rng(args.init_seed)
    eps_states = {}
    init_meta = {}                              # which seen angle was used
    for g in goals:
        if args.init_mode == "interp":
            eps_states[g] = interpolate_anchors(memory_bank, g)
            init_meta[g] = {"source": "interp",
                            "neighbors": sorted(memory_bank.keys())}
        elif args.init_mode == "nearest":
            eps_states[g] = nearest_anchor(memory_bank, g)
            init_meta[g] = {"source": "nearest"}
        elif args.init_mode == "random_seen":
            seen = [a for a in memory_bank.keys() if a != g]
            picked = int(rng_init.choice(seen))
            eps_states[g] = memory_bank[picked].astype(np.float32).copy()
            init_meta[g] = {"source": "random_seen", "picked": picked}
            print(f"  goal {g:>2}: init = ε*(seen={picked})")
    eps_initials = {g: eps_states[g].copy() for g in goals}
    tf_curves = {g: [] for g in goals}

    def _fdfd_batch(eps_dict):
        """Dispatch one FDFD per goal in parallel; return {goal: P}."""
        gs = list(eps_dict.keys())
        payloads = [pickle.dumps(eps_dict[g].astype(np.float32)) for g in gs]
        results = list(fdfd_one.map(payloads))
        out = {}
        for g, r in zip(gs, results):
            P, _P_loss = pickle.loads(r)
            out[g] = P
        return out

    # Step 0: FDFD the warm-start states.
    print("Step 0 (init): FDFD on warm-start states...")
    t0 = time.time()
    P_init = _fdfd_batch(eps_states)
    for g in goals:
        tf = float(P_init[g][g] / max(P_init[g].sum(), 1e-9))
        tf_curves[g].append(tf)
    print(f"  step 0 wall: {time.time()-t0:.1f}s  "
          f"mean tf_init={np.mean([tf_curves[g][0] for g in goals]):.4f}")
    print(f"  per-goal tf_init: "
          f"{ {g: round(tf_curves[g][0], 4) for g in goals} }")
    if run is not None:
        log = {"step": 0,
               "mean_target_frac": float(np.mean(
                   [tf_curves[g][0] for g in goals])),
               "min_target_frac": float(min(
                   tf_curves[g][0] for g in goals)),
               "max_target_frac": float(max(
                   tf_curves[g][0] for g in goals)),
               "n_hit_0.2": int(sum(tf_curves[g][0] >= 0.2 for g in goals)),
               "n_hit_0.6": int(sum(tf_curves[g][0] >= 0.6 for g in goals))}
        for g in goals:
            log[f"goal_{g:02d}/target_frac"] = tf_curves[g][0]
            log[f"goal_{g:02d}/best_target_frac"] = tf_curves[g][0]
        run.log(log, step=0)

    # Closed-loop rollout in lockstep.
    overall_t0 = time.time()
    for t in range(1, args.T + 1):
        # Policy step (sub-ms each, no Modal).
        for g in goals:
            delta = policy.predict(eps_states[g], int(g)).astype(np.float32)
            eps_states[g] = np.clip(eps_states[g] + delta,
                                    -1.0, 1.0).astype(np.float32)

        # FDFD verification step (parallel across goals).
        step_t0 = time.time()
        P_step = _fdfd_batch(eps_states)
        for g in goals:
            tf = float(P_step[g][g] / max(P_step[g].sum(), 1e-9))
            tf_curves[g].append(tf)
        step_wall = time.time() - step_t0

        # Log compactly.
        mean_tf = np.mean([tf_curves[g][-1] for g in goals])
        max_tf = max(tf_curves[g][-1] for g in goals)
        min_tf = min(tf_curves[g][-1] for g in goals)
        print(f"  step {t:>3}/{args.T}  wall={step_wall:.1f}s  "
              f"tf mean={mean_tf:.4f}  min={min_tf:.4f}  max={max_tf:.4f}",
              flush=True)

        # wandb: per-goal as namespaced keys + launch-level aggregates.
        if run is not None:
            log = {
                "step": t,
                "mean_target_frac": float(mean_tf),
                "min_target_frac": float(min_tf),
                "max_target_frac": float(max_tf),
                "n_hit_0.2": int(sum(
                    tf_curves[g][-1] >= 0.2 for g in goals)),
                "n_hit_0.6": int(sum(
                    tf_curves[g][-1] >= 0.6 for g in goals)),
                "step_wall_seconds": step_wall,
            }
            for g in goals:
                log[f"goal_{g:02d}/target_frac"] = tf_curves[g][-1]
                log[f"goal_{g:02d}/best_target_frac"] = float(max(tf_curves[g]))
            run.log(log, step=t)

        # Optional early-stop.
        if (args.early_stop_all is not None and
                all(tf_curves[g][-1] >= args.early_stop_all for g in goals)):
            print(f"  EARLY STOP @ step {t}: all goals hit "
                  f"{args.early_stop_all}.")
            break

    overall_wall = time.time() - overall_t0
    print(f"\nTotal rollout wall: {overall_wall/60:.1f} min "
          f"({len(tf_curves[goals[0]])-1} FDFD-verified steps per goal, "
          f"{len(tf_curves[goals[0]])} states including init).\n")

    # --- Aggregate & save ---------------------------------------------
    out = args.out_dir / args.policy.stem
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for g in goals:
        c = tf_curves[g]
        per_goal = {
            "goal": g,
            "init": init_meta[g],
            "tf_init": c[0],
            "tf_final": c[-1],
            "best_tf": float(max(c)),
            "best_tf_step": int(np.argmax(c)),
            "tf_curve": c,
        }
        for th in thresholds:
            per_goal[f"first_step_to_{th}"] = first_hit(c, th)
        results.append(per_goal)

    print(f"{'goal':>4}  {'tf_init':>7}  {'tf_final':>8}  "
          f"{'best_tf':>7}  {'best@':>5}  "
          + "  ".join(f"→{th:.1f}" for th in thresholds))
    print("-" * (40 + 8 * len(thresholds)))
    for r in results:
        thr_str = "  ".join(
            f"{r[f'first_step_to_{th}']:>4}" if r[f'first_step_to_{th}'] is not None
            else "   —"
            for th in thresholds
        )
        print(f"  {r['goal']:>2}    {r['tf_init']:.4f}    {r['tf_final']:.4f}   "
              f"{r['best_tf']:.4f}    {r['best_tf_step']:>3}    {thr_str}")
    print("-" * (40 + 8 * len(thresholds)))

    # Cross-goal aggregates per threshold.
    for th in thresholds:
        hits = [r for r in results if r[f"first_step_to_{th}"] is not None]
        if hits:
            steps = [r[f"first_step_to_{th}"] for r in hits]
            print(f"  hit {th:.1f}: {len(hits)}/{len(results)}  "
                  f"mean steps={np.mean(steps):.1f}  "
                  f"median={int(np.median(steps))}  max={max(steps)}")
        else:
            print(f"  hit {th:.1f}: 0/{len(results)}")

    mean_best = np.mean([r["best_tf"] for r in results])
    mean_final = np.mean([r["tf_final"] for r in results])
    print(f"\n  mean best_tf:  {mean_best:.4f}")
    print(f"  mean final_tf: {mean_final:.4f}")

    # Save.
    (out / "results.json").write_text(json.dumps({
        "policy": str(args.policy.resolve()),
        "memory_banks": [str(p.resolve()) for p in args.memory_banks],
        "goals": goals,
        "T": args.T,
        "thresholds": thresholds,
        "wall_seconds": overall_wall,
        "results": results,
    }, indent=2))
    np.savez(out / "tf_curves.npz",
             goals=np.array(goals),
             tf_curves=np.stack([np.array(tf_curves[g], dtype=np.float64)
                                  for g in goals]))
    print(f"\nResults → {(out / 'results.json').resolve()}")

    # --- Render before/after PNGs per goal ----------------------------
    print(f"\nRendering PNGs → {out}/...")
    png_paths = {}
    try:
        from render_phase2_ckpt import build_pm_env, render_2x2_png
        env = build_pm_env()
        for r in results:
            g = r["goal"]
            tf_curve = r["tf_curve"]
            tf_init, tf_final = tf_curve[0], tf_curve[-1]
            init_label = (f"seen={init_meta[g]['picked']}"
                          if args.init_mode == "random_seen"
                          else args.init_mode)
            suffix = (f"[π_φ deploy] init={init_label}  T={len(tf_curve)-1}  "
                      f"best_tf={r['best_tf']:.3f}  "
                      f"final_tf={tf_final:.3f}")
            png = out / f"goal_{g:02d}.png"
            render_2x2_png(env, eps_initials[g], eps_states[g],
                           g, png, title_suffix=suffix)
            png_paths[g] = png
            print(f"  goal {g:>2}: tf {tf_init:.4f} → {tf_final:.4f}  → {png.name}")
    except Exception as e:
        print(f"  PNG render failed: {e}")

    # --- wandb: final tables + PNG embeds, then finish ----------------
    if run is not None:
        cols = (["goal", "init_source", "init_picked",
                 "tf_init", "tf_final", "best_tf", "best_tf_step"]
                + [f"steps_to_{th}" for th in thresholds])
        data = []
        for r in results:
            row = [r["goal"],
                   r["init"]["source"],
                   (r["init"].get("picked", -1)
                    if r["init"]["source"] == "random_seen" else -1),
                   r["tf_init"], r["tf_final"],
                   r["best_tf"], r["best_tf_step"]]
            for th in thresholds:
                v = r[f"first_step_to_{th}"]
                row.append(-1 if v is None else v)
            data.append(row)
        summary_log = {
            "summary/per_goal_table": wandb.Table(columns=cols, data=data),
            "summary/mean_best_tf": float(mean_best),
            "summary/mean_final_tf": float(mean_final),
            "summary/n_goals": len(results),
            "summary/wall_minutes": overall_wall / 60,
        }
        for th in thresholds:
            hits = [r for r in results
                    if r[f"first_step_to_{th}"] is not None]
            summary_log[f"summary/n_hit_{th}"] = len(hits)
            summary_log[f"summary/frac_hit_{th}"] = len(hits) / len(results)
            if hits:
                steps = [r[f"first_step_to_{th}"] for r in hits]
                summary_log[f"summary/mean_steps_to_{th}"] = float(np.mean(steps))
                summary_log[f"summary/median_steps_to_{th}"] = float(np.median(steps))
                summary_log[f"summary/max_steps_to_{th}"] = int(max(steps))
        # Embed each goal's PNG.
        for g in goals:
            if g in png_paths and png_paths[g].exists():
                tf_final = tf_curves[g][-1]
                summary_log[f"viz/goal_{g:02d}"] = wandb.Image(
                    str(png_paths[g]),
                    caption=f"goal {g:02d}  final_tf={tf_final:.3f}",
                )
        run.log(summary_log)
        run.finish()
        print(f"\nwandb run: project={args.wandb_project!r}  "
              f"group={run.group}  name={run.name}")


if __name__ == "__main__":
    main()
