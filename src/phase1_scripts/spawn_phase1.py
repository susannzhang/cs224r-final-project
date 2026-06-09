"""Spawn 10 Phase 1 ES workers against the deployed Modal app.

Usage:
    python spawn_phase1.py
    python spawn_phase1.py --population-size 20 --max-iterations 250
    python spawn_phase1.py --wandb-run-id phase1-20260530-032110   # resume

Unlike `modal run`, this is NOT a Modal ephemeral entrypoint — it's a plain
Python script that calls `.spawn()` against the already-deployed function.
Each spawn queues an independent FunctionCall on the persistent
`cs224r-phase1-es` app. The calls survive local terminal close, network
blips, machine sleep, anything. Exit the script and they keep running.

Prerequisite: `modal deploy train_phase1_modal.py` (one-time, or after any
worker code change).
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import modal

# Override via env var when spawning against a parallel/alternate deployment.
# Example: PHASE1_APP_NAME=cs224r-phase1-uniform-init python spawn_phase1.py ...
APP_NAME = os.environ.get("PHASE1_APP_NAME", "cs224r-phase1-es")
FUNCTION_NAME = "train_one_target_modal"


def parse_args():
    p = argparse.ArgumentParser(description="Spawn Phase 1 ES workers on Modal")
    p.add_argument("--targets", type=str, default=None,
                   help='e.g. "0,3,6"; default = every 3rd of 30 (10 angles)')
    p.add_argument("--population-size", type=int, default=20, help="K")
    p.add_argument("--max-iterations", type=int, default=250, help="M")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=0.05, dest="alpha_1")
    p.add_argument("--eta", type=float, default=1e-2)
    p.add_argument("--k-elite", type=int, default=None)
    p.add_argument("--w-crosstalk", type=float, default=0.3)
    p.add_argument("--w-loss", type=float, default=1e-3)
    p.add_argument("--w-energy", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="phase1_training_output")
    p.add_argument("--use-critic", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="cs224r-phase1")
    p.add_argument("--wandb-entity", type=str, default="")
    p.add_argument("--wandb-run-id", type=str, default="",
                   help="Pass an existing run_id to resume; default = new timestamped id")
    return p.parse_args()


def main():
    args = parse_args()

    if args.targets is None:
        target_indices = list(range(0, 30, 3))
    else:
        target_indices = [int(x) for x in args.targets.split(",")]
    training_indices = list(target_indices)

    config_kwargs = dict(
        K=args.population_size, sigma=args.sigma, alpha_1=args.alpha_1,
        M=args.max_iterations, eta=args.eta, log_every=args.log_every,
        K_elite=args.k_elite,
        w_crosstalk=args.w_crosstalk, w_loss=args.w_loss, w_energy=args.w_energy,
    )

    wandb_run_id = args.wandb_run_id or (
        "phase1-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    launch_id = wandb_run_id

    wandb_cfg = None
    if not args.no_wandb:
        wandb_cfg = {
            "project": args.wandb_project,
            "entity": args.wandb_entity or None,
            "run_id": wandb_run_id,
        }

    print(f"Phase 1 on Modal: {len(target_indices)} targets")
    print(f"  config: K={args.population_size}  M={args.max_iterations}  "
          f"σ={args.sigma}  α_1={args.alpha_1}  η={args.eta}")
    print(f"  weights: w_crosstalk={args.w_crosstalk}  "
          f"w_loss={args.w_loss}  w_energy={args.w_energy}")
    print(f"  targets: {target_indices}")
    print(f"  launch_id: {launch_id}")
    if wandb_cfg:
        print(f"  wandb:   project={args.wandb_project}  group={wandb_run_id}")
    print()

    f = modal.Function.from_name(APP_NAME, FUNCTION_NAME)

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    spawned = []
    print("Spawning Modal function calls...")
    for seed, idx in enumerate(target_indices):
        payload = {
            "target_idx": idx,
            "training_indices": training_indices,
            "config_kwargs": config_kwargs,
            "seed": seed,
            "wandb": wandb_cfg,
            "use_critic": args.use_critic,
            "launch_id": launch_id,
            "checkpoint_every": args.checkpoint_every,
        }
        fc = f.spawn(payload)
        spawned.append({"target_idx": idx, "function_call_id": fc.object_id})
        print(f"  ✓ target {idx:02d} → {fc.object_id}")

    spawn_record = {
        "launch_id": launch_id,
        "config_kwargs": config_kwargs,
        "training_indices": training_indices,
        "spawned": spawned,
    }
    with open(out / "spawned_calls.json", "w") as fh:
        json.dump(spawn_record, fh, indent=2)

    print()
    print(f"All {len(spawned)} workers queued on persistent app {APP_NAME!r}.")
    print(f"Spawn IDs → {(out / 'spawned_calls.json').resolve()}")
    print(f"They run independently. Close this terminal anytime.")
    print()
    print(f"To download results when ready:")
    print(f"  python -m modal run train_phase1_modal.py::collect")


if __name__ == "__main__":
    main()
