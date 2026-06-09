"""Render a 'current-state' before/after PNG from a Phase 1 Modal checkpoint.

Usage:
    python render_ckpt.py <target_idx>

Downloads /buffer/target_<NN>_ckpt.pkl from the cs224r-phase1-buffer Modal
volume, runs 2 local FDFD solves (initial ε vs current ES mean ε), and writes:

    <repo>/checkpoint_output/target_<NN>/before_current.png

Run from anywhere — paths resolve to the repo root.
"""
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent          # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                       # cs153 repo root (geometry, simulation)
for _p in (PROJECT_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from geometry import (create_design_region, create_grid, create_source,
                      create_receiver, create_environment)
from simulation import initialize_environment
from algorithms.agents.es_agent import ESAgentConfig
from train_phase1 import _render_before_after_png

if len(sys.argv) != 2:
    print("usage: python render_ckpt.py <target_idx>", file=sys.stderr)
    sys.exit(2)

target_idx = int(sys.argv[1])
out_dir = REPO_ROOT / "checkpoint_output" / f"target_{target_idx:02d}"
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_local = out_dir / f"target_{target_idx:02d}_ckpt.pkl"
out_path = out_dir / "before_current.png"

# --- pull the checkpoint from the Modal volume -----------------------
print(f"  downloading target_{target_idx:02d}_ckpt.pkl from Modal volume...")
subprocess.run(
    ["modal", "volume", "get", "cs224r-phase1-buffer",
     f"target_{target_idx:02d}_ckpt.pkl", str(ckpt_local), "--force"],
    check=True,
)

ckpt = pickle.loads(ckpt_local.read_bytes())
print(f"  ckpt: iter {ckpt.next_iteration}  "
      f"best_reward={ckpt.best_reward:+.3e}  "
      f"best_target_frac={ckpt.best_target_frac:.3f}  "
      f"history_len={len(ckpt.history)}")

# --- rebuild the same env the worker built --------------------------
region = create_design_region(resolution=0.0005, bg_permittivity=1.0, margin_cells=20)
grid = create_grid(num_rods_x=10, num_rods_y=10, radius=0.01, distance=0.002,
                   rod_permittivity=1.0)
source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
receivers = []
for i in range(1, 11):
    receivers.append(create_receiver(index=i,      length=0.02, side='bottom', rod_index=i))
for i in range(1, 11):
    receivers.append(create_receiver(index=10 + i, length=0.02, side='right',  rod_index=i))
for i in range(1, 11):
    receivers.append(create_receiver(index=20 + i, length=0.02, side='top',    rod_index=11 - i))
env = create_environment(design_region=region, grid=grid,
                         sources=[source], receivers=receivers)
initialize_environment(env)
print(f"  env built; running 2 FDFD solves (initial + current ε)...")

cfg = ESAgentConfig(K=20, M=250)   # for the figure suptitle only
result = SimpleNamespace(
    iterations=ckpt.next_iteration,
    best_reward=ckpt.best_reward,
)

png_bytes, P_initial, P_final = _render_before_after_png(
    env, ckpt.eps_initial, ckpt.eps, target_idx, cfg, result,
)
out_path.write_bytes(png_bytes)
ti, tf = P_initial.sum(), P_final.sum()
fi = P_initial[target_idx] / ti if ti > 0 else 0.0
ff = P_final[target_idx]   / tf if tf > 0 else 0.0
print(f"  target_frac: {fi:.3f} → {ff:.3f}")
print(f"  wrote {out_path}")
