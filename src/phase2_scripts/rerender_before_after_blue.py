"""Re-render every Phase-1 target_XX/before_after.png so the permittivity panels
use the new normalized [-1,1] white->blue->black colormap (metallic->transparent,
black background). The field-intensity panels keep their original style. Fields
are recomputed with FDFD from the stored eps_initial / eps_star.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.draw import disk

_DBS = Path(__file__).resolve().parent
_PROJ = _DBS.parent
for _p in (_PROJ, _DBS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from geometry import (create_design_region, create_environment, create_grid,
                      create_receiver, create_source)
from simulation import initialize_environment, simulate_ez_fields_per_source
from algorithms.agents.es_agent import apply_eps_to_canvas


def build_env():
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0, margin_cells=20)
    grid = create_grid(num_rods_x=10, num_rods_y=10, radius=0.01, distance=0.002,
                       rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
    receivers = []
    for i in range(1, 11):
        receivers.append(create_receiver(index=i, length=0.02, side='bottom', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=10 + i, length=0.02, side='right', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=20 + i, length=0.02, side='top', rod_index=11 - i))
    env = create_environment(design_region=region, grid=grid, sources=[source], receivers=receivers)
    initialize_environment(env)
    return env


# white (metallic / -1) -> blue (0) -> black (transparent / +1)
CMAP = LinearSegmentedColormap.from_list("blue_metallic", ["#ffffff", "#1f5fff", "#000000"])

env = build_env()
shape = env.design_region._canvas.shape
radius_cells = env.grid.radius / env.design_region.resolution
rod_mask = np.zeros(shape, dtype=bool)
for (x, y), rod in env.grid.rods.items():
    rr, cc = disk(center=rod._center, radius=radius_cells, shape=shape)
    rod_mask[rr, cc] = True


def _draw_perm_blue(ax, canvas, title):
    disp = np.full(shape, 1.0, dtype=np.float32)   # background (transparent) -> black
    disp[canvas > 1.5] = -1.0                       # metallic walls -> white
    disp[rod_mask] = canvas[rod_mask]              # rods -> normalized eps in [-1,1]
    im = ax.imshow(disp, cmap=CMAP, origin="lower", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_facecolor("black")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label=r"$\varepsilon$ (normalized)")


def _draw_intensity(ax, intensity, canvas, title, target_receiver):
    vmax = np.percentile(intensity, 98)
    im = ax.imshow(intensity, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
    ax.contour(canvas, [3.0, 5e5], colors="white", alpha=0.5, linewidths=0.6)
    ax.contour(target_receiver._mask, [0.5], colors="cyan", linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label=r"$|E_z|^2$")


targets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
outroot = _DBS / "phase1-uniform-init-output"

for t in targets:
    tdir = outroot / f"target_{t:02d}"
    eps_initial = np.load(tdir / "eps_initial.npy").astype(np.float32)
    eps_final = np.load(tdir / "eps_star.npy").astype(np.float32)
    meta = json.loads((tdir / "metadata.json").read_text()) if (tdir / "metadata.json").exists() else {}
    iters = meta.get("iterations", "?")
    reward = meta.get("best_reward", float("nan"))

    apply_eps_to_canvas(env, eps_initial)
    canvas_initial = env.design_region._canvas.copy()
    intensity_initial = np.abs(sum(simulate_ez_fields_per_source(env).values())) ** 2

    apply_eps_to_canvas(env, eps_final)
    canvas_final = env.design_region._canvas.copy()
    intensity_final = np.abs(sum(simulate_ez_fields_per_source(env).values())) ** 2

    target_receiver = env.receivers[t]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    _draw_perm_blue(axes[0, 0], canvas_initial, "Initial permittivity")
    _draw_intensity(axes[0, 1], intensity_initial, canvas_initial, "Initial field intensity", target_receiver)
    _draw_perm_blue(axes[1, 0], canvas_final, f"Converged $\\varepsilon^\\star$ (target {t})")
    _draw_intensity(axes[1, 1], intensity_final, canvas_final,
                    f"Converged field intensity (reward={reward:+.2e})", target_receiver)
    fig.suptitle(f"ES target {t}: initial vs converged ({iters} iters)", fontsize=13)

    fig.savefig(tdir / "before_after.png", format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"re-rendered target_{t:02d}/before_after.png")

print("done")
