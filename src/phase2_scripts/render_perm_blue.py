"""Re-render the Phase-1 permittivity visualizations with a normalized [-1,1]
white->blue->black colormap (metallic->transparent) on a black background.

Background = black; rod permittivities in [-1,1] map white(-1, metallic) ->
blue(0) -> black(+1, transparent); metallic reflector walls render white. One
image per training angle, saved as target_XX/permittivity_blue.png.
"""
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
from simulation import initialize_environment
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

# Fixed rod-disk mask (positions don't change; only the per-rod value does).
rod_mask = np.zeros(shape, dtype=bool)
for (x, y), rod in env.grid.rods.items():
    rr, cc = disk(center=rod._center, radius=radius_cells, shape=shape)
    rod_mask[rr, cc] = True

targets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
outroot = _DBS / "phase1-uniform-init-output"

for t in targets:
    eps = np.load(outroot / f"target_{t:02d}" / "eps_star.npy").astype(np.float32)
    apply_eps_to_canvas(env, eps)                 # stamp rod permittivities onto canvas
    canvas = env.design_region._canvas

    disp = np.full(shape, 1.0, dtype=np.float32)   # background (transparent) -> black (+1 end)
    disp[canvas > 1.5] = -1.0                       # metallic walls (1e6) -> white (-1 end)
    disp[rod_mask] = canvas[rod_mask]              # rods -> normalized eps in [-1,1]

    fig, ax = plt.subplots(figsize=(3, 3), facecolor="black")
    ax.imshow(disp, cmap=CMAP, origin="lower", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_facecolor("black")
    ax.axis("off")
    out = outroot / f"target_{t:02d}" / "permittivity_blue.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.0, facecolor="black")
    plt.close(fig)
    print(f"rendered target_{t:02d}  eps in [{eps.min():.2f}, {eps.max():.2f}]  -> {out.name}")

print("done")
