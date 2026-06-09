# =============================================================================
# PM Setup: Simulation Parameters for SPPL Plasma Metamaterial Device
# =============================================================================

"""
Physical parameters for simulating beam steering in plasma metamaterial device,
consistent with the lab setup at Stanford Plasma Physics Laboratory (SPPL).

Contents are organized into:
    - Region (rod layout, dimensions, permittivity)
    - Grid (rod geometry, spacing)
    - Sources and receivers (waveguide configuration)
    - Simulation parameters (frequency, Drude model)
"""

import numpy as np
import matplotlib.pyplot as plt

# EM simulation framework
from geometry import create_design_region, create_grid, create_source, create_receiver, create_environment
from simulation import initialize_environment, simulate_ez_fields_per_source, measure_total_intensity_per_receiver
from visualization import visualize_wave_propagation, visualize_ez_intensity, visualize_permittivity_map

# Create a design region with reasonable defaults
region = create_design_region(
    resolution=0.0005,           # 0.0005 meters per pixel
    bg_permittivity=1.0,       # Vacuum background
    margin_cells=20            # 20-cell margin on each side
)

# Create a 10×10 grid of rods
grid = create_grid(
    num_rods_x=10,              # 10 rods in x-direction
    num_rods_y=10,              # 10 rods in y-direction
    radius=0.01,                # Rod radius: 0.01 meters
    distance=0.002,             # Gap between rods: 0.002 meters
    rod_permittivity=5.0        # Silicon (high permittivity)
)

# Create a single source
source = create_source(
    index=1,                   # Source identifier
    frequency=6e9,             # 6 GHz (microwave frequency)
    length=0.02,                # Source length: 0.2 meters
    walls=True                 # Include reflective walls
)

# Create one receiver per rod around perimeter of the grid.
# Receiver indices are assigned to ascend with angle (sweeping counter-clockwise
# from bottom-left through right to top-left), so receivers print in 1..30 order.
receivers = []
# Bottom side: rod_index 1..10 runs left → right = ascending angle (−135° → −45°)
for i in range(1, 11):
    receivers.append(create_receiver(
        index=i, length=0.02, side='bottom', rod_index=i,
    ))
# Right side: rod_index 1..10 runs bottom → top = ascending angle (−40° → +40°)
for i in range(1, 11):
    receivers.append(create_receiver(
        index=10 + i, length=0.02, side='right', rod_index=i,
    ))
# Top side: ascending angle (+45° → +135°) goes right → left, so reverse rod_index
for i in range(1, 11):
    receivers.append(create_receiver(
        index=20 + i, length=0.02, side='top', rod_index=11 - i,
    ))

# Create and initialize the environment
env = create_environment(
    design_region=region, 
    grid=grid, sources=[source], 
    receivers=receivers
)
initialize_environment(env)

# Run EM simulation
print("Simulating...")
ez_fields = simulate_ez_fields_per_source(env)
print("✓ Simulation complete!")

# Extract the field for our source
ez = ez_fields[source.index]

# Create wave propogation visualization
visualize_wave_propagation(ez, canvas=env.design_region._canvas,
                           receivers=env.receivers)

# Create field intensity (|E_z|^2) visualization
visualize_ez_intensity(ez, canvas=env.design_region._canvas,
                       receivers=env.receivers)

# Measure intensity at each receiver
intensities = measure_total_intensity_per_receiver(env)

"""
Angle of a receiver in degrees w/r/t the center of the rod grid.

Convention (Cartesian, math standard):
    0°   = directly across from the source (right side, centered vertically)
    +90° = center of top side
    -90° = center of bottom side
    ±180° = source side (left)
"""
def _receiver_angle_degrees(receiver, canvas_shape):
    # Grid is centered on the canvas, so canvas center ≈ grid center
    n_rows, n_cols = canvas_shape
    center_row = (n_rows - 1) / 2.0
    center_col = (n_cols - 1) / 2.0

    # Receiver position from its mask centroid
    rows, cols = np.where(receiver._mask == 1)
    rec_row = rows.mean()
    rec_col = cols.mean()

    # Cartesian offsets relative to grid center.
    # The visualization uses imshow(origin='lower'), so array row already
    # increases upward in the displayed image — no y-flip needed.
    cart_x = rec_col - center_col
    cart_y = rec_row - center_row

    return float(np.degrees(np.arctan2(cart_y, cart_x)))

canvas_shape = env.design_region._canvas.shape

print("Power (|E_z|^2) at receivers:")
print()
# Sort by angle for readable output
sorted_items = sorted(
    intensities.items(),
    key=lambda kv: _receiver_angle_degrees(env.receivers[kv[0] - 1], canvas_shape),
)
for receiver_idx, intensity in sorted_items:
    receiver = env.receivers[receiver_idx - 1] if receiver_idx <= len(env.receivers) else None
    if receiver:
        angle = _receiver_angle_degrees(receiver, canvas_shape)
        print(f"  Receiver {receiver_idx} at {angle:+7.2f}°: {intensity:.3e}")
    else:
        print(f"  Receiver {receiver_idx}: {intensity:.3e}")