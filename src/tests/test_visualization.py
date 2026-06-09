import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from geometry import create_design_region, create_grid, create_receiver, create_source, create_environment
from simulation import initialize_environment, simulate_ez_fields_per_source
from visualization import visualize_wave_propagation, visualize_ez_intensity, visualize_permittivity_map


def test_visualize_wave_propagation_visual_check():
    """
    Visual sanity check — saves the plot to tests/visual_output/ for inspection.
    
    Run with: 
    pytest -s tests/test_visualization.py::test_visualize_wave_propagation_visual_check
    """
    # 1. Initialize and simulate a clean setup
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=10, num_rods_y=10, radius=0.01, distance=0.002, rod_permittivity=5.0)
    source = create_source(index=1, frequency=6e9, length=0.02)
    # Create one receiver per rod around perimeter of the grid.
    receivers = []
    for i in range(1, 11):
        receivers.append(create_receiver(
            index=i, length=0.02, side='bottom', rod_index=i,
        ))
    for i in range(1, 11):
        receivers.append(create_receiver(
            index=10 + i, length=0.02, side='right', rod_index=i,
        ))
    for i in range(1, 11):
        receivers.append(create_receiver(
            index=20 + i, length=0.02, side='top', rod_index=i,
        ))
    env = create_environment(design_region=region, grid=grid, sources=[source], receivers=receivers)
    initialize_environment(env)

    # 2. Simulate the field
    ez_fields = simulate_ez_fields_per_source(env)
    ez = ez_fields[1]
    canvas = env.design_region._canvas

    # 3. Generate visualization and save to a persistent location
    output_dir = Path(__file__).parent / "visual_output"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "wave_propagation.png"

    visualize_wave_propagation(ez, canvas=canvas, receivers=env.receivers)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')

    # 4. Verify the file was saved and print location for manual inspection
    print(f"\nSaved to: {save_path}")
    assert save_path.exists()


def test_visualize_ez_intensity_visual_check():
    """
    Visual sanity check for field intensity visualization — saves the plot to tests/visual_output/.

    Shows the |E_z|² energy density distribution in the domain.

    Run with: 
    pytest -s tests/test_visualization.py::test_visualize_ez_intensity_visual_check
    """
    
    # 1. Initialize and simulate a clean setup
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=10, num_rods_y=10, radius=0.01, distance=0.002, rod_permittivity=5.0)
    source = create_source(index=1, frequency=6e9, length=0.02)
    receivers = []
    for i in range(1, 11):
        receivers.append(create_receiver(
            index=i, length=0.02, side='bottom', rod_index=i,
        ))
    for i in range(1, 11):
        receivers.append(create_receiver(
            index=10 + i, length=0.02, side='right', rod_index=i,
        ))
    for i in range(1, 11):
        receivers.append(create_receiver(
            index=20 + i, length=0.02, side='top', rod_index=i,
        ))
    env = create_environment(design_region=region, grid=grid, sources=[source], receivers=receivers)
    initialize_environment(env)


    # 2. Simulate the field
    ez_fields = simulate_ez_fields_per_source(env)
    ez = ez_fields[1]
    canvas = env.design_region._canvas

    # 3. Generate visualization and save to a persistent location
    output_dir = Path(__file__).parent / "visual_output"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "ez_intensity.png"

    visualize_ez_intensity(ez, canvas=canvas)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')

    # 4. Verify the file was saved and print location for manual inspection
    print(f"\nSaved to: {save_path}")
    assert save_path.exists()


def test_visualize_permittivity_map_visual_check():
    """
    Visual sanity check for permittivity structure visualization — saves the plot to tests/visual_output/.

    Shows the material distribution (background, rods, and walls) without running a simulation.

    Run with: pytest -s tests/test_visualization.py::test_visualize_permittivity_map_visual_check
    """
    # 1. Initialize a clean setup (no simulation needed for this visualization)
    region = create_design_region(resolution=0.02, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=10, num_rods_y=10, radius=0.1, distance=0.05, rod_permittivity=5.0)
    source = create_source(index=1, frequency=3e8, length=0.5)
    env = create_environment(design_region=region, grid=grid, sources=[source])
    initialize_environment(env)

    # 2. Get the permittivity canvas
    canvas = env.design_region._canvas

    # 3. Generate visualization and save to a persistent location
    output_dir = Path(__file__).parent / "visual_output"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "permittivity_map.png"

    visualize_permittivity_map(canvas)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')

    # 4. Verify the file was saved and print location for manual inspection
    print(f"\nSaved to: {save_path}")
    assert save_path.exists()