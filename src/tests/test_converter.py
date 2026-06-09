
import pytest
import numpy as np
from geometry import Rod, create_design_region, create_environment, create_grid, create_source, create_receiver
from converter import _convert_design_region_to_cells, _convert_grid_to_cells, _convert_sources_to_cells, _convert_receivers_to_cells, _find_center_for_first_rod_in_grid_cells, _stamp_rod_to_design_region, _stamp_grid_onto_design_region

#####################################################
###  Tests for _convert_design_region_to_cells()  ###
#####################################################
def test_convert_design_region_dimensions_and_values():
    """
    Checks that the numpy array has the correct dimensions based on grid + margins,
    and that each pixel is correctly initialized to the background permittivity.
    """
    # Grid: 5x5 rods, radius=0.05m, distance=0.02m, resolution=0.05m/cell
    # grid_x = (5 * 0.1 + 4 * 0.02) / 0.05 = 0.58 / 0.05 = 11.6 → 11 cells
    # grid_y = (5 * 0.1 + 4 * 0.02) / 0.05 = 0.58 / 0.05 = 11.6 → 11 cells
    # With margin_cells=20: total = 11 + 40 = 51 cells on each axis
    region = create_design_region(
        resolution=0.05,
        bg_permittivity=1.0,
        margin_cells=20
    )

    dummy_grid = create_grid(num_rods_x=5, num_rods_y=5, radius=0.05, distance=0.02, rod_permittivity=0.5)
    env = create_environment(design_region=region, grid=dummy_grid)

    # Run the conversion function
    _convert_design_region_to_cells(env)
    sim_grid = env.design_region._canvas

    # Verify canvas size
    assert sim_grid.shape == (51, 51)
    # Verify all pixels initialized to background permittivity
    assert np.all(sim_grid == 1.0)

def test_agent_pixel_modification():
    """
    Checks that an agent can successfully interact with the
    returned numpy array and overwrite specific pixel values, confirming it
    behaves as a standard, mutable environment grid.
    """
    region = create_design_region(
        resolution=0.1,
        bg_permittivity=1.0,
        margin_cells=20
    )
    dummy_grid = create_grid(num_rods_x=5, num_rods_y=5, radius=0.05, distance=0.02, rod_permittivity=0.5)
    env = create_environment(design_region=region, grid=dummy_grid)
    _convert_design_region_to_cells(env)
    sim_grid = env.design_region._canvas

    # Agent modifies some pixels in the center
    center = sim_grid.shape[0] // 2
    sim_grid[center-2:center+2, center-2:center+2] = 12.0

    # Verify the modification took hold
    assert sim_grid[center, center] == 12.0
    # Verify surrounding pixels remain unchanged
    assert sim_grid[0, 0] == 1.0


############################################
###  Tests for _convert_grid_to_cells()  ###
############################################
def test_convert_grid_to_cells_basic():
    """
    Checks that the function correctly computes grid size in cells
    using the formula: (N * diameter + (N - 1) * spacing) / resolution.
    """
    # 1. Initialize objects
    # diameter = 2 * 0.1 = 0.2m, spacing = 0.05m, resolution = 0.05m/cell
    # grid_x = (3 * 0.2 + 2 * 0.05) / 0.05 = (0.6 + 0.1) / 0.05 = 14 cells
    # grid_y = (4 * 0.2 + 3 * 0.05) / 0.05 = (0.8 + 0.15) / 0.05 = 19 cells
    region = create_design_region(resolution=0.05)
    grid = create_grid(num_rods_x=3, num_rods_y=4, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)

    # 2. Run the conversion function
    grid_x, grid_y = _convert_grid_to_cells(env)

    # 3. Verify correct cell counts
    assert grid_x == 14
    assert grid_y == 19

def test_convert_grid_to_cells_asymmetric():
    """
    Checks that x and y axes are computed independently and not swapped,
    using a clearly asymmetric grid.
    """
    # 1. Initialize objects
    # grid_x = (2 * 0.2 + 1 * 0.05) / 0.05 = 0.45 / 0.05 = 9 cells
    # grid_y = (7 * 0.2 + 6 * 0.05) / 0.05 = 1.70 / 0.05 = 34 cells
    region = create_design_region(resolution=0.05)
    grid = create_grid(num_rods_x=2, num_rods_y=7, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)

    # 2. Run the conversion function
    grid_x, grid_y = _convert_grid_to_cells(env)

    # 3. Verify axes are not swapped
    assert grid_x == 9
    assert grid_y == 34

def test_convert_grid_to_cells_single_rod():
    """
    Checks the edge case of a single rod, where the spacing term
    (N-1) * spacing drops to zero and result is just diameter / resolution.
    """
    # 1. Initialize objects
    # grid_x = (1 * 0.2 + 0 * 0.05) / 0.05 = 0.2 / 0.05 = 4 cells
    # grid_y = (1 * 0.2 + 0 * 0.05) / 0.05 = 0.2 / 0.05 = 4 cells
    region = create_design_region(resolution=0.05)
    grid = create_grid(num_rods_x=1, num_rods_y=1, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)

    # 2. Run the conversion function
    grid_x, grid_y = _convert_grid_to_cells(env)

    # 3. Verify spacing term correctly vanishes
    assert grid_x == 4
    assert grid_y == 4

def test_convert_grid_to_cells_truncation():
    """
    Checks that when the cell calculation produces a float, it is
    correctly truncated to an int rather than rounded.
    """
    # 1. Initialize objects
    # Deliberately chosen so division does not produce a whole number:
    # diameter = 2 * 0.1 = 0.2m, spacing = 0.05m, resolution = 0.03 m/cell
    # grid_x = (3 * 0.2 + 2 * 0.05) / 0.03 = 0.70 / 0.03 = 23.333... -> int = 23
    # grid_y = (4 * 0.2 + 3 * 0.05) / 0.03 = 0.95 / 0.03 = 31.666... -> int = 31
    region = create_design_region(resolution=0.03)
    grid = create_grid(num_rods_x=3, num_rods_y=4, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)

    # 2. Run the conversion function
    grid_x, grid_y = _convert_grid_to_cells(env)

    # 3. Verify truncation (not rounding — 23.33 should be 23, not 24)
    assert grid_x == 23
    assert grid_y == 31

    # 4. Verify the return types are ints, not floats
    assert isinstance(grid_x, int)
    assert isinstance(grid_y, int)



###############################################
###  Tests for _convert_sources_to_cells()  ###
###############################################
def test_convert_sources_empty_list():
    """
    Checks that an empty source list does nothing and does not crash.
    """
    # 1. Initialize objects with no sources
    region = create_design_region(resolution=0.05)
    dummy_grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=dummy_grid, sources=[])
    _convert_design_region_to_cells(env)

    # 2. Run the conversion function — should not crash
    _convert_sources_to_cells(env)

    # 3. Verify source list remains empty and no masks were generated
    assert len(env.sources) == 0


def test_convert_sources_single_source():
    """
    Checks that a single source produces a mask with the correct
    shape, correct binary values (only 0s and 1s), and at least one active cell.
    """
    # 1. Initialize objects with one source (fixed left side position)
    region = create_design_region(resolution=0.05)
    dummy_grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    source = create_source(index=1, frequency=1e14, length=0.5)
    env = create_environment(design_region=region, grid=dummy_grid, sources=[source])
    _convert_design_region_to_cells(env)

    # 2. Run the conversion function
    _convert_sources_to_cells(env)

    # 3. Access mask directly from source object
    mask = env.sources[0]._mask

    # 4. Verify mask shape: 3x3 grid with radius=0.1, distance=0.05, resolution=0.05
    # grid = (3*0.2 + 2*0.05)/0.05 = 14 cells, canvas = 14 + 40 = 54 cells
    assert mask.shape == (54, 54)

    # 5. Verify mask is binary — only 0s and 1s
    assert np.all((mask == 0) | (mask == 1))

    # 6. Verify at least one cell is active
    assert np.any(mask == 1)


def test_convert_sources_multiple_sources():
    """
    Checks that multiple sources each have a mask generated with correct
    shape and binary values, and that each mask is independent.
    """
    # 1. Initialize objects with three sources (all at fixed left side)
    region = create_design_region(resolution=0.05)
    dummy_grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    sources = [
        create_source(index=1, frequency=1e14, length=0.5),
        create_source(index=2, frequency=1e14, length=0.5),
        create_source(index=3, frequency=1e14, length=0.5),
    ]
    env = create_environment(design_region=region, grid=dummy_grid, sources=sources)
    _convert_design_region_to_cells(env)

    # 2. Run the conversion function
    _convert_sources_to_cells(env)

    # 3. Verify correct number of masks generated
    assert len(env.sources) == 3

    # 4. Verify each mask independently
    for source in env.sources:
        mask = source._mask
        # Correct shape: canvas = 54x54
        assert mask.shape == (54, 54)
        # Binary values only
        assert np.all((mask == 0) | (mask == 1))
        # At least one active cell
        assert np.any(mask == 1)

    # 5. Verify masks are independent objects in memory
    assert id(env.sources[0]._mask) != id(env.sources[1]._mask)
    assert id(env.sources[1]._mask) != id(env.sources[2]._mask)


def test_convert_sources_visual_check():
    """
    Visual sanity check — prints the actual mask as a 2D grid of 0s and 1s
    for a left-side source so the developer can manually verify placement.

    Not a strict assertion test — intended for developer inspection.
    Run with: pytest -s tests/test_converter.py::test_convert_sources_visual_check
    """
    # 1. Initialize objects
    # Using low resolution to keep the printed output readable
    region = create_design_region(resolution=0.1)
    dummy_grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)

    # 2. Create and display single source on left side
    source = create_source(index=1, frequency=1e14, length=1.5)
    env = create_environment(design_region=region, grid=dummy_grid, sources=[source])
    _convert_design_region_to_cells(env)
    _convert_sources_to_cells(env)
    mask = env.sources[0]._mask

    print(f"\n--- Source on left side ---")
    print(f"Mask shape: {mask.shape}")

    # Print the mask as a 2D grid of 0s and 1s
    for row in mask.astype(int):
        print(" ".join(str(cell) for cell in row))



################################################
###  Tests for _convert_receivers_to_cells() ###
################################################
def test_convert_receivers_empty_list():
    """
    Checks that an empty receiver list does nothing and does not crash.
    """
    # 1. Initialize objects with no receivers
    region = create_design_region(resolution=0.05)
    dummy_grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=dummy_grid, receivers=[])
    _convert_design_region_to_cells(env)

    # 2. Run the conversion function — should not crash
    _convert_receivers_to_cells(env)

    # 3. Verify receiver list remains empty and no masks were generated
    assert len(env.receivers) == 0


def test_convert_receivers_single_receiver():
    """
    Checks that a single receiver produces a mask with the correct
    shape, correct binary values (only 0s and 1s), and at least one active cell.
    """
    # 1. Initialize objects with one receiver on right side
    region = create_design_region(resolution=0.05)
    dummy_grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    receiver = create_receiver(index=1, length=0.5, side='right', rod_index=1)
    env = create_environment(design_region=region, grid=dummy_grid, receivers=[receiver])
    _convert_design_region_to_cells(env)

    # 2. Need to stamp grid first so receiver placement can be calculated
    from converter import _stamp_grid_onto_design_region
    _stamp_grid_onto_design_region(env)

    # 3. Run the conversion function
    _convert_receivers_to_cells(env)

    # 4. Access mask directly from receiver object
    mask = env.receivers[0]._mask

    # 5. Verify mask shape: canvas = 54x54
    assert mask.shape == (54, 54)

    # 5. Verify mask is binary — only 0s and 1s
    assert np.all((mask == 0) | (mask == 1))

    # 6. Verify at least one cell is active
    assert np.any(mask == 1)


def test_convert_receivers_multiple_receivers():
    """
    Checks that multiple receivers each have a mask generated with correct
    shape and binary values, and that each mask is independent.
    """
    # 1. Initialize objects with three receivers on different sides
    region = create_design_region(resolution=0.05)
    dummy_grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    receivers = [
        create_receiver(index=1, length=0.5, side='right', rod_index=1),
        create_receiver(index=2, length=0.5, side='top', rod_index=2),
        create_receiver(index=3, length=0.5, side='bottom', rod_index=3),
    ]
    env = create_environment(design_region=region, grid=dummy_grid, receivers=receivers)
    _convert_design_region_to_cells(env)

    # 2. Need to stamp grid first so receiver placement can be calculated
    from converter import _stamp_grid_onto_design_region
    _stamp_grid_onto_design_region(env)

    # 3. Run the conversion function
    _convert_receivers_to_cells(env)

    # 4. Verify correct number of masks generated
    assert len(env.receivers) == 3

    # 5. Verify each mask independently
    for receiver in env.receivers:
        mask = receiver._mask
        # Correct shape: canvas = 54x54
        assert mask.shape == (54, 54)
        # Binary values only
        assert np.all((mask == 0) | (mask == 1))
        # At least one active cell
        assert np.any(mask == 1)

    # 5. Verify masks are independent objects in memory
    assert id(env.receivers[0]._mask) != id(env.receivers[1]._mask)
    assert id(env.receivers[1]._mask) != id(env.receivers[2]._mask)


def test_convert_receivers_visual_check():
    """
    Visual sanity check — prints the actual mask as a 2D grid of 0s and 1s
    for receivers on right, top, and bottom sides so the developer can manually
    verify correct placement.

    Not a strict assertion test — intended for developer inspection.
    Run with: pytest -s tests/test_converter.py::test_convert_receivers_visual_check
    """
    # 1. Initialize objects
    # Using low resolution to keep the printed output readable
    region = create_design_region(resolution=0.1)
    dummy_grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    sides = {
        "right": ('right', 2),
        "top": ('top', 2),
        "bottom": ('bottom', 2),
    }

    # Need to stamp grid once for rod position calculations
    env_for_grid = create_environment(design_region=region, grid=dummy_grid)
    _convert_design_region_to_cells(env_for_grid)
    from converter import _stamp_grid_onto_design_region
    _stamp_grid_onto_design_region(env_for_grid)

    # 2. Print each mask
    for label, (side, rod_index) in sides.items():
        receiver = create_receiver(index=1, length=1.5, side=side, rod_index=rod_index)
        env = create_environment(design_region=region, grid=dummy_grid, receivers=[receiver])
        _convert_design_region_to_cells(env)
        _stamp_grid_onto_design_region(env)
        _convert_receivers_to_cells(env)
        mask = env.receivers[0]._mask

        print(f"\n--- Receiver on {label} side ---")
        print(f"Mask shape: {mask.shape}")

        # Print the mask as a 2D grid of 0s and 1s
        for row in mask.astype(int):
            print(" ".join(str(cell) for cell in row))


############################################################
### Tests for _find_center_for_first_rod_in_grid_cells() ###
############################################################

def test_find_center_single_rod():
    """
    Checks that a single rod grid places the rod exactly at the canvas center.
    For a 1x1 grid, the offset cancels exactly, landing the rod at the origin.

    Manual calculation (with margin_cells=20):
    - rod_radius = 0.1 / 0.05 = 2 cells
    - grid_x = grid_y = (1 * 0.2 + 0 * 0.05) / 0.05 = 4 cells
    - canvas shape = 4 + 40 = 44 cells → origin = (22, 22)
    - rod_x = 22 - (4/2 - 2) = 22 - 0 = 22
    - rod_y = 22 + (4/2 - 2) = 22 + 0 = 22
    """
    # 1. Initialize objects
    region = create_design_region(resolution=0.05)
    grid = create_grid(num_rods_x=1, num_rods_y=1, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Run the function
    rod_x, rod_y = _find_center_for_first_rod_in_grid_cells(env)

    # 3. Verify rod lands exactly at canvas center
    assert rod_x == 22
    assert rod_y == 22


def test_find_center_square_grid():
    """
    Checks that a symmetric square grid places the first rod at the correct
    offset from the canvas center, with equal offsets in x and y.

    Manual calculation (with margin_cells=20):
    - rod_radius = 0.1 / 0.05 = 2 cells
    - grid_x = grid_y = (3 * 0.2 + 2 * 0.05) / 0.05 = 0.7 / 0.05 = 14 cells
    - canvas shape = 14 + 40 = 54 cells → origin = (27, 27)
    - rod_x = 27 - (14/2 - 2) = 27 - 5 = 22
    - rod_y = 27 - (14/2 - 2) = 27 - 5 = 22  (rod [1,1] is bottom-left)
    """
    # 1. Initialize objects
    region = create_design_region(resolution=0.05)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Run the function
    rod_x, rod_y = _find_center_for_first_rod_in_grid_cells(env)

    # 3. Verify correct cell coordinates
    assert rod_x == 22
    assert rod_y == 22

    # 4. Verify offsets from origin are symmetric for square grid
    origin_x = env.design_region._canvas.shape[0] // 2
    origin_y = env.design_region._canvas.shape[1] // 2
    assert (origin_x - rod_x) == (origin_y - rod_y)

    # 5. Verify return types are ints
    assert isinstance(rod_x, int)
    assert isinstance(rod_y, int)


def test_find_center_asymmetric_grid():
    """
    Checks that an asymmetric grid computes x and y offsets independently
    and does not swap them.

    Manual calculation (with margin_cells=20):
    - rod_radius = 0.1 / 0.05 = 2 cells
    - grid_x = (2 * 0.2 + 1 * 0.05) / 0.05 = 0.45 / 0.05 = 9 cells
    - grid_y = (5 * 0.2 + 4 * 0.05) / 0.05 = 23 cells
    - canvas_x = 9 + 40 = 49, canvas_y = 23 + 40 = 63, origin = (24, 31)
    - rod_x = int(24 - (9/2  - 2)) = int(21.5) = 21
    - rod_y = int(31 - (23/2 - 2)) = int(21.5) = 21
      (rod [1,1] is bottom-left → both offsets subtract from origin)
    """
    # 1. Initialize objects
    region = create_design_region(resolution=0.05)
    grid = create_grid(num_rods_x=2, num_rods_y=5, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Run the function
    rod_x, rod_y = _find_center_for_first_rod_in_grid_cells(env)

    # 3. Verify correct cell coordinates
    assert rod_x == 21
    assert rod_y == 21

    # 4. Verify x and y offsets are different — confirms axes not swapped.
    #    Asymmetric grid (2x5 rods) → different offsets along the two axes.
    origin_x = env.design_region._canvas.shape[0] // 2
    origin_y = env.design_region._canvas.shape[1] // 2
    assert (origin_x - rod_x) != (origin_y - rod_y)


def test_find_center_different_resolutions():
    """
    Checks that the same physical grid at two different resolutions produces
    proportionally different cell coordinates — fine resolution (2x) gives 2x
    larger offsets.

    Manual calculation (coarse, resolution=0.1, with margin_cells=20):
    - rod_radius = 0.1 / 0.1 = 1 cell
    - grid_x = grid_y = (3 * 0.2 + 2 * 0.1) / 0.1 = 0.8 / 0.1 = 8 cells
    - canvas = 8 + 40 = 48 cells → origin = (24, 24)
    - rod_x = int(24 - (8/2 - 1)) = int(24 - 3) = 21
    - rod_y = int(24 - (8/2 - 1)) = int(24 - 3) = 21
    - offset_x = offset_y = 3 cells

    Manual calculation (fine, resolution=0.05):
    - rod_radius = 0.1 / 0.05 = 2 cells
    - grid_x = grid_y = (3 * 0.2 + 2 * 0.1) / 0.05 = 0.8 / 0.05 = 16 cells
    - canvas = 16 + 40 = 56 cells → origin = (28, 28)
    - rod_x = int(28 - (16/2 - 2)) = int(28 - 6) = 22
    - rod_y = int(28 - (16/2 - 2)) = int(28 - 6) = 22
    - offset_x = offset_y = 6 cells = 2 * 3 ✓
    """
    # 1. Initialize objects at both resolutions
    region_coarse = create_design_region(resolution=0.1)
    region_fine   = create_design_region(resolution=0.05)
    grid_coarse = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.1, rod_permittivity=1.0)
    grid_fine   = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.1, rod_permittivity=1.0)
    env_coarse = create_environment(design_region=region_coarse, grid=grid_coarse)
    env_fine   = create_environment(design_region=region_fine,   grid=grid_fine)
    _convert_design_region_to_cells(env_coarse)
    _convert_design_region_to_cells(env_fine)

    # 2. Run the function for both resolutions
    rod_x_coarse, rod_y_coarse = _find_center_for_first_rod_in_grid_cells(env_coarse)
    rod_x_fine,   rod_y_fine   = _find_center_for_first_rod_in_grid_cells(env_fine)

    # 3. Verify absolute coordinates
    assert rod_x_coarse == 21
    assert rod_y_coarse == 21
    assert rod_x_fine == 22
    assert rod_y_fine == 22

    # 4. Verify fine offsets are exactly 2x the coarse offsets
    origin_x_coarse = env_coarse.design_region._canvas.shape[0] // 2
    origin_y_coarse = env_coarse.design_region._canvas.shape[1] // 2
    origin_x_fine   = env_fine.design_region._canvas.shape[0] // 2
    origin_y_fine   = env_fine.design_region._canvas.shape[1] // 2

    assert (origin_x_fine - rod_x_fine) == 2 * (origin_x_coarse - rod_x_coarse)
    assert (rod_y_fine - origin_y_fine) == 2 * (rod_y_coarse - origin_y_coarse)


###############################################
### Tests for _stamp_rod_to_design_region() ###
###############################################

def test_stamp_rod_correct_permittivity():
    """
    Checks that the stamped pixels are set to the rod's permittivity
    and that the background pixels remain unchanged.
    """
    # 1. Initialize objects (1x1 grid, canvas = 44x44)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=1, num_rods_y=1, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Manually create a rod with a known center at the canvas center
    rod = Rod(index=(1, 1), permittivity=5.0)
    rod._center = (22, 22)

    # 3. Stamp the rod
    _stamp_rod_to_design_region(rod=rod, env=env)

    # 4. Verify stamped pixels are set to rod permittivity
    assert env.design_region._canvas[22, 22] == 5.0

    # 5. Verify background pixels remain unchanged
    assert env.design_region._canvas[0, 0] == 1.0
    assert env.design_region._canvas[43, 43] == 1.0


def test_stamp_rod_circular_shape():
    """
    Checks that the stamped region is circular by verifying pixels inside
    the radius are set to rod permittivity and pixels outside are not.

    Manual calculation:
    - radius = 0.1 / 0.05 = 2 cells
    - center = (22, 22)
    - pixel at (22, 23) is inside the radius → should be stamped
    - pixel at (22, 24) is exactly on the boundary → should not be stamped
    - pixel at (22, 25) is outside the radius → should remain background
    """
    # 1. Initialize objects (1x1 grid, canvas = 44x44)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=1, num_rods_y=1, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Manually create a rod with known center
    rod = Rod(index=(1, 1), permittivity=5.0)
    rod._center = (22, 22)

    # 3. Stamp the rod
    _stamp_rod_to_design_region(rod=rod, env=env)

    # 4. Verify pixel inside radius is stamped
    assert env.design_region._canvas[22, 23] == 5.0

    # 5. Verify pixel outside radius is untouched
    assert env.design_region._canvas[22, 25] == 1.0


def test_stamp_rod_multiple_rods_independent():
    """
    Checks that stamping multiple rods at different locations does not
    cause them to interfere with each other.
    """
    # 1. Initialize objects (2x1 grid, canvas = 49x44)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=2, num_rods_y=1, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Create two rods at well separated centers with different permittivities
    rod_1 = Rod(index=(1, 1), permittivity=5.0)
    rod_1._center = (10, 10)

    rod_2 = Rod(index=(2, 1), permittivity=9.0)
    rod_2._center = (40, 35)

    # 3. Stamp both rods
    _stamp_rod_to_design_region(rod=rod_1, env=env)
    _stamp_rod_to_design_region(rod=rod_2, env=env)

    # 4. Verify each rod stamped correctly at its own location
    assert env.design_region._canvas[10, 10] == 5.0
    assert env.design_region._canvas[40, 35] == 9.0

    # 5. Verify rods did not bleed into each other's locations
    assert env.design_region._canvas[10, 10] != 9.0
    assert env.design_region._canvas[40, 35] != 5.0


def test_stamp_rod_in_place():
    """
    Checks that the function modifies _canvas in place and does not
    return anything.
    """
    # 1. Initialize objects (1x1 grid, canvas = 44x44)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=1, num_rods_y=1, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    rod = Rod(index=(1, 1), permittivity=5.0)
    rod._center = (22, 22)

    # 2. Verify function returns None
    result = _stamp_rod_to_design_region(rod=rod, env=env)
    assert result is None

    # 3. Verify canvas was modified in place
    assert env.design_region._canvas[22, 22] == 5.0


def test_stamp_rod_overwrite():
    """
    Checks that stamping a second rod onto the same location correctly
    overwrites the first rod's permittivity values in place.
    """
    # 1. Initialize objects (1x1 grid, canvas = 44x44)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=1, num_rods_y=1, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Create two rods at the same center with different permittivities
    rod_1 = Rod(index=(1, 1), permittivity=5.0)
    rod_1._center = (22, 22)

    rod_2 = Rod(index=(1, 1), permittivity=9.0)
    rod_2._center = (22, 22)

    # 3. Stamp rod_1 first and verify it was stamped correctly
    _stamp_rod_to_design_region(rod=rod_1, env=env)
    assert env.design_region._canvas[22, 22] == 5.0

    # 4. Stamp rod_2 on top and verify it overwrote rod_1
    _stamp_rod_to_design_region(rod=rod_2, env=env)
    assert env.design_region._canvas[22, 22] == 9.0

    # 5. Verify rod_1 permittivity is completely gone
    assert env.design_region._canvas[22, 22] != 5.0


def test_stamp_rod_visual_check():
    """
    Visual sanity check — prints the design region as a 2D grid after stamping
    rods to manually verify correct placement and shape.

    Run with: pytest -s tests/test_converter.py::test_stamp_rod_visual_check
    """
    # 1. Initialize objects
    # Using low resolution to keep the printed output readable
    region = create_design_region(resolution=0.01, bg_permittivity=0.0)
    grid = create_grid(num_rods_x=2, num_rods_y=2, radius=0.05, distance=0.02, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Manually place rods at known locations
    rod_1 = Rod(index=(1, 1), permittivity=1.0)
    rod_1._center = (36, 24)

    rod_2 = Rod(index=(2, 1), permittivity=1.0)
    rod_2._center = (36, 36) 

    rod_3 = Rod(index=(1, 2), permittivity=1.0)
    rod_3._center = (24, 24)

    rod_4 = Rod(index=(2, 2), permittivity=1.0)
    rod_4._center = (24, 36)

    # 3. Stamp all rods
    _stamp_rod_to_design_region(rod=rod_1, env=env)
    _stamp_rod_to_design_region(rod=rod_2, env=env)
    _stamp_rod_to_design_region(rod=rod_3, env=env)
    _stamp_rod_to_design_region(rod=rod_4, env=env)

    # 4. Print the canvas — 0 is background, 1 is rod
    print(f"\n--- Visual Check: 2x2 grid of stamped rods ---")
    print(f"Canvas shape: {env.design_region._canvas.shape}")
    print(f"Expected: four circular blobs arranged in a 2x2 pattern\n")
    for row in env.design_region._canvas.astype(int):
        print(" ".join(str(cell) for cell in row))


##################################################
### Tests for _stamp_grid_onto_design_region() ###
##################################################

def test_stamp_grid_single_rod():
    """
    Checks that a single rod grid stamps exactly one rod at the canvas center
    with the correct permittivity.

    Manual calculation (with margin_cells=20):
    - rod_radius = 0.1 / 0.05 = 2 cells
    - grid_x = grid_y = (1 * 0.2 + 0 * 0.05) / 0.05 = 4 cells
    - canvas = 4 + 40 = 44 cells → origin = (22, 22)
    - first_rod = (22 - (4/2 - 2), 22 + (4/2 - 2)) = (22, 22)
    """
    # 1. Initialize objects (1x1 grid, canvas = 44x44)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=1, num_rods_y=1, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Stamp the grid
    _stamp_grid_onto_design_region(env)

    # 3. Verify center pixel is stamped with rod permittivity
    assert env.design_region._canvas[22, 22] == 5.0

    # 4. Verify background pixels remain unchanged
    assert env.design_region._canvas[0, 0] == 1.0
    assert env.design_region._canvas[43, 43] == 1.0

    # 5. Verify rod count
    assert len(env.grid.rods) == 1


def test_stamp_grid_square():
    """
    Checks that a square grid stamps all rods at correct locations with
    correct permittivity, and that rod centers are symmetric.

    Manual calculation (with margin_cells=20):
    - rod_radius = 0.1 / 0.05 = 2 cells
    - shift = (0.2 + 0.05) / 0.05 = 5 cells
    - grid_x = grid_y = (3 * 0.2 + 2 * 0.05) / 0.05 = 14 cells
    - canvas = 14 + 40 = 54 cells → origin = (27, 27)
    - first_rod = (27 - (14/2 - 2), 27 + (14/2 - 2)) = (22, 32)
    - rod centers:
        [1,1]: (22, 32)   [2,1]: (27, 32)   [3,1]: (32, 32)
        [1,2]: (22, 27)   [2,2]: (27, 27)   [3,2]: (32, 27)
        [1,3]: (22, 22)   [2,3]: (27, 22)   [3,3]: (32, 22)
    """
    # 1. Initialize objects (3x3 grid, canvas = 54x54)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Stamp the grid
    _stamp_grid_onto_design_region(env)

    # 3. Verify all rod centers are stamped correctly
    assert env.design_region._canvas[22, 32] == 5.0  # rod [1,1]
    assert env.design_region._canvas[27, 27] == 5.0  # rod [2,2] center
    assert env.design_region._canvas[32, 22] == 5.0  # rod [3,3]

    # 4. Verify rod count
    assert len(env.grid.rods) == 9

    # 5. Verify background pixel is unchanged
    assert env.design_region._canvas[0, 0] == 1.0


def test_stamp_grid_rectangular():
    """
    Checks that a rectangular grid stamps rods correctly along both axes
    independently, confirming x and y are not swapped.

    Manual calculation (with margin_cells=20):
    - rod_radius = 0.1 / 0.05 = 2 cells
    - shift = (0.2 + 0.05) / 0.05 = 5 cells
    - grid_x = (2 * 0.2 + 1 * 0.05) / 0.05 = int(9.0) = 9 cells
    - grid_y = (4 * 0.2 + 3 * 0.05) / 0.05 = int(19.0) = 19 cells
    - canvas_x = 9 + 40 = 49, canvas_y = 19 + 40 = 59, origin = (24, 29)
    - first_rod_x = int(24 - (9/2  - 2)) = int(21.5) = 21
    - first_rod_y = int(29 - (19/2 - 2)) = int(21.5) = 21
      (rod [1,1] is bottom-left; both _center components grow with grid index)
    - rod [1,1]: (21, 21)
    - rod [2,1]: (26, 21)
    - rod [1,4]: (21, 36)
    - rod [2,4]: (26, 36)
    """
    # 1. Initialize objects (2x4 grid)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=2, num_rods_y=4, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Stamp the grid
    _stamp_grid_onto_design_region(env)

    # 3. Verify corner rod centers are stamped correctly
    assert env.design_region._canvas[21, 21] == 5.0  # rod [1, 1] bottom-left
    assert env.design_region._canvas[26, 21] == 5.0  # rod [2, 1] top-left
    assert env.design_region._canvas[21, 36] == 5.0  # rod [1, 4] bottom-right
    assert env.design_region._canvas[26, 36] == 5.0  # rod [2, 4] top-right
    # 4. Verify rod count
    assert len(env.grid.rods) == 8

    # 5. Verify x and y axes handled independently: stepping along grid-x
    #    should change the row only, and stepping along grid-y should change
    #    the col only. (Tests that x and y are not swapped in the placement.)
    c11 = env.grid.rods[(1, 1)]._center
    c21 = env.grid.rods[(2, 1)]._center  # +1 in grid x → row changes, col same
    c12 = env.grid.rods[(1, 2)]._center  # +1 in grid y → col changes, row same
    assert c21[0] != c11[0] and c21[1] == c11[1]
    assert c12[1] != c11[1] and c12[0] == c11[0]


def test_stamp_grid_same_permittivity_as_background():
    """
    Checks that when rod permittivity matches background, the canvas
    remains visually uniform but rods are still correctly saved in grid.
    """
    # 1. Initialize objects with matching permittivities
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Stamp the grid
    _stamp_grid_onto_design_region(env)

    # 3. Verify entire canvas remains at background permittivity
    assert np.all(env.design_region._canvas == 1.0)

    # 4. Verify rods are still saved correctly in grid despite visual uniformity
    assert len(env.grid.rods) == 9
    assert env.grid.rods[(1, 1)].permittivity == 1.0


def test_stamp_grid_overwrites_existing_grid():
    """
    Checks that stamping a new grid onto an existing one correctly
    overwrites the previous permittivity values in place.
    """
    # 1. Initialize objects (1x1 grid, canvas = 44x44)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=1, num_rods_y=1, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Stamp the first grid and verify
    _stamp_grid_onto_design_region(env)
    assert env.design_region._canvas[22, 22] == 5.0

    # 3. Update rod permittivity and stamp again
    env.grid.rod_permittivity = 9.0
    _stamp_grid_onto_design_region(env)

    # 4. Verify new permittivity overwrote the old one
    assert env.design_region._canvas[22, 22] == 9.0
    assert env.design_region._canvas[22, 22] != 5.0


def test_stamp_grid_saved_rod_info():
    """
    Checks that after stamping, each rod in env.grid.rods has the correct
    index, permittivity, and _center saved.

    Manual calculation (with margin_cells=20):
    - rod_radius = 0.1 / 0.05 = 2 cells
    - shift = (0.2 + 0.05) / 0.05 = 5 cells
    - grid_x = grid_y = (2 * 0.2 + 1 * 0.05) / 0.05 = int(9.0) = 9 cells
    - canvas = 9 + 40 = 49 cells → origin = (24, 24)
    - first_rod_x = int(24 - (9/2 - 2)) = int(21.5) = 21
    - first_rod_y = int(24 - (9/2 - 2)) = int(21.5) = 21
    - rod [1,1]._center = (21, 21)  (bottom-left)
    - rod [2,1]._center = (26, 21)  (+1 in grid x → +shift in row)
    - rod [1,2]._center = (21, 26)  (+1 in grid y → +shift in col)
    - rod [2,2]._center = (26, 26)
    """
    # 1. Initialize objects (2x2 grid, canvas = 49x49)
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=2, num_rods_y=2, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Stamp the grid
    _stamp_grid_onto_design_region(env)

    # 3. Verify index saved correctly for each rod
    assert env.grid.rods[(1, 1)].index == (1, 1)
    assert env.grid.rods[(2, 2)].index == (2, 2)

    # 4. Verify permittivity saved correctly for each rod
    assert env.grid.rods[(1, 1)].permittivity == 5.0
    assert env.grid.rods[(2, 2)].permittivity == 5.0

    # 5. Verify _center saved in array coordinates
    assert env.grid.rods[(1, 1)]._center == (21, 21)
    assert env.grid.rods[(2, 1)]._center == (26, 21)
    assert env.grid.rods[(1, 2)]._center == (21, 26)
    assert env.grid.rods[(2, 2)]._center == (26, 26)


def test_stamp_grid_center_matches_stamped_location():
    """
    Checks that the _center stored on each rod matches the actual location
    of the stamped permittivity on the canvas.
    """
    # 1. Initialize objects
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Stamp the grid
    _stamp_grid_onto_design_region(env)

    # 3. For every rod, verify _center pixel matches rod permittivity on canvas
    for index, rod in env.grid.rods.items():
        row, col = rod._center
        assert env.design_region._canvas[row, col] == rod.permittivity


def test_stamp_grid_rod_count():
    """
    Checks that the number of rods saved in env.grid.rods after stamping
    exactly matches num_rods_x * num_rods_y.
    """
    # 1. Initialize objects
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=4, num_rods_y=4, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Stamp the grid
    _stamp_grid_onto_design_region(env)

    # 3. Verify rod count matches expected total
    expected_count = env.grid.num_rods_x * env.grid.num_rods_y
    assert len(env.grid.rods) == expected_count
    assert len(env.grid.rods) == 16


def test_stamp_grid_visual_check():
    """
    Visual sanity check — prints the design region after stamping a 3x3 grid
    to manually verify correct placement and circular shape.

    Run with: pytest -s tests/test_converter.py::test_stamp_grid_visual_check
    """
    # 1. Initialize objects
    # Using parameters that give readable output
    region = create_design_region(resolution=0.01, bg_permittivity=0.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.05, distance=0.02, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid)
    _convert_design_region_to_cells(env)

    # 2. Stamp the grid
    _stamp_grid_onto_design_region(env)

    # 3. Print the canvas — 0 is background, 1 is rod
    print(f"\n--- Visual Check: 3x3 grid of stamped rods ---")
    print(f"Canvas shape: {env.design_region._canvas.shape}")
    print(f"Expected: nine circular blobs arranged in a 3x3 pattern\n")
    for row in env.design_region._canvas.astype(int):
        print(" ".join(str(cell) for cell in row))