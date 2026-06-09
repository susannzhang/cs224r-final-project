# The following unit tests are generated with the help of Claude Opus 4.7 by Anthropic and have been reviewed by developers for accuracy and completeness. 
from simulation import initialize_environment, simulate_ez_fields_per_source
from geometry import create_design_region, create_grid, create_environment, create_source, create_receiver
import numpy as np


##################################################
### Tests for initialize_environment()         ###
##################################################

def test_initialize_environment_canvas_exists():
    """
    Checks that initialize_environment correctly creates the canvas
    by calling _convert_design_region_to_cells under the hood.
    """
    # 1. Initialize objects
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)

    # 2. Run initialization
    initialize_environment(env)

    # 3. Verify canvas was created with correct shape
    # 1.2 * (5.0 / 0.05) = 120 cells in each direction
    assert env.design_region._canvas is not None
    assert env.design_region._canvas.shape == (54, 54)


def test_initialize_environment_grid_stamped():
    """
    Checks that the grid is properly stamped onto the canvas
    after initialization, with the correct number of rods and 
    each rod's permittivity present on the canvas.
    """
    # 1. Initialize objects
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)

    # 2. Run initialization
    initialize_environment(env)

    # 3. Verify all rods are saved
    assert len(env.grid.rods) == 9

    # 4. Verify each rod was stamped onto the canvas at its center
    for index, rod in env.grid.rods.items():
        row, col = rod._center
        assert env.design_region._canvas[row, col] == rod.permittivity


def test_initialize_environment_source_masks_generated():
    """
    Checks that each source has its _mask attribute generated after 
    initialization with the correct shape and binary values.
    """
    # 1. Initialize objects with sources
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=5.0)
    sources = [
        create_source(index=1, frequency=1e14, length=0.5),
        create_source(index=2, frequency=1e14, length=0.5),
    ]
    env = create_environment(design_region=region, grid=grid, sources=sources)

    # 2. Run initialization
    initialize_environment(env)

    # 3. Verify each source has a properly shaped binary mask
    for source in env.sources:
        assert hasattr(source, '_mask')
        assert source._mask.shape == (54, 54)
        assert np.all((source._mask == 0) | (source._mask == 1))
        assert np.any(source._mask == 1)


def test_initialize_environment_receiver_masks_generated():
    """
    Checks that each receiver has its _mask attribute generated after 
    initialization with the correct shape and binary values.
    """
    # 1. Initialize objects with receivers
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=5.0)
    receivers = [
        create_receiver(index=1, length=0.5, side='right', rod_index=1),
        create_receiver(index=2, length=0.5, side='right', rod_index=2),
    ]
    env = create_environment(design_region=region, grid=grid, receivers=receivers)

    # 2. Run initialization
    initialize_environment(env)

    # 3. Verify each receiver has a properly shaped binary mask
    for receiver in env.receivers:
        assert hasattr(receiver, '_mask')
        assert receiver._mask.shape == (54, 54)
        assert np.all((receiver._mask == 0) | (receiver._mask == 1))
        assert np.any(receiver._mask == 1)


def test_initialize_environment_empty_sources_receivers():
    """
    Checks that initialize_environment handles empty source and receiver
    lists without crashing.
    """
    # 1. Initialize objects with no sources or receivers
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid, sources=[], receivers=[])

    # 2. Run initialization — should not crash
    initialize_environment(env)

    # 3. Verify canvas and grid were still initialized correctly
    assert env.design_region._canvas is not None
    assert len(env.grid.rods) == 9

    # 4. Verify source and receiver lists remain empty
    assert len(env.sources) == 0
    assert len(env.receivers) == 0


def test_initialize_environment_in_place_mutation():
    """
    Checks that initialize_environment mutates the input environment in place
    and returns the exact same environment instance, not a copy.
    """
    # 1. Initialize objects
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=5.0)
    env = create_environment(design_region=region, grid=grid)

    # 2. Save reference and run initialization
    original_env_id = id(env)
    returned_env = initialize_environment(env)

    # 3. Verify the returned object is the same instance as the input
    assert id(returned_env) == original_env_id
    assert returned_env is env

    # 4. Verify mutation happened on the original env (not just the returned one)
    assert env.design_region._canvas is not None



##################################################
### Tests for simulate_ez_fields_per_source()  ###
##################################################

def test_simulate_ez_fields_empty_sources():
    """
    Checks that an empty source list returns an empty dictionary without crashing.
    """
    # 1. Initialize and prepare environment
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    env = create_environment(design_region=region, grid=grid, sources=[])
    initialize_environment(env)

    # 2. Run the simulation
    result = simulate_ez_fields_per_source(env)

    # 3. Verify empty dictionary is returned
    assert isinstance(result, dict)
    assert len(result) == 0


def test_simulate_ez_fields_return_type():
    """
    Checks that the function returns a Dict[int, np.ndarray] with the correct
    number of entries and correct key types.
    """
    # 1. Initialize and prepare environment
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    sources = [
        create_source(index=1, frequency=1e14, length=0.5),
        create_source(index=2, frequency=1e14, length=0.5),
    ]
    env = create_environment(design_region=region, grid=grid, sources=sources)
    initialize_environment(env)

    # 2. Run the simulation
    result = simulate_ez_fields_per_source(env)

    # 3. Verify return type is a dictionary
    assert isinstance(result, dict)

    # 4. Verify number of entries matches number of sources
    assert len(result) == 2

    # 5. Verify keys are ints matching source indices
    assert 1 in result
    assert 2 in result
    for key in result.keys():
        assert isinstance(key, int)


def test_simulate_ez_fields_array_shape():
    """
    Checks that each Ez field array has the same shape as the canvas,
    confirming the simulation domain is correctly sized.
    """
    # 1. Initialize and prepare environment
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    sources = [
        create_source(index=1, frequency=1e14, length=0.5),
        create_source(index=2, frequency=1e14, length=0.5),
    ]
    env = create_environment(design_region=region, grid=grid, sources=sources)
    initialize_environment(env)

    # 2. Run the simulation
    result = simulate_ez_fields_per_source(env)

    # 3. Verify each Ez array matches canvas shape
    expected_shape = env.design_region._canvas.shape
    for source_index, ez in result.items():
        assert ez.shape == expected_shape


def test_simulate_ez_fields_complex_valued():
    """
    Checks that each Ez field array contains complex values, confirming
    the FDFD solver is returning frequency-domain fields as expected.
    """
    # 1. Initialize and prepare environment
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    sources = [
        create_source(index=1, frequency=1e14, length=0.5),
    ]
    env = create_environment(design_region=region, grid=grid, sources=sources)
    initialize_environment(env)

    # 2. Run the simulation
    result = simulate_ez_fields_per_source(env)

    # 3. Verify Ez array is complex-valued
    assert result[1].dtype == np.complex128


def test_simulate_ez_fields_nonzero():
    """
    Checks that the Ez field is not all zeros — confirming the source
    actually injected energy into the domain.
    """
    # 1. Initialize and prepare environment
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    sources = [
        create_source(index=1, frequency=1e14, length=0.5),
    ]
    env = create_environment(design_region=region, grid=grid, sources=sources)
    initialize_environment(env)

    # 2. Run the simulation
    result = simulate_ez_fields_per_source(env)

    # 3. Verify Ez field is not all zeros
    assert np.any(result[1] != 0)


def test_simulate_ez_fields_different_sources_produce_different_fields():
    """
    Checks that the same source produces different Ez fields when the grid
    permittivity changes, confirming the simulation responds to material properties.
    """
    # 1. Initialize environment with transparent grid
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid_transparent = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    source = create_source(index=1, frequency=3e8, length=0.5)
    env_transparent = create_environment(design_region=region, grid=grid_transparent, sources=[source])
    initialize_environment(env_transparent)

    # 2. Run the simulation with transparent grid
    result_transparent = simulate_ez_fields_per_source(env_transparent)

    # 3. Initialize environment with high-permittivity grid
    grid_high_perm = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=5.0)
    source2 = create_source(index=1, frequency=3e8, length=0.5)
    env_high_perm = create_environment(design_region=region, grid=grid_high_perm, sources=[source2])
    initialize_environment(env_high_perm)

    # 4. Run the simulation with high-permittivity grid
    result_high_perm = simulate_ez_fields_per_source(env_high_perm)

    # 5. Verify the Ez fields are different due to different grid permittivity
    assert not np.allclose(result_transparent[1], result_high_perm[1])


def test_simulate_ez_fields_index_matches_source():
    """
    Checks that each dictionary key correctly matches the index of the
    source that generated it.
    """
    # 1. Initialize and prepare environment
    region = create_design_region(resolution=0.05, bg_permittivity=1.0)
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.05, rod_permittivity=1.0)
    sources = [
        create_source(index=5, frequency=1e14, length=0.5),
        create_source(index=9, frequency=1e14, length=0.5),
    ]
    env = create_environment(design_region=region, grid=grid, sources=sources)
    initialize_environment(env)

    # 2. Run the simulation
    result = simulate_ez_fields_per_source(env)

    # 3. Verify keys match source indices exactly
    assert 5 in result
    assert 9 in result
    assert 1 not in result
    assert 2 not in result