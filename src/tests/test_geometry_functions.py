### Note: These tests have been developed with the help of Gemini Pro.
### The tests were reviewed by developers for complete coverage and accuracy. 


import pytest
import random
# Assuming your factory functions and dataclasses are in a file named geometry.py
from geometry import create_design_region, DesignRegion, create_source, Source, create_receiver, Receiver, _create_rod, create_grid, Grid, Rod, Environment, create_environment


############################################
### Tests for create_design_region()     ###
############################################
def test_create_design_region_defaults():
    """
    Check basic initialization and ensure omitted parameters fallback to defaults.
    """
    region = create_design_region()

    # Check that defaults were applied correctly when omitted
    assert region.resolution == 0.01
    assert region.bg_permittivity == 1.0
    assert region.margin_cells == 20
    assert region.plate_permittivity == 1e6

def test_create_design_region_overrides():
    """
    Check that a user can successfully overwrite the default parameters.
    """
    region = create_design_region(
        resolution=0.05,        # Overwriting default 0.01
        bg_permittivity=2.25,   # Overwriting default 1.0
        margin_cells=30,        # Overwriting default 20
        plate_length=10         # Overwriting default 15
    )

    # Check that all values, including overrides, were set correctly
    assert region.resolution == 0.05
    assert region.bg_permittivity == 2.25
    assert region.margin_cells == 30
    assert region.plate_length == 10


#################################
### Tests for create_source() ###
#################################

def test_create_source_defaults():
    """
    Check basic initialization and ensure the 'walls' parameter defaults to True
    when omitted.
    """
    # Providing only the mandatory parameters
    src = create_source(index=1, frequency=2.5e14, length=2.0)

    # Check that required inputs were set correctly
    assert src.frequency == 2.5e14
    assert src.length == 2.0

    # Check that the defaults were applied correctly
    assert src.walls is True
    assert src.back_wall_offset == 5

def test_create_source_overrides():
    """
    Check that a user can successfully overwrite the default parameters.
    """
    # Overwriting the default walls=True and back_wall_offset=5 parameters
    src = create_source(index=1, frequency=3.0e14, length=1.5, walls=False, back_wall_offset=10)

    # Check that all values, including the overrides, were set correctly
    assert src.frequency == 3.0e14
    assert src.length == 1.5
    assert src.walls is False
    assert src.back_wall_offset == 10


###################################
### Tests for create_receiver() ###
###################################

def test_create_receiver_defaults():
    """
    Check basic initialization and ensure all required parameters are set correctly.
    """
    # Providing the mandatory parameters
    rec = create_receiver(index=1, length=1.5, side='right', rod_index=2)

    # Check that required inputs were set correctly
    assert rec.length == 1.5
    assert rec.side == 'right'
    assert rec.rod_index == 2

###############################
### Tests for _create_rod() ###
###############################

def test_create_rod_initialization():
    """
    Check that the helper function correctly passes parameters 
    to instantiate a Rod object.
    """
    rod = _create_rod(index=(2, 3), permittivity=11.5)
    
    # Check that parameters were saved correctly
    assert rod.index == (2, 3)
    assert rod.permittivity == 11.5

def test_create_rod_hidden_center():
    """
    Check that the returned Rod object respects the init=False 
    dataclass rule for the _center variable.
    """
    rod = _create_rod(index=(1, 1), permittivity=12.0)
    
    # Attempting to access _center before the math converter assigns it 
    # should raise an AttributeError
    with pytest.raises(AttributeError):
        _ = rod._center

def test_create_rod_post_instantiation_assignment():
    """
    Check that the hidden _center variable can be successfully 
    assigned and accessed post-creation by internal simulation logic.
    """
    # 1. User creates the rod via the helper
    rod = _create_rod(index=(1, 1), permittivity=12.0)
    
    # 2. Simulate the internal math engine calculating pixel coordinates
    calculated_pixels = (150, 150)
    rod._center = calculated_pixels
    
    # 3. Verify the assignment was completely successful and readable
    assert hasattr(rod, '_center') is True
    assert rod._center == (150, 150)


###############################
### Tests for create_grid() ###
###############################

def test_create_grid_initialization():
    """
    Check basic initialization, confirm that rods are saved correctly, 
    and verify that indices and permittivity values are accurate.
    """
    # Providing standard parameters for a 3x4 grid
    grid = create_grid(num_rods_x=3, num_rods_y=4, radius=0.2, distance=0.05, rod_permittivity=11.5)
    
    # Check that required inputs were set correctly
    assert grid.radius == 0.2
    assert grid.distance == 0.05
    
    # Check that the rods dictionary was populated with the correct total amount
    assert isinstance(grid.rods, dict)
    assert len(grid.rods) == 12
    
    # Check that individual rods were instantiated correctly with accurate properties
    for x in range(1, 4):
        for y in range(1, 5):
            # Check that the index exists in the dictionary
            assert (x, y) in grid.rods
            
            # Extract the rod for inspection
            rod = grid.rods[(x, y)]
            
            # Check that the object type, index, and permittivity are correct
            assert isinstance(rod, Rod)
            assert rod.index == (x, y)
            assert rod.permittivity == 11.5

def test_create_grid_asymmetry():
    """
    Check that the grid correctly handles asymmetric rod counts 
    without swapping X and Y bounds.
    """
    # Initialize an asymmetric 2x5 grid
    grid = create_grid(num_rods_x=2, num_rods_y=5, radius=0.1, distance=0.2, rod_permittivity=1.0)
    
    # Check that valid asymmetric indices exist
    assert (2, 5) in grid.rods
    
    # Check that out-of-bounds indices are correctly excluded
    assert (5, 2) not in grid.rods

def test_create_grid_empty():
    """
    Check that passing 0 for rod counts safely creates an empty grid 
    without breaking internal logic.
    """
    # Initialize a grid with 0 rods
    grid = create_grid(num_rods_x=0, num_rods_y=0, radius=0.1, distance=0.2, rod_permittivity=12.0)
    
    # Check that rod counts reflect the inputs
    assert grid.num_rods_x == 0
    assert grid.num_rods_y == 0
    
    # Check that the rods dictionary is perfectly empty
    assert len(grid.rods) == 0

def test_create_grid_transpose():
    """
    Check that a 2x5 grid and a 5x2 grid act as transposes of each other,
    where every (x, y) rod in the first maps to a (y, x) rod in the second.
    """
    # Initialize a 2x5 and a 5x2 grid with identical physical properties
    grid_2x5 = create_grid(num_rods_x=2, num_rods_y=5, radius=0.1, distance=0.2, rod_permittivity=12.0)
    grid_5x2 = create_grid(num_rods_x=5, num_rods_y=2, radius=0.1, distance=0.2, rod_permittivity=12.0)
    
    # Check that both grids generated the same total number of rods
    assert len(grid_2x5.rods) == len(grid_5x2.rods) == 10
    
    # Iterate through every rod in the 2x5 grid
    for (x, y), rod_2x5 in grid_2x5.rods.items():
        # Check that the transposed coordinate exists in the 5x2 grid
        assert (y, x) in grid_5x2.rods
        
        # Extract the transposed rod
        rod_5x2 = grid_5x2.rods[(y, x)]
        
        # Check that the object's internal index matches the transposed position
        assert rod_5x2.index == (y, x)
        
        # Check that the permittivity remains identical
        assert rod_5x2.permittivity == rod_2x5.permittivity

def test_create_grid_modify_single_rod():
    """
    Check that modifying a specific rod's permittivity correctly 
    updates the rod within the grid's dictionary via object reference.
    """
    # Initialize a 3x3 grid
    grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.2, rod_permittivity=12.0)
    
    # Access a specific rod 
    target_rod = grid.rods[(2, 2)]
    
    # Change its permittivity
    target_rod.permittivity = 5.5
    
    # Check that the change is automatically reflected when accessing the grid
    assert grid.rods[(2, 2)].permittivity == 5.5
    
    # Check that every other rod in the grid remains completely unaffected
    for index, rod in grid.rods.items():
        if index == (2, 2):
            assert rod.permittivity == 5.5
        else:
            assert rod.permittivity == 12.0

def test_create_grid_modify_random_sampled_rods():
    """
    Check that randomly sampling 10% of the rods into a separate list and modifying 
    them correctly updates the main grid, while leaving the remaining 90% unchanged.
    """
    # Initialize a 10x10 grid (100 rods total)
    grid = create_grid(num_rods_x=10, num_rods_y=10, radius=0.1, distance=0.2, rod_permittivity=12.0)
    
    # Extract all keys and determine the 10% sample size
    all_keys = list(grid.rods.keys())
    sample_size = int(len(all_keys) * 0.10)  # 10% of 100 = 10 rods
    
    # Setting a random seed is a good habit in unit testing so the test is deterministic
    # (it will pick the same "random" 10 rods every time this specific test runs)
    random.seed(42)
    sampled_keys = random.sample(all_keys, sample_size)
    
    # Pull the randomly selected rods into a separate list
    sampled_rods = [grid.rods[key] for key in sampled_keys]
    
    # Change the permittivity of ONLY the rods in our sample list
    for rod in sampled_rods:
        rod.permittivity = 9.9
        
    # Iterate through the ENTIRE grid to verify the split
    for key, rod in grid.rods.items():
        if key in sampled_keys:
            # Check that the 10% we sampled were correctly updated in the grid
            assert rod.permittivity == 9.9
        else:
            # Check that the 90% we did NOT sample were left completely alone
            assert rod.permittivity == 12.0

def test_create_grid_independent_instances():
    """
    Check that multiple grid instances do not accidentally share the same 
    dictionary object in memory (verifying default_factory=dict protection).
    """
    # Create two completely separate grids
    grid_A = create_grid(num_rods_x=2, num_rods_y=2, radius=0.1, distance=0.2, rod_permittivity=12.0)
    grid_B = create_grid(num_rods_x=2, num_rods_y=2, radius=0.1, distance=0.2, rod_permittivity=1.0)
    
    # 1. Prove that the dictionary objects have different memory addresses
    assert id(grid_A.rods) != id(grid_B.rods)
    
    # 2. Modify a rod in Grid A
    grid_A.rods[(1, 1)].permittivity = 9.9
    
    # 3. Prove that Grid B was completely unaffected
    assert grid_B.rods[(1, 1)].permittivity == 1.0


######################################
### Tests for create_environment() ###
######################################

def test_create_environment_defaults():
    """
    Check that an environment can be initialized with only a DesignRegion,
    safely handling missing grids, sources, and receivers.
    """
    # 1. Create the mandatory objects
    region = create_design_region()

    # 2. Initialize environment
    dummy_grid = create_grid(num_rods_x=2, num_rods_y=2, radius=0.1, distance=0.2, rod_permittivity=12.0)
    env = create_environment(design_region=region, grid=dummy_grid)

    # 3. Verify defaults
    assert env.design_region.margin_cells == 20
    assert isinstance(env.sources, list)
    assert isinstance(env.grid, Grid)
    assert len(env.sources) == 0
    assert isinstance(env.receivers, list)
    assert len(env.receivers) == 0

def test_create_environment_fully_loaded():
    """
    Check that the environment correctly binds all pre-constructed objects
    together without losing data.
    """
    # 1. Initialize all dummy objects
    region = create_design_region()
    grid = create_grid(num_rods_x=2, num_rods_y=2, radius=0.1, distance=0.2, rod_permittivity=12.0)
    source = create_source(index=1, frequency=1e14, length=1.0)
    receiver = create_receiver(index=1, length=1.0, side='bottom', rod_index=1)

    # 2. Pass them all to the environment
    env = create_environment(
        design_region=region,
        grid=grid,
        sources=[source],
        receivers=[receiver]
    )

    # 3. Verify successful integration
    assert env.design_region.resolution == 0.01
    assert len(env.grid.rods) == 4
    assert env.sources[0].frequency == 1e14
    assert env.receivers[0].side == 'bottom'

def test_create_environment_list_isolation():
    """
    Check that multiple blank environments do not accidentally share the same
    empty lists in memory (protecting against the mutable default argument trap).
    """
    region = create_design_region()
    grid_A = create_grid(num_rods_x=5, num_rods_y=5, radius=0.05, distance=0.02, rod_permittivity=5.0)
    grid_B = create_grid(num_rods_x=5, num_rods_y=5, radius=0.05, distance=0.02, rod_permittivity=5.0)

    # 1. Create two totally separate environments
    env_A = create_environment(design_region=region, grid=grid_A)
    env_B = create_environment(design_region=region, grid=grid_B)

    # 2. Agent adds a source ONLY to Environment A
    new_source = create_source(index=1, frequency=2e14, length=1.0)
    env_A.sources.append(new_source)

    # 3. Verify Environment B was completely unaffected
    assert len(env_A.sources) == 1
    assert len(env_B.sources) == 0

def test_create_environment_agent_interaction():
    """
    Check that altering objects through the environment (as an agent would)
    correctly modifies the original standalone objects via memory referencing.
    """
    # 1. Initialize standalone objects
    region = create_design_region()
    original_grid = create_grid(num_rods_x=3, num_rods_y=3, radius=0.1, distance=0.2, rod_permittivity=12.0)
    original_source = create_source(index=1, frequency=1e14, length=1.0)

    # 2. Bind to environment
    env = create_environment(
        design_region=region,
        grid=original_grid,
        sources=[original_source]
    )

    # 3. Agent action: Alter the grid permittivity and source frequency via the env
    env.grid.rods[(2, 2)].permittivity = 5.0
    env.sources[0].frequency = 9.9e14

    # 4. Verify the original standalone objects reflect the agent's changes
    assert original_grid.rods[(2, 2)].permittivity == 5.0
    assert original_source.frequency == 9.9e14