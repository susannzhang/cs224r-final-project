# The following tests are generated with the support of Gemini and evaluated by the author to confirm accuracy/completeness.

import pytest
import numpy as np
from geometry import Source, Receiver, Rod, Grid, DesignRegion, Environment

# ==========================================
# 1 & 2. SOURCE AND RECEIVER TESTS
# ==========================================
def test_source_instantiation():
    """Test instantiating a source object."""
    src = Source(index=0, frequency=1e14, length=2.0)
    assert src.frequency == 1e14
    assert src.length == 2.0
    assert src.back_wall_offset == 5  # Default value check
    assert src.walls is True  # Default value check

def test_receiver_instantiation():
    """Test instantiating a receiver object."""
    rec = Receiver(index=0, length=1.5, side='right', rod_index=1)
    assert rec.length == 1.5
    assert rec.side == 'right'
    assert rec.rod_index == 1

# ==========================================
# 3 & 4. ROD TESTS
# ==========================================
def test_rod_instantiation_and_mutability():
    """Test instantiating a rod and altering its permittivity (mutability)."""
    rod = Rod(index=(1, 1), permittivity=12.0)
    assert rod.permittivity == 12.0
    
    # User alters permittivity
    rod.permittivity = 14.5
    assert rod.permittivity == 14.5

def test_rod_hides_center_and_uninitialized_state():
    """Check if rod properly hides _center and handles uninitialized state."""
    rod = Rod(index=(1, 1), permittivity=12.0)
    
    # Test 1: repr string does not expose _center
    assert "_center" not in repr(rod)
    
    # Test 2: _center raises AttributeError before internal code assigns it
    with pytest.raises(AttributeError):
        _ = rod._center
        
    # Test 3: System can assign it later (what your stamping function will do)
    rod._center = (5, 10)
    assert rod._center == (5, 10)

# ==========================================
# 5. GRID TESTS
# ==========================================
def test_grid_instantiation_and_rod_access():
    """Test instantiating grid, storing rods, and read/write access."""
    # Instantiate the grid with the new arguments (rods is init=False)
    grid = Grid(
        num_rods_x=5, 
        num_rods_y=5, 
        radius=0.2,
        distance=0.5,
        rod_permittivity=12.0
    )
    
    # Create some rods
    rod_1 = Rod(index=(1, 1), permittivity=12.0)
    rod_2 = Rod(index=(1, 2), permittivity=12.0)
    
    # Write access: Add them to the initialized empty dictionary manually
    grid.rods[(1, 1)] = rod_1
    grid.rods[(1, 2)] = rod_2
    
    # Read access
    assert grid.num_rods_x == 5
    assert grid.radius == 0.2
    assert len(grid.rods) == 2
    assert grid.rods[(1, 1)].index == (1, 1)

# ==========================================
# 6. DESIGN REGION TESTS
# ==========================================
def test_design_region_defaults_and_hidden_canvas():
    """Test DesignRegion defaults and ensure canvas is hidden/uninitialized."""
    des_reg = DesignRegion()

    # Check default values applied correctly
    assert des_reg.resolution == 0.01
    assert des_reg.bg_permittivity == 1.0
    assert des_reg.margin_cells == 20
    assert des_reg.plate_permittivity == 1e6

    # Check _canvas is hidden from repr
    assert "_canvas" not in repr(des_reg)

    # Check uninitialized state
    with pytest.raises(AttributeError):
        _ = des_reg._canvas

    # Check assignment works (simulating your generation function)
    des_reg._canvas = np.zeros((500, 1000))
    assert des_reg._canvas.shape == (500, 1000)

# ==========================================
# 7. ENVIRONMENT TESTS
# ==========================================
def test_environment_defaults_and_access():
    """Test Environment defaults and ability to access sub-objects."""
    des_reg = DesignRegion()
    dummy_grid = Grid(num_rods_x=5, num_rods_y=5, radius=0.2, distance=0.5, rod_permittivity=12.0)
    env = Environment(design_region=des_reg, grid=dummy_grid)

    # Check defaults
    assert env.sources == []
    assert env.receivers == []

    # Check assigning and accessing sources
    env.sources.append(Source(index=0, frequency=1e14, length=2.0))
    assert len(env.sources) == 1
    assert env.sources[0].frequency == 1e14

    # Check assigning and accessing receivers
    env.receivers.append(Receiver(index=0, length=1.5, side='top', rod_index=2))
    assert len(env.receivers) == 1
    assert env.receivers[0].side == 'top'
    assert env.receivers[0].rod_index == 2

    # Check assigning and accessing the grid
    dummy_grid = Grid(num_rods_x=5, num_rods_y=5, radius=0.2, distance=0.5, rod_permittivity=12.0)
    env.grid = dummy_grid
    assert env.grid is not None
    assert env.grid.num_rods_x == 5
    assert env.grid.distance == 0.5

def test_environment_independent_lists():
    """Ensure multiple environments do not share the same source/receiver lists."""
    des_reg_1 = DesignRegion()
    grid_1 = Grid(num_rods_x=5, num_rods_y=5, radius=0.05, distance=0.02, rod_permittivity=5.0)
    env_1 = Environment(design_region=des_reg_1, grid=grid_1)

    des_reg_2 = DesignRegion()
    grid_2 = Grid(num_rods_x=5, num_rods_y=5, radius=0.05, distance=0.02, rod_permittivity=5.0)
    env_2 = Environment(design_region=des_reg_2, grid=grid_2)

    # Add a source to env_1
    env_1.sources.append(Source(index=0, frequency=1e14, length=2.0))

    # env_2 should remain completely empty
    assert len(env_1.sources) == 1
    assert len(env_2.sources) == 0