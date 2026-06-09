"""
CS153, Spring 2026, Susan Zhang and Selin Ertan

The following geometry module is designed to create the physical 
grid structures, sources, receivers and overall design regions in 
Ceviche. The objective is to make the necessary conversion from 
real-world physical requirements to inner simulation dynamics. 
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# TODO: Review class structures in Python/how do I think about these again precisely? 
# TODO: Might need a helper function to go from voltage to permittivity_


@dataclass
class Source:
    """
    Defines a source object designed to insert EM waves into the design region.

    The source is placed at a fixed position: left side of the grid, centered vertically.
    All units are metric.
    """
    index: int          # index of source for simulation reference [unitless]
    frequency: float    # frequency of the wave injected by source [1/seconds]
    length: float       # the length of the source [meters]
    walls: bool = True  # whether or not to include the directional walls [unitless]
    back_wall_offset: int = 5  # cells behind source for the reflective back wall [cells]

    # The numpy mask representing source placement
    _mask: np.ndarray = field(init=False, repr=False)


@dataclass
class Receiver:
    """
    Defines a receiver object designed to measure EM waves at receiver location.

    Receivers are placed on the right, top, and bottom sides of the grid,
    aligned with specific rod rows/columns.
    All units are metric.
    """
    index: int             # index of receiver for simulation reference [unitless]
    length: float          # the length of the receiver [meters]
    side: str              # 'right', 'top', or 'bottom' placement relative to grid
    rod_index: int         # which rod row (right/left) or column (top/bottom) this aligns with [unitless]

    # The numpy array representing receiver placement
    _mask: np.ndarray = field(init=False, repr=False)


@dataclass 
class Rod:    
    """
    Defines a rod object with a set permittivity value. 

    Rod class uses cartesian coordinates.
    x-axis -> horizontal, y-axis -> vertical.
    
    All units are metric. 

    The index of the rod defines its location in the grid in relation to the bottom left rod [1,1]. 
    The rod counts increase in +x and +y directions. 
    """
    index: Tuple[int, int]  # rod location in grid [bottom left rod is [1,1]] 
    permittivity: float     # permittivity of the rod [unitless]

    # this hidden variable is for internal use only 
    # keeps track of rod center for each read/write operations
    # the center is in [pixels, pixels]
    # the reference system is based on np.array
    # top left pixel is [0,0], with [row, col] format
    # rows increase by going down, columns increase by going right
    _center: Tuple[int, int] = field(init=False, repr=False)

@dataclass
class Grid: 
    """
    Defines a grid object with a set size. 

    All units are metric. 

    Grid class uses cartesian coordinates.
    x-axis -> horizontal, y-axis -> vertical.
    
    Uses grid-specific indexing system, where the bottom-left rod is indexed [1,1]. 
    Index counts increase in +x and +y directions. 
    """
    # empty dict of rods, index is rod coordinate in grid [unitless]  
    rods: Dict[Tuple[int, int], Rod] = field(default_factory=dict, init=False)   
    num_rods_x: int          # number of rods across x-axis [unitless]
    num_rods_y: int          # number of rods across y-axis [unitless]
    radius: float            # radius of the rods [meters]
    distance: float          # distance between walls of two rods [meters] 
    rod_permittivity: float  # initial permittivity value for rods
    
@dataclass
class DesignRegion:
    """
    Defines a design region object with automatic sizing based on grid dimensions.

    Canvas size is derived from grid size plus margin. All units are metric.
    DesignRegion uses Cartesian coordinates: x-axis horizontal, y-axis vertical.
    """
    resolution: float = 0.01              # distance per simulation pixel [meters/pixel]
    bg_permittivity: float = 1.0          # background permittivity, default vacuum [unitless]
    num_pml_cells: int = 10               # number of absorbing boundary cells to prevent reflections
    margin_cells: int = 20                # extra cells on each side beyond the grid [cells]
    plate_length: int = 15                # wall plate length in cells [cells]
    plate_permittivity: float = 1e6       # high permittivity = metallic reflector [unitless]

    # the numpy array containing information about the design region
    _canvas: np.ndarray = field(init=False, repr=False)

@dataclass
class Environment:
    """
    Defines an environment object containing the design region, sources, receivers, and a grid as sub-objects. 

    Primarily designed for agent interactions. 

    Units and reference systems are as defined in original classes. 
    """
    design_region: DesignRegion       # design region for all objects [unitless]
    grid: Grid                        # list of rods, none by default [unitless]
    sources: List[Source] = field(default_factory=list)      # list of sources, none by default [unitless]
    receivers: List[Receiver] = field(default_factory=list)  # list of receivers, none by default [unitless]
    


def create_design_region(resolution: float = 0.01, bg_permittivity: float = 1.0,
                         margin_cells: int = 20, plate_length: int = 15,
                         plate_permittivity: float = 1e6) -> DesignRegion:
    """
    Initialize empty design region.

    Canvas size is derived from the grid dimensions plus margins.
    Uses metric units and Cartesian coordinates.

    Inputs:
    --> resolution [float]: Physical distance per pixel, default 0.01 [meters/pixel]
    --> bg_permittivity [float]: Background permittivity, default 1.0 [unitless]
    --> margin_cells [int]: Extra cells on each side beyond the grid, default 20 [cells]
    --> plate_length [int]: Wall plate length in cells, default 15 [cells]
    --> plate_permittivity [float]: High permittivity for reflective walls, default 1e6 [unitless]

    Returns:
    -->[DesignRegion]: Design region object with specified parameters
    """
    return DesignRegion(resolution=resolution, bg_permittivity=bg_permittivity,
                       margin_cells=margin_cells, plate_length=plate_length,
                       plate_permittivity=plate_permittivity)



def create_source(index: int, frequency: float, length: float, walls: bool = True,
                  back_wall_offset: int = 5) -> Source:
    """
    Creates a source object based on given physical parameters.

    Source is placed at a fixed position: left side of the grid, centered vertically.
    Uses metric units and Cartesian coordinates.
    Inserts directional walls around the source by default for isolation.

    Inputs:
    --> index [int]: Index of source for simulation reference [unitless]
    --> frequency [float]: Frequency of the wave generated by source [1/seconds]
    --> length [float]: Physical length of the source object [meters]
    --> walls [bool]: Specifies whether or not to include directional walls, default True [unitless]
    --> back_wall_offset [int]: Distance behind source for back wall, default 5 [cells]

    Returns:
    --> [Source]: Source object with saved parameters
    """
    return Source(index=index, frequency=frequency, length=length, walls=walls,
                  back_wall_offset=back_wall_offset) 



def create_receiver(index: int, length: float, side: str, rod_index: int) -> Receiver:
    """
    Creates a receiver object based on given physical parameters.

    Receiver is placed on one of the grid sides (right, top, or bottom),
    aligned with a specific rod row or column.
    Uses metric units and Cartesian coordinates.

    Inputs:
    --> index [int]: Index of receiver for simulation reference [unitless]
    --> length [float]: Physical length of the receiver object [meters]
    --> side [str]: Placement side: 'right', 'top', or 'bottom'
    --> rod_index [int]: Rod row (for right side) or column (for top/bottom) to align with [unitless]

    Returns:
    --> [Receiver]: Receiver object with saved parameters
    """
    return Receiver(index=index, length=length, side=side, rod_index=rod_index)



def _create_rod(index: Tuple[int, int], permittivity: float) -> Rod:
    """
    Helper function to create a rod object for grid generation.

    Uses metric units and Cartesian coordinates. 

    Rod is relative to Grid, where bottom-left rod is defined as [1,1].
    Rod indexes increase in +x and +y directions. 

    Inputs: 
    --> index Tuple[int, int]: Rod index representing [x, y] in grid [unitless]
    --> permittivity [float]: Permittivity value of the rod [unitless]

    Returns: 
    --> [Rod]: Rod object with saved parameters 
    """
    return Rod(index=index, permittivity=permittivity) 



def create_grid(num_rods_x: int, num_rods_y: int, radius: float, 
                distance: float, rod_permittivity: float) -> Grid:
    """
    Creates grid object based on given physical parameters. 

    Note: Each rod in the grid object shares the same radius; building 
    grids of various rod sizes is not supported at this time. 

    Uses metric units and Cartesian coordinates. 

    Saves rod indices in Grid coordinates, where bottom-left rod is defined as [1,1]. 
    Rod indices increase in +x and +y directions of the Cartesian reference frame. 

    Inputs: 
    --> num_rods_x [int]: Number of rods across x-axis [unitless]
    --> num_rods_y [int]: Number of rods across y-axis [unitless]
    --> radius [float]: Physical radius per rod [meters]
    --> distance [float]: Physical distance between walls of two consecutive rods [meters]
    --> rod_permittivity [float]: Permittivity value assigned to all rods [unitless]

    Returns: 
    --> [Grid]: Grid object with specified parameters
    """
    # Objective: generate a grid dictionary consisting of all of the rods
    # Do this by looping through rod indices and adding them to grid dict
    grid = Grid(num_rods_x=num_rods_x, num_rods_y=num_rods_y, radius=radius, 
                distance=distance, rod_permittivity=rod_permittivity)

    this_dict = grid.rods

    for x in range(1, num_rods_x + 1, 1): 
        for y in range(1, num_rods_y + 1, 1): 
            this_rod = _create_rod(index=(x, y), permittivity=rod_permittivity)
            this_dict[(x, y)] = this_rod

    return grid 



def create_receivers_for_grid(grid: Grid, receiver_length: float = 0.05) -> List[Receiver]:
    """
    Auto-generates a full set of receivers for the given grid.

    Creates one receiver per rod row on the right side, one per rod column on the
    top side, and one per rod column on the bottom side.

    Inputs:
    --> grid [Grid]: Grid with defined rod dimensions
    --> receiver_length [float]: Physical length of each receiver in meters, default 0.05 [meters]

    Returns:
    --> [List[Receiver]]: List of receivers positioned for right, top, and bottom sides
    """
    receivers = []
    index = 0

    # Right-side receivers: one per rod row
    for row in range(1, grid.num_rods_y + 1):
        receivers.append(create_receiver(index=index, length=receiver_length,
                                        side='right', rod_index=row))
        index += 1

    # Top-side receivers: one per rod column
    for col in range(1, grid.num_rods_x + 1):
        receivers.append(create_receiver(index=index, length=receiver_length,
                                        side='top', rod_index=col))
        index += 1

    # Bottom-side receivers: one per rod column
    for col in range(1, grid.num_rods_x + 1):
        receivers.append(create_receiver(index=index, length=receiver_length,
                                        side='bottom', rod_index=col))
        index += 1

    return receivers


def create_environment(design_region: DesignRegion, grid: Grid, sources: List[Source]= None, receivers: List[Receiver] = None) -> Environment:
    """
    Creates environment from the given physical input objects.

    During training sessions, agent primarily interacts with this environment.

    Inputs:
    --> design_region [DesignRegion]: Primary design regions for all objects
    --> sources [List[Source]]: List of sources in design region
    --> receivers [List[Receiver]]: List of receivers in design region
    --> grid [Grid]: Set of rods in design region

    Returns:
    --> [Environment]: Environment object with specified parameters
    """

    # Convert None to empty lists
    sources = sources if sources is not None else []
    receivers = receivers if receivers is not None else []

    return Environment(design_region=design_region, sources=sources,
                       receivers=receivers, grid=grid)



