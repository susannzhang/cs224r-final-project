from geometry import DesignRegion, Source, Environment, Grid, Rod
import numpy as np
from typing import List
from skimage.draw import line, disk


def _cart_to_array(cart_x: float, cart_y: float, offset_x: int, offset_y: int) -> tuple[int, int]:
    """
    Convert Cartesian (x, y) to numpy array (row, col).
    Row = offset_y - cart_y (y-axis flipped), col = offset_x + cart_x.
    """
    return (int(offset_y - cart_y), int(offset_x + cart_x))


def _array_to_cart(row: int, col: int, offset_x: int, offset_y: int) -> tuple[float, float]:
    """
    Convert numpy array (row, col) to Cartesian (x, y).
    """
    return (col - offset_x, offset_y - row)


def _convert_design_region_to_cells(env: Environment) -> None:
    """
    Initializes the design region canvas with size derived from grid dimensions plus margins.

    Canvas size = grid_size + 2 * margin_cells on each axis.
    All cells are initialized to background permittivity.

    Inputs:
    --> env [Environment]: Environment with design region and grid

    Returns:
    --> None: Initializes env.design_region._canvas in place
    """
    grid_x, grid_y = _convert_grid_to_cells(env=env)  # [cells, cells]
    margin = env.design_region.margin_cells  # [cells]
    bg_perm = env.design_region.bg_permittivity  # [unitless]

    cells_x = grid_x + 2 * margin  # [cells]
    cells_y = grid_y + 2 * margin  # [cells]

    env.design_region._canvas = np.full((cells_x, cells_y), bg_perm)


def _convert_grid_to_cells(env: Environment) -> tuple[int, int]:
    """
    Calculates the amount of space occupied by a grid in cells.

    The function's primary objective is to convert the metric unit-based grid
    into a cell unit-based grid for easy of reference and internal referencing. 

    Coordinate System:
    --> The input grid uses the Cartesian coordinate system. Since this function's 
        primary objective is mere unit conversion, we do not alter the reference frame 
        in any way. 

    Inputs:
    --> env [Environment]: The environment where the grid lives [unitless]

    Returns:
    --> tuple[int, int]: The number of cells the grid occupies across [x, y] axes respectively [cells]
    """
    # Pull the variable values
    grid = env.grid                             # [unitless]
    diameter = 2 * grid.radius                  # [meters]
    spacing = grid.distance                     # [meters]
    resolution = env.design_region.resolution   # [meters]

    # Total space occupied by the grid is found as follows:
    # N_x * diameter + (N_x - 1) * spacing [meters] 
    # Divide by resolution to get the distance in terms of cells
    grid_x = int((grid.num_rods_x * diameter + (grid.num_rods_x - 1) * spacing) / resolution)  # [cells]
    grid_y = int((grid.num_rods_y * diameter + (grid.num_rods_y - 1) * spacing) / resolution) # [cells]

    # Return truncated grid size
    return (grid_x, grid_y)


def _convert_sources_to_cells(env: Environment) -> None:
    """
    Converts all sources into boolean masks for Ceviche.

    Sources are placed at a fixed position: left side of the grid, centered vertically.
    Source is a vertical line of length source.length.

    Inputs:
    --> env [Environment]: Environment instance where sources live

    Returns:
    --> None: Saves source masks inside source objects for future access.
    """
    if not env.sources:
        return

    canvas = env.design_region._canvas
    resolution = env.design_region.resolution  # [meters]
    margin = env.design_region.margin_cells  # [cells]
    offset_x = canvas.shape[0] // 2  # [cells]
    offset_y = canvas.shape[1] // 2  # [cells]

    for source in env.sources:
        mask = np.zeros_like(canvas)

        # Source position: halfway into left margin, centered vertically
        # In array coords: col = margin // 2, row = offset_y
        # In Cartesian coords (relative to offset): x = (margin//2) - offset_x, y = 0
        cart_center_x = (margin // 2) - offset_x  # [cells]
        cart_center_y = 0  # [cells]

        # Source is a vertical line of given length
        half_len = int((source.length / 2) / resolution)  # [cells]
        cart_y1 = cart_center_y - half_len
        cart_y2 = cart_center_y + half_len
        cart_x1 = cart_center_x
        cart_x2 = cart_center_x

        # Convert Cartesian to array coordinates
        row_1, col_1 = _cart_to_array(cart_x1, cart_y1, offset_x, offset_y)
        row_2, col_2 = _cart_to_array(cart_x2, cart_y2, offset_x, offset_y)

        # Stamp source as a vertical line
        row, col = line(row_1, col_1, row_2, col_2)
        mask[row, col] = 1
        source._mask = mask


def _stamp_source_walls_to_design_region(env: Environment) -> None:
    """
    Stamps reflective wall plates around each source onto the design region.

    For the fixed left-side source: two horizontal walls (top and bottom) extend
    rightward from source endpoints toward the grid, plus a back wall to the left
    of the source to prevent backward radiation.

    Inputs:
    --> env [Environment]: Environment instance where sources and design region live.

    Returns:
    --> None: Alters design_region._canvas in place by stamping wall permittivity.
    """
    if not env.sources:
        return

    canvas = env.design_region._canvas
    plate_perm = env.design_region.plate_permittivity  # [unitless]
    margin = env.design_region.margin_cells  # [cells]
    resolution = env.design_region.resolution  # [meters]
    offset_x = canvas.shape[0] // 2  # [cells]
    offset_y = canvas.shape[1] // 2  # [cells]

    for source in env.sources:
        if not source.walls:
            continue

        # Source position
        cart_center_x = (margin // 2) - offset_x
        cart_center_y = 0

        # Source vertical extent
        half_len = int((source.length / 2) / resolution)  # [cells]

        # Back wall: to the left of source center
        back_wall_x = cart_center_x - source.back_wall_offset

        # Top wall: from back wall leftward, extending rightward to just past source.
        # Negative cart_y here because _cart_to_array flips y (row = offset_y - cart_y),
        # so cart_y < 0 lands at LARGE row → visual top of imshow(origin='lower').
        cart_y_top = -half_len
        cart_x_wall_end = cart_center_x + 3  # extend a bit past source to avoid gap, but don't overlap rods
        row_top_start, col_top_start = _cart_to_array(back_wall_x, cart_y_top, offset_x, offset_y)
        row_top_end, col_top_end = _cart_to_array(cart_x_wall_end, cart_y_top, offset_x, offset_y)
        row, col = line(row_top_start, col_top_start, row_top_end, col_top_end)
        canvas[row, col] = plate_perm

        # Bottom wall: positive cart_y → small row → visual bottom.
        cart_y_bottom = half_len
        row_bot_start, col_bot_start = _cart_to_array(back_wall_x, cart_y_bottom, offset_x, offset_y)
        row_bot_end, col_bot_end = _cart_to_array(cart_x_wall_end, cart_y_bottom, offset_x, offset_y)
        row, col = line(row_bot_start, col_bot_start, row_bot_end, col_bot_end)
        canvas[row, col] = plate_perm

        # Back wall: vertical wall to the left of source, connecting top and bottom walls.
        row_back_top, col_back_top = _cart_to_array(back_wall_x, cart_y_top, offset_x, offset_y)
        row_back_bot, col_back_bot = _cart_to_array(back_wall_x, cart_y_bottom, offset_x, offset_y)
        row, col = line(row_back_top, col_back_top, row_back_bot, col_back_bot)
        canvas[row, col] = plate_perm


def _convert_receivers_to_cells(env: Environment) -> None:
    """
    Converts receivers into boolean masks for Ceviche.

    Receivers are placed on right, top, or bottom sides of the grid, with each
    receiver centered on the rod it is aligned with:
      - side='right'  → vertical receiver at the same array row as rods with grid x = rod_index
      - side='top'    → horizontal receiver at the same array col as rods with grid y = rod_index
      - side='bottom' → horizontal receiver at the same array col as rods with grid y = rod_index

    rod_index is 1-based, matching the grid convention where bottom-left rod is [1, 1].

    Inputs:
    --> env [Environment]: Environment instance where receivers live (requires
                           env.grid.rods to be populated by _stamp_grid_onto_design_region)

    Returns:
    --> None: Saves receiver masks inside receiver objects for future access.
    """
    if not env.receivers:
        return

    canvas = env.design_region._canvas
    resolution = env.design_region.resolution  # [meters/cell]
    margin = env.design_region.margin_cells    # [cells]
    n_rows, n_cols = canvas.shape

    for receiver in env.receivers:
        mask = np.zeros_like(canvas)
        half_len = int((receiver.length / 2) / resolution)  # [cells]

        if receiver.side == 'right':
            # Vertical receiver near the right edge, centered on the rod's row.
            # Any rod with grid x = rod_index has the same array row, so use (rod_index, 1).
            ref_rod = env.grid.rods[(receiver.rod_index, 1)]
            row_center = ref_rod._center[0]
            col_position = n_cols - (margin // 2)
            row_a, col_a = row_center - half_len, col_position
            row_b, col_b = row_center + half_len, col_position

        elif receiver.side == 'top':
            # Horizontal receiver near the visual TOP of the displayed image.
            # With imshow(origin='lower'), large array row → top of display.
            ref_rod = env.grid.rods[(1, receiver.rod_index)]
            col_center = ref_rod._center[1]
            row_position = n_rows - (margin // 2)
            row_a, col_a = row_position, col_center - half_len
            row_b, col_b = row_position, col_center + half_len

        elif receiver.side == 'bottom':
            # Horizontal receiver near the visual BOTTOM of the displayed image.
            # With imshow(origin='lower'), small array row → bottom of display.
            ref_rod = env.grid.rods[(1, receiver.rod_index)]
            col_center = ref_rod._center[1]
            row_position = margin // 2
            row_a, col_a = row_position, col_center - half_len
            row_b, col_b = row_position, col_center + half_len

        else:
            raise ValueError(f"Unknown receiver side: {receiver.side!r}. "
                             f"Expected 'right', 'top', or 'bottom'.")

        row, col = line(row_a, col_a, row_b, col_b)
        mask[row, col] = 1
        receiver._mask = mask



def _find_center_for_first_rod_in_grid_cells(env: Environment) -> tuple[int, int]: 
    """
    Helper function to find the center of the bottom-left most rod in the grid [1,1]. 
    Finds the center in cells, returning the coordinates in the design region. 

    --> Assumes that the design region is initialized and discretized into cells. 
    --> Operates directly on the design region, using ndarray coordinates. 

    Inputs:
    --> env [Environment]: Environment instance where design region resides. 

    Returns:
    --> tuple[int, int]: Coordinates for the center of first rod [in ndarray cells].
    """
    # Convert rod radius from metric to simulation units
    # This point represents the origin for the rest of the calculations
    rod_radius = env.grid.radius / env.design_region.resolution # [meters / [meters/cell]] = [cells]

    # Find the center of the design region for reference 
    origin_x = env.design_region._canvas.shape[0] // 2  # [cells]
    origin_y = env.design_region._canvas.shape[1] // 2  # [cells]

    # In the np.ndarray, rows increase downward and cols increase rightward.
    # With origin='lower' display, smaller row = lower on screen.
    # Bottom-left rod (grid index [1,1]) should be at small row AND small col.
    grid_x, grid_y = _convert_grid_to_cells(env=env) # [cells, cells]
    rod_x = origin_x - (grid_x / 2 - rod_radius) # [cells]  (small row → bottom)
    rod_y = origin_y - (grid_y / 2 - rod_radius) # [cells]  (small col → left)

    return (int(rod_x), int(rod_y)) # [cells, cells]



def _stamp_rod_to_design_region(rod: Rod, env: Environment) -> None: 
    """
    Stamps a rod onto an existing design region at the given center, in place. 

    Coordinate System:
    --> rod._center is given in ndarray coordinates. The rod can be stamped directly
        at the given coordinates on the design region without further conversions. 

    Important Notes:
    --> Assumes that a design region and rod object both already exists and are 
        instantiated/initialized. 

    Inputs:
    --> rod [Rod]: Rod object to be stamped onto the design region
    --> env [Environment]: Environment in which the design region lives

    Returns:
    --> None: Alters the design_region._canvas in place
    """
    # Convert radius from metric to cells 
    radius = env.grid.radius / env.design_region.resolution # [cells]
    
    # Get rod center coordinates from _center, in array coordinates
    row, col = disk(center=rod._center, radius=radius, shape=env.design_region._canvas.shape)

    # Stamp the rod onto the existing design region
    env.design_region._canvas[row, col] = rod.permittivity


def _stamp_grid_onto_design_region(env: Environment) -> None:
    """
    Stamps a full grid of rods onto design region, in place. 

    Coordinate System:
    --> Exclusively uses ndarray Coodinates, where the top-left most pixel is [0, 0] and the coordinates increase to the right and down in the array. 

    Important Notes:
    --> Assumes that the design region and grid are already defined/initialized. 
    --> Function alters the design region in place.
    --> Function assumes that user has passed in a square grid.
    --> The rod counts increase in +x and +y directions, bottom-left rod is indexed [1,1]
    --> The function implements the grid onto a design region by reference, in place. 

    Inputs:
    --> env [Environment]: Environmnet where the design region and grid lives. 

    Returns:
    --> None: Alters design region in place by stamping the rods as permittivity values onto the design region. 
    """
    first_rod_x, first_rod_y = _find_center_for_first_rod_in_grid_cells(env=env) # [cells, cells]
    radius = env.grid.radius         # [meters]
    distance = env.grid.distance     # [meters]
    resolution = env.design_region.resolution # [meters/cell]

    # Distance between two rods: 2 * radius + distance between rod walls
    shift = int((2 * radius + distance) / resolution) # [cells]
    
    # Generate the grid, starting from the bottom-left-most rod, indexed [1, 1]
    for x in range(env.grid.num_rods_x):
        for y in range(env.grid.num_rods_y): 
            # Save the rod information into Rod object
            # Indexing starts at [0, 0], add +1 to shift to [1, 1]
            # Array coordinates increase right and down
            # To find the centers for next rods, increase in x and decrease in y
            #   (going from bottom-left-most to top-right-most rod)
            this_rod = Rod(index=(x+1, y+1), permittivity=env.grid.rod_permittivity) 
            this_rod._center = (first_rod_x + x * shift, first_rod_y + y * shift)

            # Save rod info to grid for future access
            env.grid.rods[x+1, y+1] = this_rod

            # Stamp the rod permittivity onto design region in place 
            _stamp_rod_to_design_region(this_rod, env) 
    