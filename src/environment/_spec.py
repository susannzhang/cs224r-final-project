"""
Spring 2026, Selin Ertan, Susan Zhang -- 
Automated electomagnetic field configuration design module specifications.
See docs/architecture.tex for detailed documentation.
TODO: Create this documentation separately. 
"""

from geometry import DesignRegion

##################################
### Constants module functions ###
##################################

# No functions here, we will just define our system constants. 


#################################
### Geometry module functions ###
#################################

### TODO: Define Location object
### Dependency trees:
### create_rod() needs -> Location (rod)
### create_grid() needs -> create_rod() 
### create_source() needs -> Location (source)
### create_receiver() needs -> Location (source)
### create_design_region() needs -> create_grid() + create_source() + create_receiver()

### TODO: need function voltage --> permittivity

def create_rod(center: tuple, radius: float, permittivity: float, 
               design_region: DesignRegion, rod_index: tuple = None) -> None:
    """
    Creates a single rod by stamping a circle of the given permittivity
    into the design region's permittivity grid. Modifies in place.
    """
    pass

def create_grid(num_rods_x: int, num_rods_y: int, rod_permittivity: float, 
                rod_radius: float, rod_spacing: float, design_region: DesignRegion) -> DesignRegion: 
    """
    Creates a set x-by-y rods in the design space of the set permittivity.
    Modifies design_region.permittivity_grid in place.
    """
    pass

def create_source():
    # Creates a single source
    raise NotImplementedError

def create_receiver():
    # Creates a receiver
    raise NotImplementedError

def create_design_region(x_length: float, y_length: float, permittivity: float, 
                         grid_cell_dL: float, num_pml_cells: int) -> DesignRegion:
    """
    Initializes an empty design region with the set background permittivity.
    Reminder: Permittivity of vacuum is 1.0.
    """
    pass

def create_design_config():
    # Creates design configuration consisting of a design region, source, and receiver



###################################
### Simulation module functions ###
###################################

### simulate_e_field() needs -> create_design_region()

def simulate_e_field():
    # Simulates the electric field strength 
    raise NotImplementedError

def simulate_h_field():
    # Simulates the magnetic field strength




####################################
### Measurement module functions ###
####################################

def get_total_e_field():
    # Get total E field strength (sum of E fields in all rods)
    raise NotImplementedError  

def get_grid_e_field():
    # Get E field strength values from all rods in the grid
    raise NotImplementedError

def get_grid_permittivity(): 
    # Get permittivity values from all rods in the grid
    raise NotImplementedError

def get_rod_e_field():
    # Get E field strength at a specific rod
    raise NotImplementedError

def get_rod_permittivity():
    # Get the permittivity value of a a single rod 
    raise NotImplementedError

def get_source_signal():
    # Get measurement for the source signal
    raise NotImplementedError

def get_receiver_signal():
    # Get measurement for the received signal
    raise NotImplementedError



################################
### Rewards module functions ###
################################

def calculate_steering_reward():
    # Calculate the beam steering reward at a set angle
    raise NotImplementedError

def calculate_demultiplexing_reward():
    raise NotImplementedError



