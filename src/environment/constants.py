"""
System constants for the EM training pipeline.
Physical constants, grid resolution, and default simulation parameters.
"""

import numpy as np

# ============================================================
# Physical Constants
# ============================================================
EPSILON_0 = 8.85418782e-12   # F/m, vacuum permittivity
MU_0 = 1.25663706e-6         # H/m, vacuum permeability
C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)  # m/s, speed of light

# ============================================================
# Grid / Simulation Defaults (override these as needed)
# ============================================================
DL = None       # meters per grid cell -- set by user during function call
NPML = None     # number of PML cells on each side — set by user during function call
