"""pytest collection hook for the dynamic_beam_steering subpackage.

Tests under dynamic_beam_steering/tests/ import the moved algorithms package
as `from algorithms.policies... import ...`. For that to resolve, this
directory itself must be on sys.path during test collection. The root
conftest.py covers the project-root level (so `from geometry import ...`
and `from simulation import ...` keep working); this one covers the
beam-steering local imports.
"""

import sys
from pathlib import Path

_DBS_ROOT = Path(__file__).resolve().parent
if str(_DBS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DBS_ROOT))
