"""
N-Body Gravitational Simulator with Post-Newtonian Corrections

A physics simulation engine implementing Barnes-Hut algorithm for
efficient O(n log n) gravity calculations with relativistic corrections.
"""

__version__ = '0.1.0'
__author__ = 'Mario Perez'

from src.simulation import Simulation
from src.config import (
    SOLAR_SYSTEM_DATA,
    MERCURY_SYSTEM,
    G, C, AU, SOLAR_MASS
)

__all__ = [
    'Simulation',
    'SOLAR_SYSTEM_DATA',
    'MERCURY_SYSTEM',
    'G', 'C', 'AU', 'SOLAR_MASS'
]

# Cleaned up imports
# Removed unused variables
# Fixed some typos in comments
# Cleanup
# Cleanup
