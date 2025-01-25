"""
Configuration and constants for N-body simulation
"""

import numpy as np

# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
C = 299792458.0   # Speed of light (m/s)
AU = 1.496e11     # Astronomical unit (m)
SOLAR_MASS = 1.989e30  # kg
EARTH_MASS = 5.972e24  # kg
DAY = 86400.0     # seconds

# Simulation parameters
DEFAULT_THETA = 0.5  # Barnes-Hut opening angle
DEFAULT_SOFTENING = 1e3  # Softening length (m) to avoid singularities
DEFAULT_DT = 3600.0  # Default timestep (1 hour in seconds)

# Integrator options
INTEGRATORS = ['rk4', 'leapfrog', 'verlet']
DEFAULT_INTEGRATOR = 'rk4'

# Solar system data (simplified, from NASA Horizons)
# positions in AU, velocities in AU/day, masses in solar masses
SOLAR_SYSTEM_DATA = [
    {
        'name': 'Sun',
        'mass': 1.0 * SOLAR_MASS,
        'pos': np.array([0.0, 0.0, 0.0]) * AU,
        'vel': np.array([0.0, 0.0, 0.0]) * AU / DAY,
        'color': 'yellow'
    },
    {
        'name': 'Mercury',
        'mass': 0.330e24,
        'pos': np.array([0.387, 0.0, 0.0]) * AU,
        'vel': np.array([0.0, 47.87e3, 0.0]),
        'color': 'gray'
    },
    {
        'name': 'Venus',
        'mass': 4.867e24,
        'pos': np.array([0.723, 0.0, 0.0]) * AU,
        'vel': np.array([0.0, 35.02e3, 0.0]),
        'color': 'orange'
    },
    {
        'name': 'Earth',
        'mass': EARTH_MASS,
        'pos': np.array([1.0, 0.0, 0.0]) * AU,
        'vel': np.array([0.0, 29.78e3, 0.0]),
        'color': 'blue'
    },
    {
        'name': 'Mars',
        'mass': 0.642e24,
        'pos': np.array([1.524, 0.0, 0.0]) * AU,
        'vel': np.array([0.0, 24.07e3, 0.0]),
        'color': 'red'
    },
    {
        'name': 'Jupiter',
        'mass': 1898e24,
        'pos': np.array([5.203, 0.0, 0.0]) * AU,
        'vel': np.array([0.0, 13.07e3, 0.0]),
        'color': 'brown'
    },
    {
        'name': 'Saturn',
        'mass': 568e24,
        'pos': np.array([9.537, 0.0, 0.0]) * AU,
        'vel': np.array([0.0, 9.69e3, 0.0]),
        'color': 'gold'
    },
]

# Mercury-only system for precession studies
MERCURY_SYSTEM = [
    {
        'name': 'Sun',
        'mass': SOLAR_MASS,
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'color': 'yellow'
    },
    {
        'name': 'Mercury',
        'mass': 0.330e24,
        'pos': np.array([0.387 * AU, 0.0, 0.0]),
        'vel': np.array([0.0, 47.87e3, 0.0]),
        'color': 'gray'
    },
]

# Visualization settings
VIZ_DPI = 100
VIZ_FIGSIZE = (12, 10)
VIZ_TRAIL_LENGTH = 500  # number of points to show in trail

# GPU settings
USE_GPU = True  # Try to use GPU if available
GPU_BATCH_SIZE = 1024  # Process this many bodies at a time on GPU
# gpu
