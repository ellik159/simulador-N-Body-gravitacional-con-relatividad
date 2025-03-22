# Quick Start Guide

## Installation

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: CuPy requires CUDA toolkit. If you don't have a GPU, the code will automatically fall back to NumPy.

## Running Examples

### Solar System Simulation
```bash
python examples/solar_system.py
```

Simulates the inner solar system (Sun through Saturn) for 10 years and plots trajectories.

### Mercury Precession
```bash
python examples/mercury_precession.py
```

Demonstrates Mercury's perihelion precession with and without post-Newtonian corrections. Shows the famous 43 arcseconds/century effect.

### Performance Benchmark
```bash
python examples/benchmark.py
```

Compares different methods (direct, Barnes-Hut, GPU) across various system sizes.

## Running Tests

```bash
pytest tests/ -v
```

## Basic Usage

```python
from src.simulation import Simulation

# Define bodies
bodies = [
    {
        'mass': 1e30,  # kg
        'pos': [0, 0, 0],  # m
        'vel': [0, 0, 0],  # m/s
        'name': 'Star',
        'color': 'yellow'
    },
    {
        'mass': 1e24,
        'pos': [1e11, 0, 0],
        'vel': [0, 3e4, 0],
        'name': 'Planet',
        'color': 'blue'
    }
]

# Create simulation
sim = Simulation(
    bodies,
    dt=3600,  # 1 hour timestep
    use_barnes_hut=True,
    use_pn=True,
    integrator='rk4'
)

# Run simulation
sim.run(days=365)  # 1 year

# Plot results
sim.plot_trajectories()
sim.plot_energy()
```

## Configuration

Key parameters in `src/config.py`:
- `G`: Gravitational constant
- `C`: Speed of light
- `DEFAULT_THETA`: Barnes-Hut opening angle (0.5 is good default)
- `DEFAULT_DT`: Default timestep

## Troubleshooting

**GPU not working**: Install CUDA toolkit or disable GPU with `use_gpu=False`

**Energy not conserved**: Try smaller timestep or use leapfrog/verlet integrator

**Simulation too slow**: Increase theta parameter or use GPU acceleration

**Crashes with many bodies**: Reduce timestep or use Barnes-Hut algorithm
