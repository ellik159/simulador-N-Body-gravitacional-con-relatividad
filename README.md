# N-Body Gravitational Simulator 🌌

Started this to understand Mercury's orbit precession but got carried away. Simulates gravitational interactions with relativistic corrections using Barnes-Hut algorithm.

## Quick Start ⚡

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python examples/solar_system.py
```

## What it does

Simulates N-body gravity with Barnes-Hut (O(n log n) instead of O(n²)). Has post-Newtonian corrections for relativity - needed for Mercury's 43"/century precession. GPU acceleration with CuPy if available.

## Features

- Barnes-Hut octree for fast gravity
- Post-Newtonian corrections (1PN)
- GPU acceleration with CuPy 🚀
- 3D visualization
- Energy tracking (kinda works)
- RK4, leapfrog, verlet integrators
- Adaptive timesteps (experimental, buggy)

## Examples

**Solar System**
```python
from src.simulation import Simulation
from src.config import SOLAR_SYSTEM_DATA

sim = Simulation(SOLAR_SYSTEM_DATA, dt=3600)
sim.run(days=3650)  # 10 years
sim.plot_trajectories()
```

**Custom system**
```python
bodies = [
    {"mass": 1e30, "pos": [0, 0, 0], "vel": [0, 0, 0]},
    {"mass": 1e24, "pos": [1e11, 0, 0], "vel": [0, 3e4, 0]}
]
sim = Simulation(bodies)
sim.run(days=100)
```

## Performance

| Bodies | Barnes-Hut (CPU) | Direct (CPU) |
|--------|------------------|--------------|
| 100    | ~0.5s            | ~2s          |
| 1000   | ~8s              | ~180s        |
| 10000  | ~95s             | forever      |

*On i7-9700K, theta=0.5, 1000 steps. GPU gives 5-10x speedup if you have CUDA.*

## TODO (lots)

- [ ] 2PN corrections (second order relativity)
- [ ] Collision detection
- [ ] Fix adaptive timestep - crashes sometimes
- [ ] Parallel tree construction (only force calc is GPU now)
- [ ] More examples (galaxies, clusters)
- [ ] Fix energy drift in leapfrog
- [ ] Actually write documentation
- [ ] Profile and optimize

## Known Issues

- Adaptive timestepping crashes with eccentric orbits
- GPU memory issues with >50k particles
- Visualization slow for >5k bodies
- Energy conservation not perfect (drift higher than should be)
- Some edge cases in tree traversal

## Notes

Barnes-Hut took forever to get right - pointer arithmetic and tree traversal bugs everywhere. Post-Newtonian was another rabbit hole: 5 different conventions for the PN metric, had to pick one and hope it's right.

GPU stuff was easier than expected once I figured out CuPy. It's basically NumPy but on GPU.

Energy conservation is... okay. RK4 drifts, leapfrog is better but has issues with eccentric orbits. Good enough for visualization but not for serious science.

If you find bugs (you will), feel free to open an issue or just fix it yourself.

## References

- Barnes & Hut (1986) - original paper
- "Gravitation" by MTW - PN formalism
- Springel (2005) - GADGET-2 code reference

## License

MIT# update
# cleanup
