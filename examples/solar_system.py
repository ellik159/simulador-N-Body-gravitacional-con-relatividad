"""
Solar system simulation example

Simulates the inner solar system for 10 years and plots trajectories
"""

import numpy as np
import sys
sys.path.append('..')

from src.simulation import Simulation
from src.config import SOLAR_SYSTEM_DATA


def main():
    print("="*60)
    print("Solar System Simulation Example")
    print("="*60)
    
    # Create simulation
    # Using just inner planets + Jupiter for reasonable runtime
    inner_system = SOLAR_SYSTEM_DATA[:6]  # Sun through Jupiter
    
    sim = Simulation(
        inner_system,
        dt=3600 * 6,  # 6 hour timestep
        use_barnes_hut=False,  # direct summation ok for small N
        use_gpu=False,
        use_pn=True,  # include relativistic effects
        integrator='rk4'
    )
    
    # Run for 10 years
    print("\nRunning simulation...")
    sim.run(days=365 * 10, save_every=24)  # save once per day
    
    # Plot results
    print("\nGenerating plots...")
    sim.plot_trajectories(show=False)
    sim.plot_energy(show=True)
    
    # Print some stats
    print("\n" + "="*60)
    print("Simulation Statistics")
    print("="*60)
    for i, name in enumerate(sim.names):
        if name == 'Sun':
            continue
        traj = sim.get_trajectory(i)
        distances = np.linalg.norm(traj, axis=1)
        print(f"{name}:")
        print(f"  Min distance: {distances.min()/1e9:.2f} million km")
        print(f"  Max distance: {distances.max()/1e9:.2f} million km")
        print(f"  Mean distance: {distances.mean()/1e9:.2f} million km")


if __name__ == '__main__':
    main()

# Added Jupiter and Saturn masses
# Improved initial conditions accuracy
# Solar system
# Solar system
