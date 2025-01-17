"""
Mercury precession demonstration

Simulates Mercury's orbit with and without post-Newtonian corrections
to demonstrate the famous 43 arcseconds/century precession
"""

import numpy as np
import sys
sys.path.append('..')

from src.simulation import Simulation
from src.config import MERCURY_SYSTEM
from src.visualization import plot_precession
import matplotlib.pyplot as plt


def run_simulation(use_pn, label):
    """Run simulation with or without PN corrections"""
    print(f"\n{'='*60}")
    print(f"Running simulation: {label}")
    print(f"{'='*60}")
    
    sim = Simulation(
        MERCURY_SYSTEM,
        dt=3600 * 2,  # 2 hour timestep
        use_barnes_hut=False,
        use_gpu=False,
        use_pn=use_pn,
        integrator='rk4',
        softening=1e6  # 1000 km softening
    )
    
    # Simulate for 100 years (Mercury orbits ~415 times)
    years = 100
    sim.run(days=365.25 * years, save_every=12, verbose=True)
    
    return sim


def main():
    print("="*60)
    print("Mercury Perihelion Precession Demonstration")
    print("="*60)
    print("\nThis demonstrates how post-Newtonian corrections")
    print("reproduce Mercury's famous 43\"/century precession")
    print()
    
    # Run with Newtonian gravity only
    sim_newtonian = run_simulation(use_pn=False, label="Newtonian Gravity")
    
    # Run with post-Newtonian corrections
    sim_pn = run_simulation(use_pn=True, label="Post-Newtonian (1PN)")
    
    # Plot precession for both
    print("\nAnalyzing precession...")
    
    fig1, angles_newt = plot_precession(sim_newtonian, body_index=1, show=False)
    fig2, angles_pn = plot_precession(sim_pn, body_index=1, show=False)
    
    # Calculate precession rates
    if len(angles_newt) > 2:
        # Fit linear trend
        orbits_newt = np.arange(len(angles_newt))
        rate_newt = np.polyfit(orbits_newt, angles_newt, 1)[0]
        
        orbits_pn = np.arange(len(angles_pn))
        rate_pn = np.polyfit(orbits_pn, angles_pn, 1)[0]
        
        # Mercury has ~415 orbits per century
        orbits_per_century = 415.2
        
        # Convert to arcseconds per century
        precession_newt = rate_newt * orbits_per_century * 3600  # degrees to arcsec
        precession_pn = rate_pn * orbits_per_century * 3600
        
        print("\n" + "="*60)
        print("Precession Results")
        print("="*60)
        print(f"Newtonian:       {precession_newt:8.2f} arcsec/century")
        print(f"Post-Newtonian:  {precession_pn:8.2f} arcsec/century")
        print(f"Difference:      {precession_pn - precession_newt:8.2f} arcsec/century")
        print()
        print("Expected (from GR): ~43 arcsec/century")
        print("(Our 1PN approximation should recover most of this)")
        print("="*60)
    
    # Compare trajectories
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Newtonian trajectory
    traj_newt = sim_newtonian.get_trajectory(1) / 1e9  # to million km
    axes[0].plot(traj_newt[:, 0], traj_newt[:, 1], 'b-', linewidth=0.5, alpha=0.7)
    axes[0].scatter([0], [0], color='yellow', s=200, marker='*', label='Sun')
    axes[0].set_xlabel('X (million km)')
    axes[0].set_ylabel('Y (million km)')
    axes[0].set_title('Newtonian Gravity')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Post-Newtonian trajectory
    traj_pn = sim_pn.get_trajectory(1) / 1e9
    axes[1].plot(traj_pn[:, 0], traj_pn[:, 1], 'r-', linewidth=0.5, alpha=0.7)
    axes[1].scatter([0], [0], color='yellow', s=200, marker='*', label='Sun')
    axes[1].set_xlabel('X (million km)')
    axes[1].set_ylabel('Y (million km)')
    axes[1].set_title('With Post-Newtonian Corrections')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.suptitle('Mercury Orbit Comparison (100 years)', fontsize=14)
    plt.tight_layout()
    
    plt.show()


if __name__ == '__main__':
    main()

# Added comparison with GR prediction
# Shows 43 arcseconds/century precession
# Mercury example
# Mercury example
