"""
Tests for main simulation class
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from src.simulation import Simulation
from src.config import MERCURY_SYSTEM, SOLAR_MASS, AU


def test_simulation_initialization():
    """Test basic simulation setup"""
    bodies = [
        {'mass': 1e30, 'pos': np.array([0, 0, 0]), 'vel': np.array([0, 0, 0])},
        {'mass': 1e24, 'pos': np.array([1e11, 0, 0]), 'vel': np.array([0, 3e4, 0])}
    ]
    
    sim = Simulation(bodies, dt=3600)
    
    assert sim.n_bodies == 2
    assert sim.dt == 3600
    assert sim.time == 0.0
    assert len(sim.position_history) == 1


def test_single_step():
    """Test that simulation can take a step"""
    bodies = MERCURY_SYSTEM
    sim = Simulation(bodies, dt=3600, use_barnes_hut=False, use_pn=False)
    
    initial_pos = sim.positions.copy()
    sim.step()
    
    # Position should have changed
    assert not np.allclose(sim.positions, initial_pos)
    assert sim.time == 3600
    assert sim.step_count == 1


def test_run_simulation():
    """Test running simulation for multiple steps"""
    bodies = MERCURY_SYSTEM
    sim = Simulation(bodies, dt=3600, use_barnes_hut=False)
    
    sim.run(n_steps=10, save_every=1, verbose=False)
    
    assert sim.step_count == 10
    assert len(sim.position_history) == 11  # initial + 10 steps


def test_energy_tracking():
    """Test that energy is tracked"""
    bodies = MERCURY_SYSTEM
    sim = Simulation(bodies, dt=3600, use_barnes_hut=False)
    
    sim.run(n_steps=100, save_every=10, verbose=False)
    
    # Should have energy history
    assert len(sim.energy_history) > 0
    
    # Energy should be approximately conserved (within ~1% for short run)
    dE = abs(sim.energy_history[-1] - sim.initial_energy) / abs(sim.initial_energy)
    assert dE < 0.05  # 5% tolerance


def test_different_integrators():
    """Test that different integrators work"""
    bodies = MERCURY_SYSTEM
    
    for integrator in ['rk4', 'leapfrog', 'verlet']:
        sim = Simulation(bodies, dt=3600, integrator=integrator, 
                        use_barnes_hut=False)
        sim.run(n_steps=10, verbose=False)
        
        assert sim.step_count == 10


def test_barnes_hut_vs_direct():
    """Test that Barnes-Hut gives similar results to direct"""
    bodies = [
        {'mass': 1e30, 'pos': np.array([0, 0, 0]), 'vel': np.array([0, 0, 0])},
        {'mass': 1e24, 'pos': np.array([1e11, 0, 0]), 'vel': np.array([0, 3e4, 0])},
        {'mass': 1e24, 'pos': np.array([0, 1.5e11, 0]), 'vel': np.array([-2e4, 0, 0])}
    ]
    
    # Direct summation
    sim_direct = Simulation(bodies, dt=3600, use_barnes_hut=False, use_pn=False)
    sim_direct.run(n_steps=10, verbose=False)
    
    # Barnes-Hut
    sim_bh = Simulation(bodies, dt=3600, use_barnes_hut=True, 
                       theta=0.3, use_pn=False)
    sim_bh.run(n_steps=10, verbose=False)
    
    # Results should be similar (within 10% due to approximation)
    pos_diff = np.linalg.norm(sim_direct.positions - sim_bh.positions)
    pos_scale = np.linalg.norm(sim_direct.positions)
    
    assert pos_diff / pos_scale < 0.1


def test_get_trajectory():
    """Test trajectory extraction"""
    bodies = MERCURY_SYSTEM
    sim = Simulation(bodies, dt=3600, use_barnes_hut=False)
    
    sim.run(n_steps=20, save_every=5, verbose=False)
    
    traj = sim.get_trajectory(1)  # Mercury
    
    assert traj.shape == (5, 3)  # 5 saved positions, 3D


def test_mercury_orbits():
    """Test that Mercury actually orbits the Sun"""
    bodies = MERCURY_SYSTEM
    sim = Simulation(bodies, dt=3600*2, use_barnes_hut=False, use_pn=False)
    
    # Run for ~1 Mercury orbit (~88 days)
    sim.run(days=88, save_every=24, verbose=False)
    
    traj = sim.get_trajectory(1)
    
    # Mercury should return close to starting position
    # (won't be exact due to numerical error and discretization)
    initial_pos = traj[0]
    final_pos = traj[-1]
    
    dist_initial = np.linalg.norm(initial_pos)
    dist_final = np.linalg.norm(final_pos)
    
    # Distance from Sun should be similar
    assert abs(dist_final - dist_initial) / dist_initial < 0.2  # 20% tolerance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
