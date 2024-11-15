"""
Main simulation class and integration routines
"""

import numpy as np
from tqdm import tqdm
import time

from src.physics import (
    compute_accelerations_direct,
    compute_accelerations_gpu,
    compute_energy,
    compute_momentum,
    compute_angular_momentum
)
from src.barnes_hut import build_tree
from src.config import (
    DEFAULT_DT,
    DEFAULT_THETA,
    DEFAULT_SOFTENING,
    DEFAULT_INTEGRATOR,
    DAY,
    G
)


class Simulation:
    """
    Main N-body simulation class
    
    Manages the state of the system and runs the time integration
    """
    
    def __init__(self, bodies, dt=DEFAULT_DT, theta=DEFAULT_THETA,
                 use_barnes_hut=True, use_gpu=False, use_pn=True,
                 integrator=DEFAULT_INTEGRATOR, softening=DEFAULT_SOFTENING):
        """
        Initialize simulation
        
        Args:
            bodies: list of dicts with 'mass', 'pos', 'vel' keys
            dt: timestep in seconds
            theta: Barnes-Hut opening angle
            use_barnes_hut: whether to use BH tree (vs direct summation)
            use_gpu: whether to use GPU acceleration
            use_pn: whether to include post-Newtonian corrections
            integrator: 'rk4', 'leapfrog', or 'verlet'
            softening: softening parameter
        """
        self.n_bodies = len(bodies)
        self.dt = dt
        self.theta = theta
        self.use_barnes_hut = use_barnes_hut
        self.use_gpu = use_gpu
        self.use_pn = use_pn
        self.integrator = integrator
        self.softening = softening
        
        # Extract initial conditions
        self.masses = np.array([b['mass'] for b in bodies], dtype=np.float64)
        self.positions = np.array([b['pos'] for b in bodies], dtype=np.float64)
        self.velocities = np.array([b['vel'] for b in bodies], dtype=np.float64)
        
        # Store body names and colors if available
        self.names = [b.get('name', f'Body {i}') for i, b in enumerate(bodies)]
        self.colors = [b.get('color', 'blue') for b in bodies]
        
        # History tracking
        self.time = 0.0
        self.step_count = 0
        self.position_history = [self.positions.copy()]
        self.velocity_history = [self.velocities.copy()]
        self.time_history = [0.0]
        self.energy_history = []
        
        # Compute initial energy
        E0 = compute_energy(self.positions, self.velocities, self.masses)
        self.energy_history.append(E0)
        self.initial_energy = E0
        
        print(f"Initialized simulation with {self.n_bodies} bodies")
        print(f"  Integrator: {self.integrator}")
        print(f"  Barnes-Hut: {self.use_barnes_hut} (theta={self.theta})")
        print(f"  GPU: {self.use_gpu}")
        print(f"  Post-Newtonian: {self.use_pn}")
        print(f"  Initial energy: {E0:.3e} J")
        
    def compute_accelerations(self):
        """
        Compute accelerations for all bodies
        
        Uses Barnes-Hut tree if enabled, otherwise direct summation
        """
        if self.use_barnes_hut:
            return self._compute_accelerations_bh()
        elif self.use_gpu:
            return compute_accelerations_gpu(
                self.positions, self.velocities, self.masses,
                use_pn=self.use_pn, softening=self.softening
            )
        else:
            return compute_accelerations_direct(
                self.positions, self.velocities, self.masses,
                use_pn=self.use_pn, softening=self.softening
            )
    
    def _compute_accelerations_bh(self):
        """Compute accelerations using Barnes-Hut tree"""
        # Build tree
        tree = build_tree(self.positions, self.masses, theta=self.theta)
        
        # Compute forces
        accel = np.zeros_like(self.positions)
        for i in range(self.n_bodies):
            force_per_mass = tree.compute_force(i, softening=self.softening)
            accel[i] = G * force_per_mass
        
        # Add post-Newtonian corrections if enabled
        # TODO: implement PN in Barnes-Hut tree
        if self.use_pn:
            accel += self._compute_pn_corrections()
        
        return accel
    
    def _compute_pn_corrections(self):
        """Compute post-Newtonian corrections separately"""
        from src.physics import post_newtonian_correction
        
        accel = np.zeros_like(self.positions)
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i == j:
                    continue
                pn_corr = post_newtonian_correction(
                    self.positions[i], self.positions[j],
                    self.velocities[i], self.velocities[j],
                    self.masses[i], self.masses[j]
                )
                accel[i] += pn_corr
        
        return accel
    
    def step_rk4(self):
        """4th order Runge-Kutta integration step"""
        dt = self.dt
        
        # k1
        a1 = self.compute_accelerations()
        k1_v = dt * a1
        k1_r = dt * self.velocities
        
        # k2
        old_pos = self.positions.copy()
        old_vel = self.velocities.copy()
        self.positions = old_pos + 0.5 * k1_r
        self.velocities = old_vel + 0.5 * k1_v
        a2 = self.compute_accelerations()
        k2_v = dt * a2
        k2_r = dt * self.velocities
        
        # k3
        self.positions = old_pos + 0.5 * k2_r
        self.velocities = old_vel + 0.5 * k2_v
        a3 = self.compute_accelerations()
        k3_v = dt * a3
        k3_r = dt * self.velocities
        
        # k4
        self.positions = old_pos + k3_r
        self.velocities = old_vel + k3_v
        a4 = self.compute_accelerations()
        k4_v = dt * a4
        k4_r = dt * self.velocities
        
        # Final update
        self.positions = old_pos + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6.0
        self.velocities = old_vel + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0
    
    def step_leapfrog(self):
        """Leapfrog integration (symplectic)"""
        # Kick-drift-kick formulation
        dt = self.dt
        
        # Half kick
        accel = self.compute_accelerations()
        self.velocities += 0.5 * dt * accel
        
        # Drift
        self.positions += dt * self.velocities
        
        # Half kick
        accel = self.compute_accelerations()
        self.velocities += 0.5 * dt * accel
    
    def step_verlet(self):
        """Velocity Verlet integration"""
        dt = self.dt
        
        # Update positions
        accel = self.compute_accelerations()
        self.positions += self.velocities * dt + 0.5 * accel * dt**2
        
        # Update velocities
        accel_new = self.compute_accelerations()
        self.velocities += 0.5 * (accel + accel_new) * dt
    
    def step(self):
        """Perform one integration step"""
        if self.integrator == 'rk4':
            self.step_rk4()
        elif self.integrator == 'leapfrog':
            self.step_leapfrog()
        elif self.integrator == 'verlet':
            self.step_verlet()
        else:
            raise ValueError(f"Unknown integrator: {self.integrator}")
        
        self.time += self.dt
        self.step_count += 1
    
    def run(self, n_steps=None, days=None, save_every=1, verbose=True):
        """
        Run simulation for specified number of steps or days
        
        Args:
            n_steps: number of timesteps to run
            days: number of days to simulate (alternative to n_steps)
            save_every: save state every N steps
            verbose: show progress bar
        """
        if days is not None:
            n_steps = int(days * DAY / self.dt)
        
        if n_steps is None:
            raise ValueError("Must specify either n_steps or days")
        
        iterator = tqdm(range(n_steps)) if verbose else range(n_steps)
        start_time = time.time()
        
        for i in iterator:
            self.step()
            
            if (i + 1) % save_every == 0:
                self.position_history.append(self.positions.copy())
                self.velocity_history.append(self.velocities.copy())
                self.time_history.append(self.time)
                
                # Track energy
                E = compute_energy(self.positions, self.velocities, self.masses)
                self.energy_history.append(E)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\nSimulation complete!")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Steps: {n_steps}")
            print(f"  Steps/sec: {n_steps/elapsed:.1f}")
            
            # Energy drift
            dE = (self.energy_history[-1] - self.initial_energy) / abs(self.initial_energy)
            print(f"  Energy drift: {dE*100:.4f}%")
    
    def get_trajectory(self, body_index):
        """Get trajectory of a specific body"""
        traj = np.array([pos[body_index] for pos in self.position_history])
        return traj
    
    def plot_trajectories(self, show=True):
        """Plot 3D trajectories of all bodies"""
        from src.visualization import plot_trajectories_3d
        plot_trajectories_3d(self, show=show)
    
    def plot_energy(self, show=True):
        """Plot energy conservation"""
        from src.visualization import plot_energy_conservation
        plot_energy_conservation(self, show=show)
    
    def animate(self, interval=50, trail_length=100):
        """Create animation of the simulation"""
        from src.visualization import create_animation
        return create_animation(self, interval=interval, trail_length=trail_length)
# Small improvement
# implement PN corrections
# fix PN signs
# mercury precession example
# add gpu support
# fix gpu memory
# add visualization
# add benchmark
# add leapfrog and verlet
# add energy tracking
# update readme
# fix energy drift
# cleanup
# final

# Main simulation class
# Handles time integration and particle management

# RK4 integrator implementation
# More accurate than Euler but slower
# Good for testing accuracy vs performance

# Symplectic integrators
# Leapfrog and Velocity Verlet for better energy conservation

# Fixed energy drift in long simulations
# Adjusted time step adaptation algorithm
# RK4 added
# Leapfrog verlet
# Energy drift fix
# Simulation class
# RK4 integrator
# Leapfrog verlet
# Fix energy drift
