"""
Visualization utilities for N-body simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from src.config import VIZ_DPI, VIZ_FIGSIZE, AU


def plot_trajectories_3d(simulation, show=True, save_path=None):
    """
    Plot 3D trajectories of all bodies
    
    Args:
        simulation: Simulation object with history
        show: whether to display the plot
        save_path: path to save figure (optional)
    """
    fig = plt.figure(figsize=VIZ_FIGSIZE, dpi=VIZ_DPI)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each body's trajectory
    for i in range(simulation.n_bodies):
        traj = simulation.get_trajectory(i)
        color = simulation.colors[i]
        name = simulation.names[i]
        
        # Convert to AU for better scale
        traj_au = traj / AU
        
        # Plot trajectory
        ax.plot(traj_au[:, 0], traj_au[:, 1], traj_au[:, 2],
                color=color, alpha=0.6, linewidth=1, label=name)
        
        # Mark final position
        ax.scatter(traj_au[-1, 0], traj_au[-1, 1], traj_au[-1, 2],
                  color=color, s=50, marker='o')
    
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_title('N-Body Simulation Trajectories')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([
        traj_au[:, 0].max() - traj_au[:, 0].min(),
        traj_au[:, 1].max() - traj_au[:, 1].min(),
        traj_au[:, 2].max() - traj_au[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (traj_au[:, 0].max() + traj_au[:, 0].min()) * 0.5
    mid_y = (traj_au[:, 1].max() + traj_au[:, 1].min()) * 0.5
    mid_z = (traj_au[:, 2].max() + traj_au[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ_DPI)
        print(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_energy_conservation(simulation, show=True, save_path=None):
    """
    Plot energy conservation over time
    
    Shows both absolute and relative energy drift
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    times = np.array(simulation.time_history)
    energies = np.array(simulation.energy_history)
    E0 = simulation.initial_energy
    
    # Convert time to days
    times_days = times / (24 * 3600)
    
    # Absolute energy
    ax1.plot(times_days, energies, 'b-', linewidth=1)
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Total Energy (J)')
    ax1.set_title('Energy vs Time')
    ax1.grid(True, alpha=0.3)
    
    # Relative energy error
    rel_error = (energies - E0) / abs(E0) * 100
    ax2.plot(times_days, rel_error, 'r-', linewidth=1)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Relative Energy Error (%)')
    ax2.set_title('Energy Conservation Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ_DPI)
    
    if show:
        plt.show()
    
    return fig


def plot_orbit_elements(simulation, body_index, show=True):
    """
    Plot orbital elements over time (semi-major axis, eccentricity, etc.)
    
    Useful for studying precession and other orbital changes
    """
    # TODO: implement orbital element calculation
    # Need to extract a, e, i, omega, Omega from position/velocity
    pass


def create_animation(simulation, interval=50, trail_length=100):
    """
    Create animation of the simulation
    
    Args:
        simulation: Simulation object
        interval: milliseconds between frames
        trail_length: number of previous positions to show
        
    Returns:
        FuncAnimation object
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    n_frames = len(simulation.position_history)
    positions = np.array(simulation.position_history)
    
    # Convert to AU
    positions_au = positions / AU
    
    # Initialize plot elements
    scatters = []
    trails = []
    
    for i in range(simulation.n_bodies):
        # Body position
        scatter = ax.scatter([], [], [], color=simulation.colors[i], 
                            s=100, label=simulation.names[i])
        scatters.append(scatter)
        
        # Trail
        trail, = ax.plot([], [], [], color=simulation.colors[i], 
                        alpha=0.5, linewidth=1)
        trails.append(trail)
    
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.legend()
    
    # Set limits based on full trajectory
    max_range = positions_au.max() * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    def update(frame):
        for i in range(simulation.n_bodies):
            # Update body position
            pos = positions_au[frame, i]
            scatters[i]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            
            # Update trail
            start = max(0, frame - trail_length)
            trail_data = positions_au[start:frame+1, i]
            trails[i].set_data(trail_data[:, 0], trail_data[:, 1])
            trails[i].set_3d_properties(trail_data[:, 2])
        
        ax.set_title(f'Time: {simulation.time_history[frame]/(24*3600):.1f} days')
        return scatters + trails
    
    anim = FuncAnimation(fig, update, frames=n_frames, 
                        interval=interval, blit=False)
    
    return anim


def plot_precession(simulation, body_index=1, show=True):
    """
    Plot perihelion precession for an orbiting body
    
    Useful for Mercury precession analysis
    """
    traj = simulation.get_trajectory(body_index)
    
    # Calculate distance from origin (Sun)
    distances = np.linalg.norm(traj, axis=1)
    
    # Find local minima (perihelia)
    from scipy.signal import find_peaks
    minima_indices, _ = find_peaks(-distances, distance=50)
    
    if len(minima_indices) < 2:
        print("Not enough orbits to measure precession")
        return
    
    # Calculate angle of perihelion
    perihelion_angles = []
    for idx in minima_indices:
        pos = traj[idx]
        angle = np.arctan2(pos[1], pos[0])
        perihelion_angles.append(angle)
    
    perihelion_angles = np.array(perihelion_angles)
    
    # Unwrap angles
    perihelion_angles = np.unwrap(perihelion_angles)
    
    # Convert to degrees
    perihelion_angles_deg = np.degrees(perihelion_angles)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    orbit_numbers = np.arange(len(perihelion_angles))
    ax.plot(orbit_numbers, perihelion_angles_deg, 'o-', markersize=8)
    
    # Fit linear trend
    if len(orbit_numbers) > 1:
        coeffs = np.polyfit(orbit_numbers, perihelion_angles_deg, 1)
        fit_line = np.poly1d(coeffs)
        ax.plot(orbit_numbers, fit_line(orbit_numbers), 'r--', 
               label=f'Precession: {coeffs[0]:.4f}°/orbit')
    
    ax.set_xlabel('Orbit Number')
    ax.set_ylabel('Perihelion Angle (degrees)')
    ax.set_title(f'Perihelion Precession - {simulation.names[body_index]}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, perihelion_angles_deg

# Basic matplotlib visualization
# TODO: add 3D visualization with mayavi
# Visualization
# Visualization
