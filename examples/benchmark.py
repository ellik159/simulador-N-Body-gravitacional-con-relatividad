"""
Performance benchmark comparing different methods

Compares:
- Direct summation (CPU)
- Barnes-Hut (CPU)
- Direct summation (GPU)
- Different theta values for Barnes-Hut
"""

import numpy as np
import sys
import time
sys.path.append('..')

from src.simulation import Simulation
from src.config import SOLAR_MASS, AU


def generate_random_system(n_bodies):
    """Generate random system of bodies"""
    bodies = []
    
    # Central massive body
    bodies.append({
        'mass': 10 * SOLAR_MASS,
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0]),
        'color': 'yellow'
    })
    
    # Random bodies in orbit
    for i in range(n_bodies - 1):
        # Random distance and angle
        r = np.random.uniform(0.5, 10.0) * AU
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # Circular orbit velocity
        v_circ = np.sqrt(6.67e-11 * 10 * SOLAR_MASS / r)
        
        # Perpendicular velocity
        v_dir = np.array([-y, x, 0])
        v_dir = v_dir / np.linalg.norm(v_dir)
        
        bodies.append({
            'mass': np.random.uniform(1e20, 1e24),
            'pos': np.array([x, y, z]),
            'vel': v_dir * v_circ * np.random.uniform(0.8, 1.2),
            'color': 'blue'
        })
    
    return bodies


def benchmark(n_bodies, n_steps, method_name, **kwargs):
    """Run benchmark for a specific configuration"""
    print(f"\n{method_name}:")
    print(f"  Bodies: {n_bodies}, Steps: {n_steps}")
    
    bodies = generate_random_system(n_bodies)
    
    sim = Simulation(bodies, dt=3600*24, **kwargs)
    
    start = time.time()
    sim.run(n_steps=n_steps, save_every=n_steps, verbose=False)
    elapsed = time.time() - start
    
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Steps/sec: {n_steps/elapsed:.1f}")
    
    return elapsed


def main():
    print("="*60)
    print("Performance Benchmark")
    print("="*60)
    
    # Small system comparison
    print("\n" + "="*60)
    print("Small System (100 bodies, 100 steps)")
    print("="*60)
    
    times = {}
    times['direct_cpu'] = benchmark(
        100, 100, "Direct (CPU)",
        use_barnes_hut=False, use_gpu=False, use_pn=False
    )
    
    times['bh_cpu'] = benchmark(
        100, 100, "Barnes-Hut (CPU, theta=0.5)",
        use_barnes_hut=True, use_gpu=False, use_pn=False, theta=0.5
    )
    
    try:
        times['direct_gpu'] = benchmark(
            100, 100, "Direct (GPU)",
            use_barnes_hut=False, use_gpu=True, use_pn=False
        )
    except Exception as e:
        print(f"  GPU failed: {e}")
        times['direct_gpu'] = None
    
    # Medium system
    print("\n" + "="*60)
    print("Medium System (500 bodies, 50 steps)")
    print("="*60)
    
    times['medium_bh'] = benchmark(
        500, 50, "Barnes-Hut (CPU, theta=0.5)",
        use_barnes_hut=True, use_gpu=False, use_pn=False, theta=0.5
    )
    
    # Test different theta values
    print("\n" + "="*60)
    print("Barnes-Hut Theta Comparison (500 bodies, 50 steps)")
    print("="*60)
    
    for theta in [0.3, 0.5, 0.7, 1.0]:
        benchmark(
            500, 50, f"Barnes-Hut (theta={theta})",
            use_barnes_hut=True, use_gpu=False, use_pn=False, theta=theta
        )
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("Smaller theta = more accurate but slower")
    print("Larger theta = faster but less accurate")
    print("Barnes-Hut wins for large N (typically N > 200)")
    print("GPU acceleration helps significantly for direct summation")


if __name__ == '__main__':
    main()

# Performance benchmark
# Compares Barnes-Hut vs direct summation
# Benchmark
# Benchmark
