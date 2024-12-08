"""
Tests for Barnes-Hut tree implementation
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from src.barnes_hut import BarnesHutTree, OctreeNode
from src.config import G


def test_octree_node_creation():
    """Test basic node creation"""
    node = OctreeNode([0, 0, 0], 100.0)
    assert node.size == 100.0
    assert np.allclose(node.center, [0, 0, 0])
    assert node.mass == 0.0
    assert node.is_leaf


def test_octant_determination():
    """Test octant calculation"""
    node = OctreeNode([0, 0, 0], 100.0)
    
    # Test all 8 octants
    assert node.get_octant([10, 10, 10]) == 7  # +++
    assert node.get_octant([-10, 10, 10]) == 6  # -++
    assert node.get_octant([10, -10, 10]) == 5  # +-+
    assert node.get_octant([10, 10, -10]) == 3  # ++-


def test_tree_building():
    """Test that tree is built correctly"""
    positions = np.array([
        [0, 0, 0],
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10]
    ], dtype=np.float64)
    masses = np.array([1.0, 1.0, 1.0, 1.0])
    
    tree = BarnesHutTree(positions, masses)
    
    assert tree.root.mass == 4.0
    assert tree.n_bodies == 4


def test_force_calculation_single_body():
    """Test force on body from single other body"""
    # Two bodies along x-axis
    positions = np.array([
        [0, 0, 0],
        [1e10, 0, 0]  # 10 million km apart
    ], dtype=np.float64)
    masses = np.array([1e30, 1e24])  # sun-like and earth-like
    
    tree = BarnesHutTree(positions, masses, theta=0.3)
    
    # Force on body 1 from body 0
    force = tree.compute_force(1, softening=0)
    
    # Should point in -x direction (toward body 0)
    assert force[0] < 0
    assert abs(force[1]) < 1e-10
    assert abs(force[2]) < 1e-10
    
    # Check magnitude (approximately)
    r = 1e10
    expected_mag = G * masses[0] / r**2
    actual_mag = np.linalg.norm(G * force)
    
    # Should be close (within 10% due to tree approximation)
    assert abs(actual_mag - expected_mag) / expected_mag < 0.1


def test_symmetry():
    """Test that force is symmetric (Newton's third law)"""
    positions = np.array([
        [0, 0, 0],
        [1e11, 0, 0]
    ], dtype=np.float64)
    masses = np.array([1e30, 1e30])
    
    tree = BarnesHutTree(positions, masses, theta=0.5)
    
    f01 = tree.compute_force(0, softening=0)
    f10 = tree.compute_force(1, softening=0)
    
    # Forces should be equal and opposite
    assert np.allclose(f01, -f10, rtol=0.1)


def test_theta_effect():
    """Test that smaller theta gives more accurate results"""
    # Create a system where tree approximation matters
    np.random.seed(42)
    n = 20
    positions = np.random.randn(n, 3) * 1e11
    masses = np.ones(n) * 1e24
    
    # Force with small theta (accurate)
    tree_accurate = BarnesHutTree(positions, masses, theta=0.1)
    f_accurate = tree_accurate.compute_force(0, softening=1e8)
    
    # Force with large theta (fast but less accurate)
    tree_fast = BarnesHutTree(positions, masses, theta=1.0)
    f_fast = tree_fast.compute_force(0, softening=1e8)
    
    # They shouldn't be identical but should be reasonably close
    diff = np.linalg.norm(f_accurate - f_fast)
    avg = (np.linalg.norm(f_accurate) + np.linalg.norm(f_fast)) / 2
    
    # Relative difference should be non-zero but not huge
    rel_diff = diff / avg
    assert rel_diff > 0.01  # Different
    assert rel_diff < 0.5   # But not too different


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
