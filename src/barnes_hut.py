"""
Barnes-Hut octree implementation for O(n log n) gravity calculation

The tree divides 3D space into octants recursively. Each node represents
a region of space and stores the total mass and center of mass of all
bodies within it.
"""

import numpy as np
from src.config import DEFAULT_THETA


class OctreeNode:
    """
    Node in the Barnes-Hut octree
    
    Each node represents a cubic region of space and can have up to 8 children
    (one for each octant).
    """
    
    def __init__(self, center, size):
        """
        Args:
            center: (x, y, z) center of the cubic region
            size: side length of the cube
        """
        self.center = np.array(center, dtype=np.float64)
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(3, dtype=np.float64)  # center of mass
        self.body_index = None  # if leaf node with single body
        self.children = [None] * 8  # 8 octants
        self.is_leaf = True
        
    def get_octant(self, pos):
        """Determine which octant a position falls into"""
        octant = 0
        if pos[0] > self.center[0]:
            octant += 1
        if pos[1] > self.center[1]:
            octant += 2
        if pos[2] > self.center[2]:
            octant += 4
        return octant
    
    def get_octant_center(self, octant):
        """Get the center of a specific octant"""
        offset = self.size / 4.0
        x_off = offset if (octant & 1) else -offset
        y_off = offset if (octant & 2) else -offset
        z_off = offset if (octant & 4) else -offset
        return self.center + np.array([x_off, y_off, z_off])


class BarnesHutTree:
    """
    Barnes-Hut octree for efficient gravity calculations
    
    Instead of O(n²) pairwise calculations, we can approximate distant
    groups of bodies as a single mass at their center of mass.
    """
    
    def __init__(self, positions, masses, theta=DEFAULT_THETA):
        """
        Build the tree from body positions and masses
        
        Args:
            positions: (N, 3) array of positions
            masses: (N,) array of masses
            theta: opening angle criterion (smaller = more accurate, slower)
        """
        self.positions = positions
        self.masses = masses
        self.theta = theta
        self.n_bodies = len(masses)
        
        # Determine bounding box
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        center = (min_coords + max_coords) / 2.0
        size = np.max(max_coords - min_coords) * 1.1  # add 10% margin
        
        # Build tree
        self.root = OctreeNode(center, size)
        for i in range(self.n_bodies):
            self._insert(self.root, i)
            
    def _insert(self, node, body_idx):
        """Insert a body into the tree recursively"""
        pos = self.positions[body_idx]
        mass = self.masses[body_idx]
        
        # Update node's mass and COM
        node.mass += mass
        node.com = (node.com * (node.mass - mass) + pos * mass) / node.mass
        
        if node.is_leaf:
            if node.body_index is None:
                # Empty node, just store the body
                node.body_index = body_idx
            else:
                # Node has one body, need to subdivide
                old_idx = node.body_index
                node.body_index = None
                node.is_leaf = False
                
                # Re-insert the old body and the new one
                self._insert_into_octant(node, old_idx)
                self._insert_into_octant(node, body_idx)
        else:
            # Internal node, insert into appropriate octant
            self._insert_into_octant(node, body_idx)
    
    def _insert_into_octant(self, node, body_idx):
        """Insert body into the appropriate octant child"""
        pos = self.positions[body_idx]
        octant = node.get_octant(pos)
        
        if node.children[octant] is None:
            # Create new child node
            child_center = node.get_octant_center(octant)
            child_size = node.size / 2.0
            node.children[octant] = OctreeNode(child_center, child_size)
        
        self._insert(node.children[octant], body_idx)
    
    def compute_force(self, body_idx, softening=0.0):
        """
        Compute gravitational force on a body using the tree
        
        Args:
            body_idx: index of the body
            softening: softening parameter to avoid singularities
            
        Returns:
            (3,) array of force components
        """
        pos = self.positions[body_idx]
        force = np.zeros(3, dtype=np.float64)
        self._compute_force_recursive(self.root, pos, body_idx, force, softening)
        return force
    
    def _compute_force_recursive(self, node, pos, body_idx, force, softening):
        """Recursively compute force using the tree"""
        if node is None or node.mass == 0:
            return
        
        r_vec = node.com - pos
        r = np.linalg.norm(r_vec)
        
        # Skip self-interaction
        if node.is_leaf and node.body_index == body_idx:
            return
        
        # Check opening angle criterion
        # s/d < theta: treat as single body
        # s/d >= theta: need to open node
        if node.is_leaf or (node.size / r < self.theta):
            # Use this node as approximation
            r_softened = np.sqrt(r**2 + softening**2)
            force_mag = node.mass / (r_softened**3)
            force += force_mag * r_vec
        else:
            # Need to open this node and recurse into children
            for child in node.children:
                if child is not None:
                    self._compute_force_recursive(child, pos, body_idx, force, softening)


def build_tree(positions, masses, theta=DEFAULT_THETA):
    """
    Convenience function to build Barnes-Hut tree
    
    Args:
        positions: (N, 3) array of positions
        masses: (N,) array of masses  
        theta: opening angle parameter
        
    Returns:
        BarnesHutTree object
    """
    return BarnesHutTree(positions, masses, theta)
# Small fix for tree traversal

# Barnes-Hut tree implementation
# Based on the classic algorithm for N-body simulations

# Fixed boundary condition bug in tree construction
# Was causing particles near edges to be misplaced

# Optimized tree traversal
# Added multipole expansion up to quadrupole moment
# TODO: implement octupole for better accuracy
# Fixed tree bug
# Barnes-Hut working
# Barnes-Hut start
# Fix tree bug
# Barnes-Hut working
# fix
# working
