#!/usr/bin/env python3
"""
PCA Exploration Module for ARIA
================================

Implements PCA subspace rotation exploration for query embeddings:
- Projects query into top-k PCA components
- Rotates in 3D subspace using random rotation matrices
- Projects back to full embedding space
- Finds documents from multiple semantic perspectives

Based on local_rag_context_v6_9k_adaptive_learn.py exploration strategy.
"""

import math
import random
from typing import List, Tuple, Optional
import numpy as np


# ============================================================================
# ROTATION MATRIX GENERATION
# ============================================================================

def create_rotation_matrix_3d(ax: float, ay: float, az: float) -> np.ndarray:
    """
    Create 3D rotation matrix from Euler angles
    
    Args:
        ax: Rotation around X axis (radians)
        ay: Rotation around Y axis (radians)
        az: Rotation around Z axis (radians)
    
    Returns:
        3x3 rotation matrix
    """
    # Rotation around X
    cx, sx = math.cos(ax), math.sin(ax)
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])
    
    # Rotation around Y
    cy, sy = math.cos(ay), math.sin(ay)
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])
    
    # Rotation around Z
    cz, sz = math.cos(az), math.sin(az)
    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def random_rotation_matrix_3d(max_degrees: float = 12.0, rng: Optional[random.Random] = None) -> np.ndarray:
    """
    Generate random 3D rotation matrix
    
    Args:
        max_degrees: Maximum rotation angle in any axis (degrees)
        rng: Optional random number generator
    
    Returns:
        3x3 rotation matrix
    """
    if rng is None:
        rng = random.Random()
    
    # Random angles in each axis
    ax = math.radians(rng.uniform(-max_degrees, max_degrees))
    ay = math.radians(rng.uniform(-max_degrees, max_degrees))
    az = math.radians(rng.uniform(-max_degrees, max_degrees))
    
    return create_rotation_matrix_3d(ax, ay, az)


def create_givens_rotation(dim: int, angle: float, plane_i: int = 0, plane_j: int = 1) -> np.ndarray:
    """
    Create Givens rotation matrix in specified 2D plane of n-dimensional space
    
    Args:
        dim: Dimensionality of space
        angle: Rotation angle (radians)
        plane_i: First dimension of rotation plane
        plane_j: Second dimension of rotation plane
    
    Returns:
        dim x dim rotation matrix
    """
    R = np.eye(dim)
    c, s = math.cos(angle), math.sin(angle)
    
    R[plane_i, plane_i] = c
    R[plane_i, plane_j] = -s
    R[plane_j, plane_i] = s
    R[plane_j, plane_j] = c
    
    return R


# ============================================================================
# PCA PROJECTION
# ============================================================================

def pca_project(X: np.ndarray, k: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project data to top-k PCA components
    
    Args:
        X: Data matrix (N x D)
        k: Number of components to keep
    
    Returns:
        (mu, V) where:
            mu: Mean vector (1 x D)
            V: Top k principal components (k x D)
    """
    # Center data
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    
    # SVD
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    
    # Top k components
    V = VT[:k, :]
    
    return mu, V


# ============================================================================
# PCA ROTATION EXPLORATION
# ============================================================================

class PCAExplorer:
    """
    Explores embedding space using PCA subspace rotations
    
    Strategy:
    1. Project query + documents to PCA subspace (e.g., 16D → 8D)
    2. Rotate query in 3D within subspace
    3. Project back to full space
    4. Compute similarity from multiple rotated perspectives
    5. Combine results (max or mean)
    """
    
    def __init__(
        self,
        n_rotations: int = 8,
        subspace_dim: int = 16,
        max_angle_deg: float = 12.0,
        combine_mode: str = "max",
        seed: Optional[int] = None
    ):
        """
        Initialize PCA explorer
        
        Args:
            n_rotations: Number of random rotations to try
            subspace_dim: Dimensionality of PCA subspace
            max_angle_deg: Maximum rotation angle (degrees)
            combine_mode: How to combine rotations ("max" or "mean")
            seed: Random seed for reproducibility
        """
        self.n_rotations = n_rotations
        self.subspace_dim = subspace_dim
        self.max_angle_deg = max_angle_deg
        self.combine_mode = combine_mode
        self.rng = random.Random(seed) if seed is not None else random.Random()
    
    def explore(
        self,
        query_vec: np.ndarray,
        doc_vecs: np.ndarray,
        top_n: int = 256
    ) -> Tuple[np.ndarray, str]:
        """
        Explore embedding space using PCA rotations
        
        Args:
            query_vec: Query embedding (D,)
            doc_vecs: Document embeddings (N x D)
            top_n: Number of documents to consider for PCA
        
        Returns:
            (scores, message) where:
                scores: Similarity scores (N,)
                message: Debug message
        """
        N = len(doc_vecs)
        
        # Use subset for PCA if corpus is large
        M = min(N, max(top_n, 1))
        
        # Compute base similarity
        base_scores = np.dot(doc_vecs[:M], query_vec)
        
        # PCA projection
        mu, V = pca_project(doc_vecs[:M], k=min(self.subspace_dim, doc_vecs.shape[1]))
        k_eff = V.shape[0]
        
        # Need at least 3D for rotation
        if k_eff < 3:
            # Fall back to base scores
            scores = np.zeros(N)
            scores[:M] = base_scores
            return self._normalize_scores(scores), f"[PCA] k={k_eff}<3, using base scores only"
        
        # Project query to PCA subspace
        query_low = (query_vec - mu.squeeze()) @ V.T
        
        # Perform rotations
        rotation_scores = []
        
        for i in range(max(1, self.n_rotations)):
            # Generate random 3D rotation
            R3 = random_rotation_matrix_3d(self.max_angle_deg, self.rng)
            
            # Rotate first 3 dimensions of query
            query_rotated = query_low.copy()
            query_rotated[:3] = R3 @ query_rotated[:3]
            
            # Project back to full space
            query_full = mu.squeeze() + query_rotated @ V
            
            # Normalize
            query_full = query_full / (np.linalg.norm(query_full) + 1e-9)
            
            # Compute similarities
            sims = np.dot(doc_vecs[:M], query_full)
            rotation_scores.append(sims)
        
        # Stack all rotation results
        rotation_scores = np.stack(rotation_scores, axis=0)  # (R x M)
        
        # Combine rotations
        if self.combine_mode == "max":
            combined = np.max(rotation_scores, axis=0)
        else:  # mean
            combined = np.mean(rotation_scores, axis=0)
        
        # Take maximum of base and rotated scores
        final = np.maximum(base_scores, combined)
        
        # Pad with zeros for documents not in PCA
        scores = np.zeros(N)
        scores[:M] = final
        
        msg = f"[PCA] M={M}, k={k_eff}, rotations={self.n_rotations}, angle=±{self.max_angle_deg}°, combine={self.combine_mode}"
        
        return self._normalize_scores(scores), msg
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]"""
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score < 1e-9:
            return scores
        
        return (scores - min_score) / (max_score - min_score)


# ============================================================================
# GOLDEN RATIO SPIRAL ANGLES
# ============================================================================

def golden_ratio_angles(n_angles: int) -> List[float]:
    """
    Generate rotation angles using golden ratio spiral
    
    The golden ratio (φ) creates optimal angular spacing for exploration
    without clustering or gaps.
    
    Args:
        n_angles: Number of angles to generate
    
    Returns:
        List of angles in radians
    """
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    golden_angle = 2 * math.pi / phi  # ~137.5 degrees
    
    return [i * golden_angle for i in range(n_angles)]


def create_golden_rotation_sequence(
    dim: int,
    n_rotations: int,
    max_angle: float = 0.2
) -> List[np.ndarray]:
    """
    Create sequence of rotation matrices using golden ratio spacing
    
    Args:
        dim: Dimensionality of space
        n_rotations: Number of rotations
        max_angle: Maximum rotation angle (radians)
    
    Returns:
        List of rotation matrices
    """
    angles = golden_ratio_angles(n_rotations)
    rotations = []
    
    for angle in angles:
        # Scale angle to max
        scaled_angle = (angle % (2 * math.pi)) / (2 * math.pi) * max_angle
        
        # Create rotation in first 2 dimensions
        R = create_givens_rotation(dim, scaled_angle, 0, 1)
        rotations.append(R)
    
    return rotations


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'PCAExplorer',
    'create_rotation_matrix_3d',
    'random_rotation_matrix_3d',
    'create_givens_rotation',
    'pca_project',
    'golden_ratio_angles',
    'create_golden_rotation_sequence'
]
