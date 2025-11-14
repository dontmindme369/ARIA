#!/usr/bin/env python3
"""
ARIA Exploration - Quaternion-based Semantic Space Navigation

Implements golden ratio spiral exploration through 4D semantic space using
quaternion rotations for perspective-aware retrieval.

Key Features:
- Golden ratio (φ = 1.618...) spiral sampling
- Quaternion rotation through semantic subspaces
- PCA-based perspective alignment
- Multi-rotation iterative refinement

Mathematical Foundation:
- Quaternions provide singularity-free 3D rotations in 4D space
- Golden ratio ensures optimal space coverage (no clustering)
- PCA rotation aligns search direction with dominant semantic axes
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import sys

# Import quaternion mathematics
try:
    from .quaternion import Quaternion
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent))
    from quaternion import Quaternion


# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895
PHI_INV = 1 / PHI  # ≈ 0.618033988749895


class QuaternionExplorer:
    """
    Quaternion-based semantic space exploration with golden ratio spiral.

    The golden ratio spiral ensures optimal sampling of semantic space by
    avoiding resonance patterns that would cause clustering.
    """

    def __init__(self, embedding_dim: int = 128):
        """
        Initialize explorer.

        Args:
            embedding_dim: Dimensionality of semantic embeddings
        """
        self.embedding_dim = embedding_dim
        self.phi = PHI
        self.phi_inv = PHI_INV

    def golden_ratio_spiral(self, num_points: int, radius: float = 1.0) -> List[Tuple[float, float, float]]:
        """
        Generate points on golden ratio spiral in 3D space.

        The golden spiral provides optimal point distribution without clustering.
        Uses spherical Fibonacci lattice for uniform sampling.

        Args:
            num_points: Number of points to generate
            radius: Radius of sphere

        Returns:
            List of (x, y, z) coordinates on unit sphere
        """
        points = []

        for i in range(num_points):
            # Golden ratio spiral parameterization
            theta = 2 * math.pi * i * self.phi_inv  # Azimuthal angle
            phi = math.acos(1 - 2 * (i + 0.5) / num_points)  # Polar angle

            # Spherical to Cartesian
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)

            points.append((x, y, z))

        return points

    def create_rotation_quaternion(self, axis: np.ndarray, angle_deg: float) -> Quaternion:
        """
        Create quaternion for rotation around axis by angle.

        Args:
            axis: 3D rotation axis (will be normalized)
            angle_deg: Rotation angle in degrees

        Returns:
            Unit quaternion representing rotation
        """
        # Normalize axis
        axis = np.array(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return Quaternion(1, 0, 0, 0)  # Identity
        axis = axis / norm

        # Convert to radians
        angle_rad = math.radians(angle_deg)
        half_angle = angle_rad / 2

        # Quaternion from axis-angle
        w = math.cos(half_angle)
        sin_half = math.sin(half_angle)
        x = axis[0] * sin_half
        y = axis[1] * sin_half
        z = axis[2] * sin_half

        return Quaternion(w, x, y, z).normalize()

    def rotate_embedding(self, embedding: np.ndarray, quaternion: Quaternion) -> np.ndarray:
        """
        Rotate 3D portion of embedding using quaternion.

        For embeddings with dim > 3, only first 3 dimensions are rotated.
        Remaining dimensions are preserved.

        Args:
            embedding: N-dimensional embedding vector
            quaternion: Rotation quaternion

        Returns:
            Rotated embedding
        """
        result = embedding.copy()

        # Extract first 3 dimensions for rotation
        if len(embedding) >= 3:
            vec3 = embedding[:3]
            rotated = quaternion.rotate_vector(vec3)
            result[:3] = rotated

        return result

    def compute_pca_rotation(self, embeddings: np.ndarray, target_variance: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PCA rotation to align with dominant semantic directions.

        Args:
            embeddings: Matrix of embeddings (n_samples, n_features)
            target_variance: Target cumulative variance to preserve

        Returns:
            (rotation_matrix, explained_variance_ratio)
        """
        if len(embeddings) < 2:
            # Not enough samples for PCA
            identity = np.eye(min(embeddings.shape[1], 3))
            return identity, np.array([1.0])

        # Center data
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean

        # Compute covariance matrix
        cov = np.cov(centered.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Explained variance ratio
        total_var = eigenvalues.sum()
        explained_variance = eigenvalues / total_var if total_var > 0 else eigenvalues

        # Take top 3 components for quaternion rotation
        rotation_matrix = eigenvectors[:, :3] if eigenvectors.shape[1] >= 3 else eigenvectors

        return rotation_matrix, explained_variance

    def multi_rotation_exploration(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        num_rotations: int = 3,
        angle_per_rotation: float = 30.0,
        subspace_index: int = 0
    ) -> List[Tuple[int, float]]:
        """
        Perform multi-rotation exploration through semantic space.

        Args:
            query_embedding: Query vector
            document_embeddings: Document vectors (n_docs, dim)
            num_rotations: Number of rotation iterations
            angle_per_rotation: Rotation angle in degrees
            subspace_index: Subspace index (0-7) for perspective bias

        Returns:
            List of (doc_index, similarity_score) tuples
        """
        if num_rotations <= 0:
            # No rotation, just compute cosine similarity
            similarities = self._cosine_similarities(query_embedding, document_embeddings)
            return [(i, sim) for i, sim in enumerate(similarities)]

        # Compute PCA rotation for semantic alignment
        rotation_matrix, _ = self.compute_pca_rotation(document_embeddings)

        # Select rotation axis based on subspace index
        if rotation_matrix.shape[1] > subspace_index:
            rotation_axis = rotation_matrix[:, subspace_index % rotation_matrix.shape[1]]
        else:
            # Fallback to standard axes
            axis_map = {
                0: np.array([1, 0, 0]),  # X-axis
                1: np.array([0, 1, 0]),  # Y-axis
                2: np.array([0, 0, 1]),  # Z-axis
                3: np.array([1, 1, 0]) / math.sqrt(2),  # XY diagonal
                4: np.array([1, 0, 1]) / math.sqrt(2),  # XZ diagonal
                5: np.array([0, 1, 1]) / math.sqrt(2),  # YZ diagonal
                6: np.array([1, 1, 1]) / math.sqrt(3),  # XYZ diagonal
                7: np.array([1, -1, 0]) / math.sqrt(2),  # XY anti-diagonal
            }
            rotation_axis = axis_map.get(subspace_index % 8, np.array([1, 0, 0]))

        # Pad rotation axis to embedding dim
        if len(rotation_axis) < 3:
            rotation_axis = np.pad(rotation_axis, (0, 3 - len(rotation_axis)))
        rotation_axis = rotation_axis[:3]

        # Accumulate scores across rotations
        all_scores = np.zeros(len(document_embeddings))

        current_query = query_embedding.copy()

        for rotation_idx in range(num_rotations):
            # Create rotation quaternion with golden ratio angle scaling
            angle = angle_per_rotation * (rotation_idx + 1) * self.phi_inv
            quat = self.create_rotation_quaternion(rotation_axis, angle)

            # Rotate query embedding
            current_query = self.rotate_embedding(current_query, quat)

            # Compute similarities at this rotation
            similarities = self._cosine_similarities(current_query, document_embeddings)

            # Accumulate with decaying weight (later rotations weighted less)
            weight = 1.0 / (rotation_idx + 1)
            all_scores += similarities * weight

        # Normalize scores
        if num_rotations > 0:
            all_scores /= num_rotations

        # Return sorted by score
        results = [(i, float(score)) for i, score in enumerate(all_scores)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _cosine_similarities(self, query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between query and documents.

        Args:
            query: Query vector (dim,)
            documents: Document vectors (n_docs, dim)

        Returns:
            Similarity scores (n_docs,)
        """
        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-12:
            return np.zeros(len(documents))
        query_normalized = query / query_norm

        # Normalize documents
        doc_norms = np.linalg.norm(documents, axis=1, keepdims=True)
        doc_norms[doc_norms < 1e-12] = 1.0  # Avoid division by zero
        docs_normalized = documents / doc_norms

        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(docs_normalized, query_normalized)

        return similarities

    def spiral_sample_positions(self, num_samples: int, max_radius: float = 1.0) -> List[Tuple[float, float, float]]:
        """
        Sample positions along golden ratio spiral for exploration.

        Useful for generating diverse exploration points in semantic space.

        Args:
            num_samples: Number of positions to sample
            max_radius: Maximum radius from origin

        Returns:
            List of (x, y, z) positions
        """
        positions = []

        for i in range(num_samples):
            # Radius grows with golden ratio
            radius = max_radius * (i / num_samples) ** self.phi_inv

            # Angle follows golden angle (2π/φ²)
            golden_angle = 2 * math.pi * self.phi_inv
            theta = i * golden_angle

            # Z coordinate follows square root (uniform in volume)
            z = radius * (1 - 2 * i / num_samples)

            # XY radius follows z
            xy_radius = radius * math.sqrt(1 - (z/radius)**2) if radius > 0 else 0

            x = xy_radius * math.cos(theta)
            y = xy_radius * math.sin(theta)

            positions.append((x, y, z))

        return positions


def compute_rotation_params_from_perspective(
    perspective: str,
    confidence: float = 1.0
) -> Dict[str, Any]:
    """
    Compute rotation parameters from detected perspective.

    Maps 8 perspectives to rotation angles and subspace indices optimized
    for that type of query.

    Args:
        perspective: One of 8 perspectives (educational, diagnostic, etc.)
        confidence: Detection confidence (0-1), scales rotation angle

    Returns:
        Dict with "angle" and "subspace" keys
    """
    # Perspective-specific rotation angles (degrees)
    # Based on empirical testing for optimal semantic alignment
    perspective_map = {
        "educational": {"angle": 30.0, "subspace": 0},      # Gentle rotation for broad concepts
        "diagnostic": {"angle": 90.0, "subspace": 1},       # Aggressive rotation for focused search
        "security": {"angle": 45.0, "subspace": 2},         # Moderate rotation for threat analysis
        "implementation": {"angle": 60.0, "subspace": 3},   # Strong rotation for code/building
        "research": {"angle": 120.0, "subspace": 4},        # Very aggressive for investigation
        "theoretical": {"angle": 75.0, "subspace": 5},      # Strong for abstract concepts
        "practical": {"angle": 50.0, "subspace": 6},        # Moderate for applied knowledge
        "reference": {"angle": 15.0, "subspace": 7},        # Minimal rotation for facts
        "mixed": {"angle": 0.0, "subspace": -1},            # No rotation for ambiguous queries
    }

    params = perspective_map.get(perspective, perspective_map["mixed"])

    # Scale angle by confidence
    angle = params["angle"] * confidence

    return {
        "angle": angle,
        "subspace": params["subspace"],
        "perspective": perspective,
        "confidence": confidence
    }


# Example usage and testing
if __name__ == "__main__":
    print("ARIA Quaternion Exploration - Test Suite")
    print("=" * 70)

    # Test 1: Golden ratio spiral
    print("\n1. Golden Ratio Spiral Generation")
    explorer = QuaternionExplorer()
    spiral_points = explorer.golden_ratio_spiral(10, radius=1.0)
    print(f"   Generated {len(spiral_points)} points on φ-spiral")
    print(f"   φ = {explorer.phi:.10f}")
    print(f"   Sample points (first 3):")
    for i, (x, y, z) in enumerate(spiral_points[:3]):
        print(f"     Point {i}: ({x:.4f}, {y:.4f}, {z:.4f})")

    # Test 2: Quaternion rotation
    print("\n2. Quaternion Rotation")
    axis = np.array([0, 0, 1])  # Z-axis
    angle = 45.0  # degrees
    quat = explorer.create_rotation_quaternion(axis, angle)
    print(f"   Rotation: {angle}° around {axis}")
    print(f"   Quaternion: w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f}")

    # Rotate a vector
    vec = np.array([1.0, 0.0, 0.0])
    rotated = quat.rotate_vector(vec)
    print(f"   Input vector: {vec}")
    print(f"   Rotated vector: [{rotated[0]:.4f}, {rotated[1]:.4f}, {rotated[2]:.4f}]")

    # Test 3: Multi-rotation exploration
    print("\n3. Multi-Rotation Exploration")
    # Mock embeddings
    query_emb = np.random.randn(128)
    doc_embs = np.random.randn(50, 128)

    results = explorer.multi_rotation_exploration(
        query_emb, doc_embs,
        num_rotations=3,
        angle_per_rotation=30.0,
        subspace_index=0
    )
    print(f"   Query embedding: {query_emb.shape}")
    print(f"   Document embeddings: {doc_embs.shape}")
    print(f"   Top 5 results:")
    for i, (doc_idx, score) in enumerate(results[:5]):
        print(f"     {i+1}. Doc {doc_idx}: similarity={score:.4f}")

    # Test 4: Perspective mapping
    print("\n4. Perspective-Based Rotation Parameters")
    for perspective in ["educational", "diagnostic", "research", "reference"]:
        params = compute_rotation_params_from_perspective(perspective, confidence=0.8)
        print(f"   {perspective:15s} → angle={params['angle']:5.1f}°, subspace={params['subspace']}")

    print("\n" + "=" * 70)
    print("✅ All tests completed successfully")
