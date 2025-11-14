#!/usr/bin/env python3
"""
Quaternion Perspective Rotator
Uses quaternion rotations to explore perspective orientation space.

This is the core of ARIA's perspective rotation mechanism.
Instead of rotating semantic embeddings, we rotate perspective frames.
"""

import numpy as np
from typing import List, Tuple, Dict
import json
from pathlib import Path


class QuaternionPerspectiveRotator:
    """
    Quaternion-based rotation through perspective orientation space.

    Key insight: Perspective has geometric structure beyond semantic clustering.
    Documents occupy angular positions. Quaternion rotations explore these
    orientations to find alignment between query perspective and document perspective.
    """

    def __init__(self, num_perspectives: int = 8):
        """
        Initialize quaternion rotator.

        Args:
            num_perspectives: Dimensionality of perspective space (8 perspectives)
        """
        self.num_perspectives = num_perspectives
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.fibonacci_angles = self._generate_fibonacci_spiral()

    def _generate_fibonacci_spiral(self, num_points: int = 50) -> np.ndarray:
        """
        Generate golden ratio spiral angles for perspective exploration.

        The Fibonacci spiral converges toward precision - it explores
        perspective space efficiently without redundant rotations.

        Returns:
            Array of rotation angles following golden ratio spacing
        """
        angles = []
        for i in range(num_points):
            # Golden angle: 2π / φ²
            theta = i * 2 * np.pi / (self.phi ** 2)
            angles.append(theta)

        return np.array(angles)

    def create_quaternion(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Create quaternion from axis and angle.

        Quaternion: q = [cos(θ/2), sin(θ/2) * axis]

        Args:
            axis: 3D rotation axis (unit vector)
            angle: Rotation angle in radians

        Returns:
            4D quaternion [w, x, y, z]
        """
        axis = axis / np.linalg.norm(axis)  # Normalize

        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = np.sin(half_angle) * axis

        return np.array([w, xyz[0], xyz[1], xyz[2]])

    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions.

        Used for composing rotations in perspective space.
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def rotate_vector_3d(self, vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """
        Rotate 3D vector using quaternion.

        This is the Rodriguez formula for quaternion rotation.

        Args:
            vector: 3D vector to rotate
            quaternion: Rotation quaternion [w, x, y, z]

        Returns:
            Rotated 3D vector
        """
        # Convert to pure quaternion
        v_quat = np.array([0, vector[0], vector[1], vector[2]])

        # Conjugate quaternion
        q_conj = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])

        # Rotation: q * v * q_conj
        temp = self.quaternion_multiply(quaternion, v_quat)
        result = self.quaternion_multiply(temp, q_conj)

        return result[1:]  # Return xyz components

    def project_to_3d(self, orientation_vector: np.ndarray) -> np.ndarray:
        """
        Project 8D perspective orientation to 3D rotation space.

        Uses PCA or predefined mapping to reduce dimensionality.
        The 3D space represents the primary axes of perspective variation.

        Args:
            orientation_vector: 8D perspective scores

        Returns:
            3D vector suitable for quaternion rotation
        """
        # Simple projection: weight dominant perspectives
        # Axis 1: Theoretical <-> Practical
        axis1 = orientation_vector[5] - orientation_vector[6]  # theoretical - practical

        # Axis 2: Educational <-> Security
        axis2 = orientation_vector[0] - orientation_vector[2]  # educational - security

        # Axis 3: Research <-> Reference
        axis3 = orientation_vector[4] - orientation_vector[7]  # research - reference

        return np.array([axis1, axis2, axis3])

    def spiral_perspective_exploration(
        self,
        query_orientation: np.ndarray,
        document_orientations: Dict[str, np.ndarray]
    ) -> List[Tuple[str, float, int]]:
        """
        Explore perspective space using golden ratio spiral.

        This is the core ARIA mechanism: spiral inward through rotation
        planes to find documents with aligned perspective orientation.

        Args:
            query_orientation: 8D query perspective vector
            document_orientations: {doc_id: 8D orientation vector}

        Returns:
            List of (doc_id, alignment_score, rotation_step) sorted by alignment
        """
        # Project query to 3D
        query_3d = self.project_to_3d(query_orientation)

        results = []

        # Spiral through rotation space
        for step, angle in enumerate(self.fibonacci_angles):
            # Create rotation axis from query direction
            if np.linalg.norm(query_3d) > 0:
                axis = query_3d / np.linalg.norm(query_3d)
            else:
                axis = np.array([1, 0, 0])  # Default axis

            # Create quaternion for this rotation
            quat = self.create_quaternion(axis, angle)

            # Rotate query perspective
            rotated_query_3d = self.rotate_vector_3d(query_3d, quat)

            # Compare against all documents
            for doc_id, doc_orientation in document_orientations.items():
                doc_3d = self.project_to_3d(doc_orientation)

                # Measure angular alignment
                alignment = self._angular_alignment(rotated_query_3d, doc_3d)

                results.append((doc_id, alignment, step))

        # Sort by alignment score
        results.sort(key=lambda x: x[1], reverse=True)

        # Remove duplicates (keep best rotation for each doc)
        seen_docs = set()
        unique_results = []
        for doc_id, score, step in results:
            if doc_id not in seen_docs:
                unique_results.append((doc_id, score, step))
                seen_docs.add(doc_id)

        return unique_results

    def _angular_alignment(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute angular alignment between two perspective vectors.

        This is NOT semantic similarity - it's perspective orientation alignment.

        Returns:
            Alignment score (0.0 to 1.0)
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine of angle between vectors
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)

        # Convert to 0-1 range
        return (cos_angle + 1) / 2

    def find_perspective_anchors(
        self,
        query_orientation: np.ndarray,
        num_anchors: int = 16
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Find optimal perspective anchor points in rotation space.

        These are the "quaternion contextual rotation matrices" -
        predefined perspective reference frames that documents get
        measured against.

        Args:
            query_orientation: Query's perspective vector
            num_anchors: Number of anchor frames to generate

        Returns:
            List of (anchor_orientation_3d, relevance_score)
        """
        query_3d = self.project_to_3d(query_orientation)
        anchors = []

        # Generate anchor points using icosahedral distribution
        # (optimal spacing on sphere)
        for i in range(num_anchors):
            theta = i * 2 * np.pi / self.phi  # Golden angle
            phi = np.arccos(1 - 2 * (i + 0.5) / num_anchors)  # Fibonacci sphere

            # Spherical to Cartesian
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            anchor_point = np.array([x, y, z])

            # Measure relevance to query
            relevance = self._angular_alignment(query_3d, anchor_point)

            anchors.append((anchor_point, relevance))

        # Sort by relevance
        anchors.sort(key=lambda x: x[1], reverse=True)

        return anchors

    def visualize_rotation_path(
        self,
        query_orientation: np.ndarray,
        num_steps: int = 20
    ) -> Dict[str, List]:
        """
        Visualize the spiral rotation path through perspective space.

        Returns data for plotting rotation trajectory.
        """
        query_3d = self.project_to_3d(query_orientation)
        axis = query_3d / np.linalg.norm(query_3d) if np.linalg.norm(query_3d) > 0 else np.array([1, 0, 0])

        path = []

        for i in range(num_steps):
            angle = self.fibonacci_angles[i]
            quat = self.create_quaternion(axis, angle)
            rotated = self.rotate_vector_3d(query_3d, quat)
            path.append(rotated.tolist())

        return {
            'original': query_3d.tolist(),
            'rotation_path': path,
            'angles': self.fibonacci_angles[:num_steps].tolist()
        }


def demo():
    """Demonstrate quaternion perspective rotation"""

    print("="*80)
    print("QUATERNION PERSPECTIVE ROTATOR - DEMO")
    print("="*80)

    rotator = QuaternionPerspectiveRotator()

    # Simulate perspective orientations
    query_orientation = np.array([0.8, 0.1, 0.05, 0.1, 0.1, 0.2, 0.1, 0.05])  # Educational-heavy
    print(f"\nQuery orientation (8D): {query_orientation}")
    print("  Dominant: Educational (0.8)")

    # Simulate document orientations
    documents = {
        'doc_educational_1': np.array([0.75, 0.05, 0.05, 0.05, 0.1, 0.15, 0.1, 0.05]),
        'doc_educational_2': np.array([0.70, 0.10, 0.05, 0.08, 0.12, 0.18, 0.09, 0.06]),
        'doc_security': np.array([0.05, 0.05, 0.85, 0.10, 0.05, 0.05, 0.05, 0.05]),
        'doc_diagnostic': np.array([0.05, 0.80, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05]),
        'doc_implementation': np.array([0.10, 0.05, 0.05, 0.70, 0.10, 0.10, 0.15, 0.05])
    }

    print(f"\nDocument orientations: {len(documents)} documents")
    for doc_id, orientation in documents.items():
        dominant_idx = np.argmax(orientation)
        perspectives = ['educational', 'diagnostic', 'security', 'implementation',
                       'research', 'theoretical', 'practical', 'reference']
        print(f"  {doc_id:20s}: {perspectives[dominant_idx]} ({orientation[dominant_idx]:.2f})")

    print("\n[Phase 1] Projecting to 3D rotation space...")
    query_3d = rotator.project_to_3d(query_orientation)
    print(f"Query 3D: {query_3d}")

    print("\n[Phase 2] Generating perspective anchor frames...")
    anchors = rotator.find_perspective_anchors(query_orientation, num_anchors=8)
    print(f"Top 3 anchor frames:")
    for i, (anchor, relevance) in enumerate(anchors[:3]):
        print(f"  Anchor {i+1}: relevance={relevance:.3f}")

    print("\n[Phase 3] Spiral exploration with golden ratio...")
    results = rotator.spiral_perspective_exploration(query_orientation, documents)

    print(f"\nTop 5 aligned documents:")
    for i, (doc_id, alignment, rotation_step) in enumerate(results[:5]):
        print(f"  {i+1}. {doc_id:20s}: alignment={alignment:.3f} (found at rotation step {rotation_step})")

    print("\n[Phase 4] Rotation path visualization data...")
    viz_data = rotator.visualize_rotation_path(query_orientation, num_steps=10)
    print(f"Generated {len(viz_data['rotation_path'])} rotation steps")
    print(f"First 3 rotated positions:")
    for i, pos in enumerate(viz_data['rotation_path'][:3]):
        print(f"  Step {i+1}: {np.array(pos)}")

    print("\n" + "="*80)
    print("This is how ARIA rotates through perspective space!")
    print("="*80)


if __name__ == '__main__':
    demo()
