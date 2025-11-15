#!/usr/bin/env python3
"""
Quaternion Rotation Matrices - 4D Rotations in 3D Space
========================================================

This module provides a complete implementation of quaternion mathematics for
representing and manipulating 3D rotations using 4D quaternions.

Quaternions (w, x, y, z) provide a compact, singularity-free representation of
3D rotations. They avoid gimbal lock and enable smooth interpolation.

Key Features:
- Full quaternion algebra (multiplication, conjugate, inverse, etc.)
- Conversion to/from 3x3 rotation matrices
- Conversion to/from axis-angle representation
- Conversion to/from Euler angles (XYZ, ZYX conventions)
- Vector rotation in 3D space
- SLERP (Spherical Linear Interpolation)
- Quaternion logarithm and exponential maps

Mathematical Background:
-----------------------
A unit quaternion q = w + xi + yj + zk represents a rotation in 3D space.
- w is the scalar (real) part
- (x, y, z) is the vector (imaginary) part
- For rotations, ||q|| = 1 (unit quaternion)

Rotation by angle θ around unit axis v = (vx, vy, vz):
    q = [cos(θ/2), vx*sin(θ/2), vy*sin(θ/2), vz*sin(θ/2)]

Author: dontmindme369
License: CC BY-NC 4.0
"""

import numpy as np
from typing import Union, Tuple, List, Optional
import math


class Quaternion:
    """
    Quaternion class for 3D rotations using 4D representation.

    A quaternion is represented as q = w + xi + yj + zk, where:
    - w is the scalar (real) component
    - (x, y, z) form the vector (imaginary) component

    For rotation quaternions, we enforce ||q|| = 1 (unit quaternion).

    Attributes:
        w (float): Scalar component
        x (float): i component
        y (float): j component
        z (float): k component

    Examples:
        >>> # Identity rotation
        >>> q = Quaternion(1, 0, 0, 0)

        >>> # 90° rotation around Z-axis
        >>> q = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)

        >>> # Rotate a vector
        >>> v = np.array([1, 0, 0])
        >>> v_rotated = q.rotate_vector(v)
    """

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """
        Initialize quaternion with components.

        Args:
            w: Scalar (real) component
            x: First imaginary component (i)
            y: Second imaginary component (j)
            z: Third imaginary component (k)
        """
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ========================================================================
    # ARRAY CONVERSIONS
    # ========================================================================

    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array [w, x, y, z].

        Returns:
            4D numpy array
        """
        return np.array([self.w, self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        """
        Create quaternion from array [w, x, y, z].

        Args:
            arr: Array with at least 4 elements

        Returns:
            Quaternion instance
        """
        return cls(arr[0], arr[1], arr[2], arr[3])

    # ========================================================================
    # BASIC QUATERNION OPERATIONS
    # ========================================================================

    def norm(self) -> float:
        """
        Calculate quaternion norm (magnitude).

        Returns:
            ||q|| = sqrt(w² + x² + y² + z²)
        """
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Quaternion':
        """
        Return normalized (unit) quaternion.

        For rotation quaternions, this ensures ||q|| = 1.

        Returns:
            Unit quaternion q/||q||
        """
        n = self.norm()
        if n < 1e-12:
            return Quaternion(1, 0, 0, 0)  # Return identity
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def conjugate(self) -> 'Quaternion':
        """
        Return quaternion conjugate.

        For q = w + xi + yj + zk, conjugate is q* = w - xi - yj - zk.
        For unit quaternions, q* represents the inverse rotation.

        Returns:
            Conjugate quaternion q*
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> 'Quaternion':
        """
        Return quaternion inverse.

        For unit quaternions (rotations), inverse equals conjugate.
        General formula: q^(-1) = q* / ||q||²

        Returns:
            Inverse quaternion

        Raises:
            ZeroDivisionError: If quaternion is zero (norm² < 1e-12)
        """
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_sq < 1e-12:
            raise ZeroDivisionError("Cannot compute inverse of zero quaternion")

        conj = self.conjugate()
        return Quaternion(conj.w/norm_sq, conj.x/norm_sq,
                         conj.y/norm_sq, conj.z/norm_sq)

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Quaternion multiplication (Hamilton product).

        This is NOT commutative: q1 * q2 ≠ q2 * q1
        Represents composition of rotations: q1 * q2 means "first q2, then q1"

        Formula:
            (w1 + x1i + y1j + z1k) * (w2 + x2i + y2j + z2k) =
            (w1w2 - x1x2 - y1y2 - z1z2) +
            (w1x2 + x1w2 + y1z2 - z1y2)i +
            (w1y2 - x1z2 + y1w2 + z1x2)j +
            (w1z2 + x1y2 - y1x2 + z1w2)k

        Args:
            other: Right-hand quaternion

        Returns:
            Product quaternion
        """
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        return Quaternion(w, x, y, z)

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        """Element-wise addition (not commonly used for rotations)."""
        return Quaternion(self.w + other.w, self.x + other.x,
                         self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        """Element-wise subtraction (not commonly used for rotations)."""
        return Quaternion(self.w - other.w, self.x - other.x,
                         self.y - other.y, self.z - other.z)

    def __truediv__(self, scalar: float) -> 'Quaternion':
        """Scalar division."""
        return Quaternion(self.w/scalar, self.x/scalar,
                         self.y/scalar, self.z/scalar)

    def dot(self, other: 'Quaternion') -> float:
        """
        Dot product with another quaternion.

        Useful for measuring similarity between rotations.
        For unit quaternions: -1 ≤ dot ≤ 1

        Args:
            other: Other quaternion

        Returns:
            Dot product (scalar)
        """
        return self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z

    # ========================================================================
    # ROTATION MATRIX CONVERSION
    # ========================================================================

    def to_rotation_matrix(self) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.

        This matrix R rotates vectors in 3D space: v' = R @ v

        Formula (assuming unit quaternion):
        R = [[1-2(y²+z²),  2(xy-wz),    2(xz+wy)  ],
             [2(xy+wz),    1-2(x²+z²),  2(yz-wx)  ],
             [2(xz-wy),    2(yz+wx),    1-2(x²+y²)]]

        Returns:
            3x3 rotation matrix (numpy array)
        """
        # Normalize first to ensure valid rotation matrix
        q = self.normalize()
        w, x, y, z = q.w, q.x, q.y, q.z

        # Precompute repeated terms
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        return np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
            [    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)],
            [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)]
        ])

    @classmethod
    def from_rotation_matrix(cls, R: np.ndarray) -> 'Quaternion':
        """
        Create quaternion from 3x3 rotation matrix.

        Uses Shepperd's method for numerical stability.

        Args:
            R: 3x3 rotation matrix

        Returns:
            Quaternion representing the rotation
        """
        trace = np.trace(R)

        if trace > 0:
            # w is largest component
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            # x is largest component
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            # y is largest component
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            # z is largest component
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return cls(w, x, y, z).normalize()

    # ========================================================================
    # AXIS-ANGLE REPRESENTATION
    # ========================================================================

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
        """
        Create quaternion from axis-angle representation.

        Rotation of 'angle' radians around 'axis' vector.

        Formula:
            q = [cos(θ/2), vx*sin(θ/2), vy*sin(θ/2), vz*sin(θ/2)]
        where v = (vx, vy, vz) is the unit axis vector.

        Args:
            axis: 3D rotation axis (will be normalized)
            angle: Rotation angle in radians

        Returns:
            Quaternion representing the rotation
        """
        axis = np.array(axis, dtype=float)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-12:
            # No rotation
            return cls(1, 0, 0, 0)

        axis = axis / axis_norm
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)

        return cls(
            w=math.cos(half_angle),
            x=axis[0] * sin_half,
            y=axis[1] * sin_half,
            z=axis[2] * sin_half
        )

    def to_axis_angle(self) -> Tuple[np.ndarray, float]:
        """
        Convert quaternion to axis-angle representation.

        Returns:
            Tuple of (axis, angle) where:
            - axis: 3D unit vector
            - angle: rotation angle in radians [0, π]
        """
        q = self.normalize()

        # Handle identity rotation
        if abs(q.w) >= 1.0:
            return np.array([0, 0, 1], dtype=float), 0.0

        angle = 2.0 * math.acos(np.clip(q.w, -1.0, 1.0))
        sin_half = math.sqrt(1.0 - q.w**2)

        if sin_half < 1e-12:
            # Near identity, return arbitrary axis
            return np.array([0, 0, 1], dtype=float), 0.0

        axis = np.array([q.x, q.y, q.z]) / sin_half

        return axis, angle

    # ========================================================================
    # EULER ANGLES
    # ========================================================================

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float,
                   order: str = 'XYZ') -> 'Quaternion':
        """
        Create quaternion from Euler angles.

        Intrinsic rotations (rotate around moving axes).

        Args:
            roll: Rotation around X-axis (radians)
            pitch: Rotation around Y-axis (radians)
            yaw: Rotation around Z-axis (radians)
            order: Rotation order ('XYZ', 'ZYX', etc.)

        Returns:
            Quaternion representing the rotation
        """
        # Create individual rotation quaternions
        qx = cls.from_axis_angle(np.array([1, 0, 0]), roll)
        qy = cls.from_axis_angle(np.array([0, 1, 0]), pitch)
        qz = cls.from_axis_angle(np.array([0, 0, 1]), yaw)

        # Combine based on order (right to left application)
        if order == 'XYZ':
            return qz * qy * qx
        elif order == 'ZYX':
            return qx * qy * qz
        elif order == 'YXZ':
            return qz * qx * qy
        elif order == 'ZXY':
            return qy * qx * qz
        elif order == 'XZY':
            return qy * qz * qx
        elif order == 'YZX':
            return qx * qz * qy
        else:
            raise ValueError(f"Unknown Euler order: {order}")

    def to_euler(self, order: str = 'XYZ') -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles.

        Args:
            order: Rotation order ('XYZ' or 'ZYX')

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        q = self.normalize()

        if order == 'XYZ':
            # Roll (X-axis)
            sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1 - 2 * (q.x**2 + q.y**2)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            # Pitch (Y-axis)
            sinp = 2 * (q.w * q.y - q.z * q.x)
            pitch = math.asin(np.clip(sinp, -1.0, 1.0))

            # Yaw (Z-axis)
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
            yaw = math.atan2(siny_cosp, cosy_cosp)

        elif order == 'ZYX':
            # Roll (X-axis)
            sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1 - 2 * (q.x**2 + q.y**2)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            # Pitch (Y-axis)
            sinp = 2 * (q.w * q.y - q.z * q.x)
            pitch = math.asin(np.clip(sinp, -1.0, 1.0))

            # Yaw (Z-axis)
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
            yaw = math.atan2(siny_cosp, cosy_cosp)
        else:
            raise ValueError(f"Unsupported Euler order: {order}")

        return roll, pitch, yaw

    # ========================================================================
    # VECTOR ROTATION
    # ========================================================================

    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Rotate a 3D vector using this quaternion.

        Formula: v' = q * v * q*
        where v is treated as pure quaternion (0, vx, vy, vz)

        Args:
            v: 3D vector to rotate

        Returns:
            Rotated 3D vector
        """
        # Convert vector to pure quaternion
        v_quat = Quaternion(0, v[0], v[1], v[2])

        # Perform rotation: q * v * q*
        rotated = self * v_quat * self.conjugate()

        return np.array([rotated.x, rotated.y, rotated.z])

    def rotate_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Rotate multiple 3D vectors efficiently.

        Args:
            vectors: Nx3 array of vectors

        Returns:
            Nx3 array of rotated vectors
        """
        R = self.to_rotation_matrix()
        return vectors @ R.T

    # ========================================================================
    # INTERPOLATION
    # ========================================================================

    def slerp(self, other: 'Quaternion', t: float) -> 'Quaternion':
        """
        Spherical Linear Interpolation (SLERP) between two quaternions.

        Provides smooth interpolation along the shortest arc on the 4D unit sphere.
        Critical for animation and smooth rotations.

        Args:
            other: Target quaternion
            t: Interpolation parameter [0, 1]
                t=0 returns self, t=1 returns other

        Returns:
            Interpolated quaternion
        """
        q1 = self.normalize()
        q2 = other.normalize()

        # Compute dot product
        dot = q1.dot(q2)

        # If negative dot, negate one quaternion to take shorter path
        if dot < 0.0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot

        # Clamp dot product
        dot = np.clip(dot, -1.0, 1.0)

        # Quaternions very close? Use linear interpolation
        if dot > 0.9995:
            result = Quaternion(
                q1.w + t * (q2.w - q1.w),
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z)
            )
            return result.normalize()

        # Calculate angle between quaternions
        theta = math.acos(dot)
        sin_theta = math.sin(theta)

        # Compute interpolation coefficients
        w1 = math.sin((1 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta

        # Interpolate
        return Quaternion(
            w1 * q1.w + w2 * q2.w,
            w1 * q1.x + w2 * q2.x,
            w1 * q1.y + w2 * q2.y,
            w1 * q1.z + w2 * q2.z
        )

    # ========================================================================
    # LOGARITHM AND EXPONENTIAL MAPS
    # ========================================================================

    def log(self) -> np.ndarray:
        """
        Quaternion logarithm (maps to Lie algebra so(3)).

        For unit quaternion q = [cos(θ/2), v*sin(θ/2)]:
        log(q) = [0, v*θ/2]

        Returns:
            3D vector in tangent space (axis * angle/2)
        """
        q = self.normalize()

        if abs(q.w) >= 1.0:
            return np.zeros(3)

        angle = 2.0 * math.acos(np.clip(q.w, -1.0, 1.0))
        sin_half = math.sqrt(1.0 - q.w**2)

        if sin_half < 1e-12:
            return np.zeros(3)

        axis = np.array([q.x, q.y, q.z]) / sin_half

        return axis * (angle / 2.0)

    @classmethod
    def exp(cls, v: np.ndarray) -> 'Quaternion':
        """
        Quaternion exponential (maps from Lie algebra so(3)).

        For tangent vector v = axis * angle/2:
        exp(v) = [cos(||v||), (v/||v||)*sin(||v||)]

        Args:
            v: 3D vector in tangent space

        Returns:
            Quaternion
        """
        theta = np.linalg.norm(v)

        if theta < 1e-12:
            return cls(1, 0, 0, 0)

        axis = v / theta
        return cls.from_axis_angle(axis, float(2.0 * theta))

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def __repr__(self) -> str:
        """String representation."""
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"

    def __str__(self) -> str:
        """Human-readable string."""
        return f"[{self.w:.4f} + {self.x:.4f}i + {self.y:.4f}j + {self.z:.4f}k]"

    def is_unit(self, tolerance: float = 1e-6) -> bool:
        """
        Check if quaternion is a unit quaternion.

        Args:
            tolerance: Numerical tolerance

        Returns:
            True if ||q|| ≈ 1
        """
        return abs(self.norm() - 1.0) < tolerance


# ============================================================================
# STANDALONE UTILITY FUNCTIONS
# ============================================================================

def quaternion_distance(q1: Quaternion, q2: Quaternion) -> float:
    """
    Angular distance between two quaternions.

    Returns the angle (in radians) of the rotation needed to go from q1 to q2.

    Args:
        q1: First quaternion
        q2: Second quaternion

    Returns:
        Angular distance in radians [0, π]
    """
    q1 = q1.normalize()
    q2 = q2.normalize()
    dot = abs(q1.dot(q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * math.acos(dot)


def quaternion_average(quaternions: List[Quaternion], weights: Optional[List[float]] = None) -> Quaternion:
    """
    Compute weighted average of quaternions using iterative method.

    This is non-trivial because quaternions live on a sphere.
    Uses iterative gradient descent on the sphere.

    Args:
        quaternions: List of quaternions to average
        weights: Optional weights (normalized internally)

    Returns:
        Average quaternion
    """
    if not quaternions:
        return Quaternion(1, 0, 0, 0)

    if len(quaternions) == 1:
        return quaternions[0].normalize()

    # Initialize weights
    if weights is None:
        weights = [1.0] * len(quaternions)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Initialize with first quaternion
    q_avg = quaternions[0].normalize()

    # Iterative averaging
    for _ in range(10):  # Usually converges in a few iterations
        # Compute error vectors in tangent space
        v_sum = np.zeros(3)

        for q, w in zip(quaternions, weights):
            q_norm = q.normalize()
            # Relative rotation
            q_rel = q_avg.inverse() * q_norm
            # Log map to tangent space
            v = q_rel.log()
            v_sum += w * v

        # Update average
        q_avg = q_avg * Quaternion.exp(v_sum)

        # Check convergence
        if np.linalg.norm(v_sum) < 1e-6:
            break

    return q_avg.normalize()


def rotation_matrix_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert 3x3 rotation matrix to axis-angle representation.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tuple of (axis, angle)
    """
    q = Quaternion.from_rotation_matrix(R)
    return q.to_axis_angle()


def axis_angle_to_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert axis-angle to 3x3 rotation matrix.

    Args:
        axis: 3D rotation axis
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    q = Quaternion.from_axis_angle(axis, angle)
    return q.to_rotation_matrix()


# ============================================================================
# DEMONSTRATION AND EXAMPLES
# ============================================================================

def demo_quaternion_rotations():
    """
    Demonstrate quaternion rotation capabilities.
    """
    print("=" * 70)
    print("QUATERNION ROTATION DEMONSTRATIONS")
    print("=" * 70)

    # Example 1: Basic rotation
    print("\n1. Basic 90° rotation around Z-axis")
    print("-" * 70)
    q = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)
    print(f"Quaternion: {q}")

    v = np.array([1.0, 0.0, 0.0])
    v_rot = q.rotate_vector(v)
    print(f"Vector {v} rotated to {v_rot}")
    print(f"Expected: [0, 1, 0]")

    # Example 2: Rotation matrix
    print("\n2. Conversion to rotation matrix")
    print("-" * 70)
    R = q.to_rotation_matrix()
    print("Rotation matrix:")
    print(R)

    # Example 3: Quaternion multiplication
    print("\n3. Quaternion multiplication (rotation composition)")
    print("-" * 70)
    q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)  # 90° around Z
    q2 = Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi/2)  # 90° around X
    q_combined = q1 * q2
    print(f"q1 (90° Z-axis): {q1}")
    print(f"q2 (90° X-axis): {q2}")
    print(f"q1 * q2: {q_combined}")

    v = np.array([0, 0, 1])
    v_rot = q_combined.rotate_vector(v)
    print(f"\nVector {v} after combined rotation: {v_rot}")

    # Example 4: SLERP interpolation
    print("\n4. SLERP interpolation")
    print("-" * 70)
    q_start = Quaternion(1, 0, 0, 0)  # Identity
    q_end = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi)  # 180° around Z

    print("Interpolating from identity to 180° Z-rotation:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q_interp = q_start.slerp(q_end, t)
        axis, angle = q_interp.to_axis_angle()
        print(f"  t={t:.2f}: angle={np.degrees(angle):.1f}°, axis={axis}")

    # Example 5: Euler angles
    print("\n5. Euler angle conversion")
    print("-" * 70)
    roll, pitch, yaw = np.pi/4, np.pi/6, np.pi/3
    q_euler = Quaternion.from_euler(roll, pitch, yaw, order='XYZ')
    print(f"Input Euler (rad): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
    print(f"Quaternion: {q_euler}")

    r2, p2, y2 = q_euler.to_euler(order='XYZ')
    print(f"Recovered Euler: roll={r2:.3f}, pitch={p2:.3f}, yaw={y2:.3f}")

    # Example 6: Multiple vectors rotation
    print("\n6. Rotating multiple vectors efficiently")
    print("-" * 70)
    q_rot = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/4)  # 45° Z-rotation
    vectors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ])
    rotated = q_rot.rotate_vectors(vectors)
    print("Input vectors:")
    print(vectors)
    print("\nRotated vectors (45° around Z):")
    print(rotated)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run demonstrations
    demo_quaternion_rotations()

    # Additional tests
    print("\n\nADDITIONAL TESTS")
    print("=" * 70)

    # Test 1: Inverse property
    print("\n✓ Testing q * q^(-1) = identity")
    q = Quaternion.from_axis_angle(np.array([1, 2, 3]), 1.5)
    q_inv = q.inverse()
    q_identity = q * q_inv
    print(f"  Result: {q_identity} (should be ≈ [1, 0, 0, 0])")

    # Test 2: Rotation preservation
    print("\n✓ Testing rotation preserves vector length")
    v = np.array([3.0, 4.0, 0.0])
    v_norm_before = np.linalg.norm(v)
    v_rot = q.rotate_vector(v)
    v_norm_after = np.linalg.norm(v_rot)
    print(f"  Before: ||v|| = {v_norm_before:.6f}")
    print(f"  After:  ||v|| = {v_norm_after:.6f}")
    print(f"  Preserved: {np.isclose(v_norm_before, v_norm_after)}")

    # Test 3: Matrix conversion consistency
    print("\n✓ Testing rotation matrix roundtrip")
    q_orig = Quaternion.from_axis_angle(np.array([1, 1, 1]), np.pi/3)
    R = q_orig.to_rotation_matrix()
    q_recovered = Quaternion.from_rotation_matrix(R)
    distance = quaternion_distance(q_orig, q_recovered)
    print(f"  Angular distance: {np.degrees(distance):.6f}° (should be ≈ 0)")

    print("\n" + "=" * 70)
    print("All tests completed!")
