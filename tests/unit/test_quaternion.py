"""
Unit tests for Quaternion mathematics

Tests cover:
- Basic quaternion operations (norm, normalize, conjugate, inverse)
- Quaternion multiplication
- Rotation matrix conversion
- Axis-angle conversion
- Euler angle conversion
- Vector rotation
- SLERP interpolation
- Mathematical properties and edge cases
"""

import pytest
import numpy as np
import math

from intelligence.quaternion import Quaternion


class TestQuaternionBasics:
    """Test basic quaternion operations"""

    def test_initialization(self):
        """Test quaternion initialization"""
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_initialization_default(self):
        """Test default quaternion is identity"""
        q = Quaternion()
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_to_array(self):
        """Test conversion to numpy array"""
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_array()

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (4,)
        np.testing.assert_array_equal(arr, [1, 2, 3, 4])

    def test_from_array(self):
        """Test creation from numpy array"""
        arr = np.array([0.5, 0.5, 0.5, 0.5])
        q = Quaternion.from_array(arr)

        assert q.w == 0.5
        assert q.x == 0.5
        assert q.y == 0.5
        assert q.z == 0.5

    def test_norm_identity(self):
        """Test norm of identity quaternion"""
        q = Quaternion(1, 0, 0, 0)
        assert q.norm() == pytest.approx(1.0)

    def test_norm_general(self):
        """Test norm calculation"""
        q = Quaternion(2, 2, 1, 0)
        expected = math.sqrt(2**2 + 2**2 + 1**2 + 0**2)
        assert q.norm() == pytest.approx(expected)

    def test_normalize_identity(self):
        """Test normalizing identity quaternion"""
        q = Quaternion(1, 0, 0, 0)
        q_norm = q.normalize()

        assert q_norm.norm() == pytest.approx(1.0)
        assert q_norm.w == pytest.approx(1.0)

    def test_normalize_general(self):
        """Test normalization"""
        q = Quaternion(2, 2, 1, 0)
        q_norm = q.normalize()

        assert q_norm.norm() == pytest.approx(1.0)

        # Check proportions preserved
        norm = q.norm()
        assert q_norm.w == pytest.approx(2 / norm)
        assert q_norm.x == pytest.approx(2 / norm)
        assert q_norm.y == pytest.approx(1 / norm)

    def test_normalize_zero_quaternion(self):
        """Test normalizing zero quaternion returns identity"""
        q = Quaternion(0, 0, 0, 0)
        q_norm = q.normalize()

        assert q_norm.w == 1.0
        assert q_norm.x == 0.0
        assert q_norm.y == 0.0
        assert q_norm.z == 0.0

    def test_conjugate(self):
        """Test quaternion conjugate"""
        q = Quaternion(1, 2, 3, 4)
        q_conj = q.conjugate()

        assert q_conj.w == 1
        assert q_conj.x == -2
        assert q_conj.y == -3
        assert q_conj.z == -4

    def test_conjugate_identity(self):
        """Test conjugate of identity is itself"""
        q = Quaternion(1, 0, 0, 0)
        q_conj = q.conjugate()

        assert q_conj.w == 1
        assert q_conj.x == 0
        assert q_conj.y == 0
        assert q_conj.z == 0

    def test_inverse(self):
        """Test quaternion inverse"""
        q = Quaternion(1, 0, 0, 0)
        q_inv = q.inverse()

        # Identity inverse is itself
        assert q_inv.w == pytest.approx(1.0)
        assert q_inv.x == pytest.approx(0.0)

    def test_inverse_unit_quaternion(self):
        """Test that for unit quaternions, inverse equals conjugate"""
        q = Quaternion(0.5, 0.5, 0.5, 0.5)
        q_norm = q.normalize()

        q_inv = q_norm.inverse()
        q_conj = q_norm.conjugate()

        assert q_inv.w == pytest.approx(q_conj.w)
        assert q_inv.x == pytest.approx(q_conj.x)
        assert q_inv.y == pytest.approx(q_conj.y)
        assert q_inv.z == pytest.approx(q_conj.z)

    def test_inverse_zero_quaternion(self):
        """Test that inverse of zero quaternion raises error"""
        q = Quaternion(0, 0, 0, 0)

        with pytest.raises(ZeroDivisionError):
            q.inverse()

    def test_dot_product(self):
        """Test quaternion dot product"""
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(1, 0, 0, 0)

        dot = q1.dot(q2)
        assert dot == 1.0

    def test_dot_product_orthogonal(self):
        """Test dot product of orthogonal quaternions"""
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0, 1, 0, 0)

        dot = q1.dot(q2)
        assert dot == 0.0


class TestQuaternionArithmetic:
    """Test quaternion arithmetic operations"""

    def test_addition(self):
        """Test quaternion addition"""
        q1 = Quaternion(1, 2, 3, 4)
        q2 = Quaternion(5, 6, 7, 8)
        q_sum = q1 + q2

        assert q_sum.w == 6
        assert q_sum.x == 8
        assert q_sum.y == 10
        assert q_sum.z == 12

    def test_subtraction(self):
        """Test quaternion subtraction"""
        q1 = Quaternion(5, 6, 7, 8)
        q2 = Quaternion(1, 2, 3, 4)
        q_diff = q1 - q2

        assert q_diff.w == 4
        assert q_diff.x == 4
        assert q_diff.y == 4
        assert q_diff.z == 4

    def test_scalar_division(self):
        """Test scalar division"""
        q = Quaternion(4, 8, 12, 16)
        q_div = q / 4

        assert q_div.w == 1
        assert q_div.x == 2
        assert q_div.y == 3
        assert q_div.z == 4

    def test_multiplication_identity(self):
        """Test multiplication with identity"""
        q = Quaternion(1, 2, 3, 4)
        identity = Quaternion(1, 0, 0, 0)

        result = q * identity
        assert result.w == pytest.approx(q.w)
        assert result.x == pytest.approx(q.x)
        assert result.y == pytest.approx(q.y)
        assert result.z == pytest.approx(q.z)

    def test_multiplication_non_commutative(self):
        """Test that quaternion multiplication is non-commutative"""
        q1 = Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi/4)
        q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), np.pi/4)

        result1 = q1 * q2
        result2 = q2 * q1

        # Results should be different
        assert not (
            result1.w == pytest.approx(result2.w) and
            result1.x == pytest.approx(result2.x) and
            result1.y == pytest.approx(result2.y) and
            result1.z == pytest.approx(result2.z)
        )

    def test_multiplication_inverse(self):
        """Test that q * q^-1 = identity"""
        q = Quaternion(0.6, 0.8, 0.0, 0.0).normalize()
        q_inv = q.inverse()

        result = q * q_inv

        # Should get identity quaternion
        assert result.w == pytest.approx(1.0, abs=1e-6)
        assert result.x == pytest.approx(0.0, abs=1e-6)
        assert result.y == pytest.approx(0.0, abs=1e-6)
        assert result.z == pytest.approx(0.0, abs=1e-6)


class TestAxisAngleConversion:
    """Test axis-angle conversions"""

    def test_from_axis_angle_identity(self):
        """Test zero rotation"""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), 0.0)

        assert q.w == pytest.approx(1.0)
        assert q.norm() == pytest.approx(1.0)

    def test_from_axis_angle_90_deg_z(self):
        """Test 90° rotation around Z-axis"""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)

        # For 90° rotation: w = cos(45°), z = sin(45°)
        assert q.w == pytest.approx(math.cos(np.pi/4))
        assert q.z == pytest.approx(math.sin(np.pi/4))
        assert q.x == pytest.approx(0.0)
        assert q.y == pytest.approx(0.0)
        assert q.norm() == pytest.approx(1.0)

    def test_from_axis_angle_180_deg_x(self):
        """Test 180° rotation around X-axis"""
        q = Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi)

        assert q.w == pytest.approx(0.0, abs=1e-10)
        assert q.x == pytest.approx(1.0)
        assert q.y == pytest.approx(0.0)
        assert q.z == pytest.approx(0.0)

    def test_from_axis_angle_normalization(self):
        """Test that non-unit axis is normalized"""
        q = Quaternion.from_axis_angle(np.array([3, 0, 0]), np.pi/2)

        # Should be same as unit x-axis
        q_expected = Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi/2)

        assert q.w == pytest.approx(q_expected.w)
        assert q.x == pytest.approx(q_expected.x)

    def test_from_axis_angle_zero_axis(self):
        """Test zero axis returns identity"""
        q = Quaternion.from_axis_angle(np.array([0, 0, 0]), np.pi/2)

        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_to_axis_angle_identity(self):
        """Test converting identity quaternion to axis-angle"""
        q = Quaternion(1, 0, 0, 0)
        axis, angle = q.to_axis_angle()

        assert angle == pytest.approx(0.0)
        assert np.linalg.norm(axis) == pytest.approx(1.0)

    def test_to_axis_angle_90_deg(self):
        """Test converting 90° rotation to axis-angle"""
        original_axis = np.array([0, 0, 1])
        original_angle = np.pi/2

        q = Quaternion.from_axis_angle(original_axis, original_angle)
        axis, angle = q.to_axis_angle()

        assert angle == pytest.approx(original_angle)
        np.testing.assert_array_almost_equal(axis, original_axis)

    def test_axis_angle_roundtrip(self):
        """Test roundtrip conversion axis-angle -> quaternion -> axis-angle"""
        original_axis = np.array([1, 1, 1]) / np.sqrt(3)
        original_angle = np.pi/3

        q = Quaternion.from_axis_angle(original_axis, original_angle)
        axis, angle = q.to_axis_angle()

        assert angle == pytest.approx(original_angle)
        np.testing.assert_array_almost_equal(axis, original_axis)


class TestRotationMatrixConversion:
    """Test rotation matrix conversions"""

    def test_to_rotation_matrix_identity(self):
        """Test identity quaternion to rotation matrix"""
        q = Quaternion(1, 0, 0, 0)
        R = q.to_rotation_matrix()

        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_to_rotation_matrix_shape(self):
        """Test rotation matrix has correct shape"""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/4)
        R = q.to_rotation_matrix()

        assert R.shape == (3, 3)

    def test_rotation_matrix_orthogonal(self):
        """Test that rotation matrix is orthogonal (R^T * R = I)"""
        q = Quaternion.from_axis_angle(np.array([1, 1, 1]), np.pi/3)
        R = q.to_rotation_matrix()

        product = R.T @ R
        np.testing.assert_array_almost_equal(product, np.eye(3))

    def test_rotation_matrix_determinant(self):
        """Test that rotation matrix has determinant +1"""
        q = Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi/2)
        R = q.to_rotation_matrix()

        det = np.linalg.det(R)
        assert det == pytest.approx(1.0)

    def test_from_rotation_matrix_identity(self):
        """Test creating quaternion from identity matrix"""
        R = np.eye(3)
        q = Quaternion.from_rotation_matrix(R)

        assert q.norm() == pytest.approx(1.0)
        # Should be close to identity quaternion
        assert abs(q.w) == pytest.approx(1.0, abs=1e-6)

    def test_rotation_matrix_roundtrip(self):
        """Test roundtrip quaternion -> matrix -> quaternion"""
        q_original = Quaternion.from_axis_angle(np.array([1, 1, 1]), np.pi/4)
        q_original = q_original.normalize()

        R = q_original.to_rotation_matrix()
        q_reconstructed = Quaternion.from_rotation_matrix(R)

        # Quaternions q and -q represent the same rotation
        # So check if either q == q_reconstructed or q == -q_reconstructed
        same_sign = (
            q_original.w * q_reconstructed.w +
            q_original.x * q_reconstructed.x +
            q_original.y * q_reconstructed.y +
            q_original.z * q_reconstructed.z
        )

        assert abs(same_sign) == pytest.approx(1.0, abs=1e-5)


class TestVectorRotation:
    """Test vector rotation functionality"""

    def test_rotate_vector_identity(self):
        """Test rotating with identity quaternion"""
        q = Quaternion(1, 0, 0, 0)
        v = np.array([1, 2, 3])

        v_rotated = q.rotate_vector(v)

        np.testing.assert_array_almost_equal(v_rotated, v)

    def test_rotate_vector_90_deg_z(self):
        """Test 90° rotation around Z-axis"""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)
        v = np.array([1, 0, 0])  # X-axis

        v_rotated = q.rotate_vector(v)

        # X-axis rotated 90° around Z should give Y-axis
        expected = np.array([0, 1, 0])
        np.testing.assert_array_almost_equal(v_rotated, expected, decimal=6)

    def test_rotate_vector_180_deg_x(self):
        """Test 180° rotation around X-axis"""
        q = Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi)
        v = np.array([0, 1, 0])  # Y-axis

        v_rotated = q.rotate_vector(v)

        # Y-axis rotated 180° around X should give -Y-axis
        expected = np.array([0, -1, 0])
        np.testing.assert_array_almost_equal(v_rotated, expected, decimal=6)

    def test_rotation_preserves_length(self):
        """Test that rotation preserves vector length"""
        q = Quaternion.from_axis_angle(np.array([1, 1, 1]), np.pi/3)
        v = np.array([3, 4, 5])

        v_rotated = q.rotate_vector(v)

        original_length = np.linalg.norm(v)
        rotated_length = np.linalg.norm(v_rotated)

        assert rotated_length == pytest.approx(original_length)

    def test_rotate_multiple_vectors(self):
        """Test rotating multiple vectors at once"""
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)
        vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ])

        rotated = q.rotate_vectors(vectors)

        assert rotated.shape == (3, 3)

        # Check first vector (should rotate to Y-axis)
        np.testing.assert_array_almost_equal(rotated[0], [0, 1, 0], decimal=6)

        # Check second vector (should rotate to -X-axis)
        np.testing.assert_array_almost_equal(rotated[1], [-1, 0, 0], decimal=6)

    def test_composition_of_rotations(self):
        """Test that composing rotations works correctly"""
        # First rotate 90° around Z
        q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)
        # Then rotate 90° around Y
        q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), np.pi/2)

        # Composed rotation
        q_composed = q2 * q1

        v = np.array([1, 0, 0])

        # Apply rotations separately
        v1 = q1.rotate_vector(v)
        v2 = q2.rotate_vector(v1)

        # Apply composed rotation
        v_composed = q_composed.rotate_vector(v)

        np.testing.assert_array_almost_equal(v2, v_composed)


class TestEulerAngles:
    """Test Euler angle conversions"""

    def test_from_euler_identity(self):
        """Test creating identity from zero Euler angles"""
        q = Quaternion.from_euler(0, 0, 0)

        assert q.norm() == pytest.approx(1.0)

    def test_from_euler_90_deg_single_axis(self):
        """Test 90° rotation around single axis"""
        q = Quaternion.from_euler(np.pi/2, 0, 0)  # Roll only

        # Should be same as axis-angle
        q_expected = Quaternion.from_axis_angle(np.array([1, 0, 0]), np.pi/2)

        assert q.w == pytest.approx(q_expected.w)
        assert q.x == pytest.approx(q_expected.x)

    def test_euler_roundtrip_xyz(self):
        """Test roundtrip Euler -> quaternion -> Euler (XYZ order)"""
        roll, pitch, yaw = 0.1, 0.2, 0.3

        q = Quaternion.from_euler(roll, pitch, yaw, order='XYZ')
        roll2, pitch2, yaw2 = q.to_euler(order='XYZ')

        assert roll2 == pytest.approx(roll, abs=1e-6)
        assert pitch2 == pytest.approx(pitch, abs=1e-6)
        assert yaw2 == pytest.approx(yaw, abs=1e-6)

    def test_from_euler_different_orders(self):
        """Test that different orders produce different results"""
        angles = (np.pi/6, np.pi/4, np.pi/3)

        q_xyz = Quaternion.from_euler(*angles, order='XYZ')
        q_zyx = Quaternion.from_euler(*angles, order='ZYX')

        # Should be different
        assert not (
            q_xyz.w == pytest.approx(q_zyx.w) and
            q_xyz.x == pytest.approx(q_zyx.x)
        )


class TestSLERP:
    """Test Spherical Linear Interpolation"""

    def test_slerp_t0(self):
        """Test SLERP with t=0 returns first quaternion"""
        q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), 0.0)
        q2 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)

        q_interp = q1.slerp(q2, 0.0)

        assert q_interp.w == pytest.approx(q1.w)
        assert q_interp.x == pytest.approx(q1.x)
        assert q_interp.y == pytest.approx(q1.y)
        assert q_interp.z == pytest.approx(q1.z)

    def test_slerp_t1(self):
        """Test SLERP with t=1 returns second quaternion"""
        q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), 0.0)
        q2 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)

        q_interp = q1.slerp(q2, 1.0)

        assert q_interp.w == pytest.approx(q2.w)
        assert q_interp.x == pytest.approx(q2.x)
        assert q_interp.y == pytest.approx(q2.y)
        assert q_interp.z == pytest.approx(q2.z)

    def test_slerp_midpoint(self):
        """Test SLERP at t=0.5"""
        q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), 0.0)
        q2 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/2)

        q_mid = q1.slerp(q2, 0.5)

        # Midpoint should be 45° rotation
        q_expected = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/4)

        assert q_mid.w == pytest.approx(q_expected.w, abs=1e-6)
        assert q_mid.z == pytest.approx(q_expected.z, abs=1e-6)

    def test_slerp_preserves_unit_norm(self):
        """Test that SLERP always produces unit quaternions"""
        q1 = Quaternion.from_axis_angle(np.array([1, 0, 0]), 0.3)
        q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.7)

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            q_interp = q1.slerp(q2, t)
            assert q_interp.norm() == pytest.approx(1.0)

    def test_slerp_smooth_path(self):
        """Test that SLERP produces smooth path"""
        q1 = Quaternion.from_axis_angle(np.array([0, 0, 1]), 0.0)
        q2 = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi)

        # Generate interpolated rotations
        angles = []
        for t in np.linspace(0, 1, 10):
            q = q1.slerp(q2, t)
            axis, angle = q.to_axis_angle()
            angles.append(angle)

        # Angles should be monotonically increasing
        for i in range(len(angles) - 1):
            assert angles[i+1] >= angles[i]


class TestEdgeCases:
    """Test edge cases and numerical stability"""

    def test_very_small_rotation(self):
        """Test very small rotation angle"""
        q = Quaternion.from_axis_angle(np.array([1, 0, 0]), 1e-10)

        # Should be close to identity
        assert q.w == pytest.approx(1.0, abs=1e-9)
        assert q.norm() == pytest.approx(1.0)

    def test_very_large_angle(self):
        """Test angle larger than 2π"""
        angle = 3 * np.pi  # 540°
        q = Quaternion.from_axis_angle(np.array([0, 0, 1]), angle)

        # Should still be valid unit quaternion
        assert q.norm() == pytest.approx(1.0)

    def test_negative_angle(self):
        """Test negative rotation angle"""
        q_pos = Quaternion.from_axis_angle(np.array([0, 0, 1]), np.pi/4)
        q_neg = Quaternion.from_axis_angle(np.array([0, 0, 1]), -np.pi/4)

        # Negative angle should be inverse rotation
        q_composed = q_pos * q_neg

        # Should get identity
        assert q_composed.w == pytest.approx(1.0, abs=1e-6)

    def test_denormalized_quaternion_operations(self):
        """Test operations with non-unit quaternions"""
        q = Quaternion(2, 2, 1, 0)  # Not normalized

        # to_rotation_matrix should normalize internally
        R = q.to_rotation_matrix()
        det = np.linalg.det(R)

        assert det == pytest.approx(1.0)

    def test_gimbal_lock_avoidance(self):
        """Test that quaternions avoid gimbal lock"""
        # Create a rotation known to cause gimbal lock in Euler angles
        q = Quaternion.from_euler(0, np.pi/2, 0)

        # Should still produce valid rotation matrix
        R = q.to_rotation_matrix()
        assert np.linalg.det(R) == pytest.approx(1.0)

        # Should be able to rotate vectors
        v = np.array([1, 0, 0])
        v_rot = q.rotate_vector(v)
        assert np.linalg.norm(v_rot) == pytest.approx(np.linalg.norm(v))


class TestMathematicalProperties:
    """Test important mathematical properties of quaternions"""

    def test_identity_is_multiplicative_identity(self):
        """Test that identity quaternion is multiplicative identity"""
        identity = Quaternion(1, 0, 0, 0)
        q = Quaternion.from_axis_angle(np.array([1, 1, 1]), 0.7)

        left_mult = identity * q
        right_mult = q * identity

        assert left_mult.w == pytest.approx(q.w)
        assert right_mult.w == pytest.approx(q.w)

    def test_multiplication_associativity(self):
        """Test that quaternion multiplication is associative"""
        q1 = Quaternion.from_axis_angle(np.array([1, 0, 0]), 0.3)
        q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.5)
        q3 = Quaternion.from_axis_angle(np.array([0, 0, 1]), 0.7)

        result1 = (q1 * q2) * q3
        result2 = q1 * (q2 * q3)

        assert result1.w == pytest.approx(result2.w)
        assert result1.x == pytest.approx(result2.x)
        assert result1.y == pytest.approx(result2.y)
        assert result1.z == pytest.approx(result2.z)

    def test_inverse_property(self):
        """Test that q * q^-1 = identity for any quaternion"""
        q = Quaternion(0.6, 0.8, 0.0, 0.0)
        q_inv = q.inverse()

        product = q * q_inv

        # Allow for numerical error
        assert product.norm() == pytest.approx(1.0, abs=1e-6)
        assert abs(product.w) == pytest.approx(1.0, abs=1e-6)

    def test_conjugate_of_product(self):
        """Test that (q1 * q2)* = q2* * q1*"""
        q1 = Quaternion.from_axis_angle(np.array([1, 0, 0]), 0.3)
        q2 = Quaternion.from_axis_angle(np.array([0, 1, 0]), 0.5)

        product_conj = (q1 * q2).conjugate()
        conj_product = q2.conjugate() * q1.conjugate()

        assert product_conj.w == pytest.approx(conj_product.w)
        assert product_conj.x == pytest.approx(conj_product.x)
        assert product_conj.y == pytest.approx(conj_product.y)
        assert product_conj.z == pytest.approx(conj_product.z)
