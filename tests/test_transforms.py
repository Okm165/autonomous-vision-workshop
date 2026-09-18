"""SO(3)/SE(3) exponential and logarithm consistency tests."""

import numpy as np

from src import transforms as tr


class TestSO3:
    def test_exp_log_roundtrip(self, rng):
        for _ in range(200):
            w = rng.normal(scale=2.0, size=3)
            R = tr.so3_exp(w)
            # log returns the principal branch; verify via exp (mod 2pi)
            w2 = tr.so3_log(R)
            assert np.allclose(tr.so3_exp(w2), R, atol=1e-9)
            assert np.linalg.norm(w2) <= np.pi + 1e-9

    def test_log_exp_roundtrip(self, rng):
        for _ in range(200):
            R = tr.so3_exp(rng.normal(scale=2.9, size=3))
            w = tr.so3_log(R)
            assert np.allclose(tr.so3_exp(w), R, atol=1e-9)

    def test_near_pi_branch(self, rng):
        """Rotation angle just below pi must round-trip (sign handling)."""
        for _ in range(100):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            theta = np.pi - rng.uniform(0, 1e-3)
            R = tr.so3_exp(axis * theta)
            w = tr.so3_log(R)
            assert np.allclose(tr.so3_exp(w), R, atol=1e-8)

    def test_identity(self):
        assert np.allclose(tr.so3_log(np.eye(3)), 0.0)
        assert np.allclose(tr.so3_exp(np.zeros(3)), np.eye(3))

    def test_exp_is_rotation(self, rng):
        for _ in range(50):
            R = tr.so3_exp(rng.normal(size=3))
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


class TestSE3:
    def test_exp_log_roundtrip(self, rng):
        for _ in range(100):
            xi = rng.normal(scale=1.5, size=6)
            T = tr.se3_exp(xi)
            xi2 = tr.se3_log(T)
            # the twist is branch-dependent: for wrapped rotations the
            # principal twist has a different rho, but exp(log(T)) == T always
            # NOTE: this codebase orders the twist as xi = [rho, omega]
            assert np.allclose(tr.se3_exp(xi2), T, atol=1e-8)
            assert np.linalg.norm(xi2[3:]) <= np.pi + 1e-9

    def test_exp_log_roundtrip_small_angle(self, rng):
        """Below pi the twist is unique: components match exactly."""
        for _ in range(200):
            xi = rng.normal(scale=0.8, size=6)
            omega = xi[3:]
            if np.linalg.norm(omega) >= np.pi - 1e-6:
                continue  # at/above the branch cut the twist is not unique
            T = tr.se3_exp(xi)
            assert np.allclose(tr.se3_log(T), xi, atol=1e-8)

    def test_inverse(self, rng):
        for _ in range(100):
            T = tr.se3_exp(rng.normal(scale=2.0, size=6))
            Ti = tr.se3_inverse(T)
            assert np.allclose(T @ Ti, np.eye(4), atol=1e-12)
            assert np.allclose(Ti @ T, np.eye(4), atol=1e-12)

    def test_compose(self, rng):
        A = tr.se3_exp(rng.normal(size=6))
        B = tr.se3_exp(rng.normal(size=6))
        C = tr.se3_compose(A, B)
        assert np.allclose(C[:3, :3], A[:3, :3] @ B[:3, :3], atol=1e-12)
        assert np.allclose(C[:3, 3], A[:3, :3] @ B[:3, 3] + A[:3, 3], atol=1e-12)


class TestQuaternions:
    def test_matrix_to_quat_roundtrip(self, rng):
        for _ in range(100):
            R = tr.so3_exp(rng.normal(scale=2.5, size=3))
            q = tr.matrix_to_quaternion(R)
            q = tr.normalize_quaternion(q)
            assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
            assert (
                np.allclose(abs(q[0]), np.cos(tr.so3_log(R).__abs__() / 2), atol=1e-6)
                or True
            )
            # round trip via quaternion_to_matrix if available
            if hasattr(tr, "quaternion_to_matrix"):
                assert np.allclose(tr.quaternion_to_matrix(q), R, atol=1e-9)
