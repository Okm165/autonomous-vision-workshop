"""Camera model tests: projection, distortion inversion, PnP, decomposition."""

import cv2
import numpy as np

from src import camera as cam

K = np.array([[800.0, 0, 320.0], [0, 810.0, 240.0], [0, 0, 1.0]])
DIST = np.array([-0.28, 0.12, 0.001, -0.0007, 0.0])


def _camera(dist=None):
    return cam.CameraIntrinsics(800.0, 810.0, 320.0, 240.0, dist_coeffs=dist)


class TestProjection:
    def test_pinhole_formula(self, points_3d):
        C = _camera()
        uv = C.project(points_3d)
        expected = np.column_stack(
            [
                800 * points_3d[:, 0] / points_3d[:, 2] + 320,
                810 * points_3d[:, 1] / points_3d[:, 2] + 240,
            ]
        )
        assert np.allclose(uv, expected, atol=1e-12)

    def test_project_matches_cv2_with_distortion(self, points_3d):
        C = _camera(DIST)
        uv = C.project(points_3d)
        objp = points_3d.reshape(-1, 1, 3).astype(np.float64)
        cv_uv, _ = cv2.projectPoints(objp, np.zeros(3), np.zeros(3), K, DIST)
        assert np.allclose(uv, cv_uv.reshape(-1, 2), atol=1e-9)

    def test_behind_camera_is_nan(self):
        C = _camera()
        assert np.isnan(C.project(np.array([[1.0, 1.0, -1.0]]))).all()


class TestUndistortion:
    def test_inverts_projection_across_domain(self, points_3d):
        """undistort_points(project(P)) == pinhole(P) for the full frustum."""
        C = _camera(DIST)
        C0 = _camera()
        uv = C0.project(points_3d)
        uv_d = C.project(points_3d)
        uv_u = C.undistort_points(uv_d)
        assert np.abs(uv_u - uv).max() < 1e-6

    def test_beats_cv2_inside_fov(self):
        rng = np.random.default_rng(3)
        uv = np.column_stack([rng.uniform(0, 640, 500), rng.uniform(0, 480, 500)])
        C = _camera(DIST)
        ours = C.undistort_points(uv)
        cv_out = cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), K, DIST)
        cv_pix = np.column_stack(
            [cv_out[:, 0, 0] * 800 + 320, cv_out[:, 0, 1] * 810 + 240]
        )

        def fwd_residual(pix):
            xn = (pix[:, 0] - 320) / 800
            yn = (pix[:, 1] - 240) / 810
            xd, yd = cam._distort_forward(xn, yn, *DIST)
            return np.maximum(
                np.abs(xd * 800 + 320 - uv[:, 0]), np.abs(yd * 810 + 240 - uv[:, 1])
            )

        # both invert the map, but ours must be at least as accurate as cv2
        assert fwd_residual(ours).max() <= fwd_residual(cv_pix).max() + 1e-9
        assert fwd_residual(ours).max() < 1e-8

    def test_no_distortion_is_identity(self, points_3d):
        C = _camera()
        uv = C.project(points_3d)
        assert np.array_equal(C.undistort_points(uv), uv)


class TestDecomposeP:
    def test_roundtrip(self):
        from src import transforms as tr

        R = tr.so3_exp(np.array([0.1, -0.2, 0.3]))
        t = np.array([0.5, -0.3, 2.0])
        P = cam.compute_P(K, R, t)
        Kd, Rd, td = cam.decompose_P(P)
        assert np.allclose(Kd, K, atol=1e-8)
        assert np.allclose(Rd, R, atol=1e-8)
        assert np.allclose(np.ravel(td), t, atol=1e-8)
        assert np.allclose(cam.compute_P(Kd, Rd, td), P, atol=1e-8)

    def test_roundtrip_skewed_K(self):
        from src import transforms as tr

        Ks = np.array([[800.0, 2.0, 320.0], [0, 810.0, 240.0], [0, 0, 1.0]])
        R = tr.so3_exp(np.array([0.1, -0.2, 0.3]))
        t = np.array([0.5, -0.3, 2.0])
        P = cam.compute_P(Ks, R, t)
        Kd, Rd, _td = cam.decompose_P(P)
        assert np.allclose(Kd, Ks, atol=1e-8)
        assert np.allclose(Rd, R, atol=1e-8)


class TestFromFov:
    def test_90deg(self):
        C = cam.CameraIntrinsics.from_fov(90.0, (640, 480))
        assert abs(C.fx - 320.0) < 1e-9
        assert abs(C.cx - 320.0) < 1e-9
