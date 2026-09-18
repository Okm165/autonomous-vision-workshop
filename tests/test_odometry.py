"""Tests for src/odometry.py — scale recovery, ATE, and VO pipelines.

The synthetic scenes are chosen to sit inside each algorithm's validated
operating envelope (verified during test design):

* MonocularVO — lateral motion between two textured planes at different
  depths (forward motion on planar texture is a degenerate E-estimation
  case and is deliberately avoided).
* StereoVO — SGBM's dynamic program operates per image row, so the scene
  depth must vary *across* rows, not along columns; the surface is a wavy
  height field rendered with blob texture (raw iid noise defeats both SGBM
  and ORB).
"""

import cv2
import numpy as np
import pytest

from src.odometry import (
    MonocularVO,
    StereoVO,
    compute_relative_pose,
    compute_trajectory_error,
    scale_from_ground_truth,
)
from src.transforms import se3_from_Rt

K_VO = np.array([[100.0, 0.0, 96.0], [0.0, 100.0, 72.0], [0.0, 0.0, 1.0]])
SIZE = (144, 192)  # (H, W) — SGBM hard-requires width > numDisparities + block


# ---------------------------------------------------------------------------
# Scale recovery and trajectory error
# ---------------------------------------------------------------------------


class TestScaleFromGroundTruth:
    def test_recovers_scale(self):
        est = np.array([3.0, 4.0, 0.0])
        gt = 2.5 * est
        assert scale_from_ground_truth(est, gt) == pytest.approx(2.5)

    def test_zero_estimate_returns_one(self):
        assert scale_from_ground_truth(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 1.0


class TestTrajectoryError:
    def test_identical_trajectories_zero_error(self):
        poses = [np.eye(4), se3_from_Rt(np.eye(3), np.array([1.0, 1.0, 0.0]))]
        assert compute_trajectory_error(poses, poses) < 1e-12

    def test_matches_ate_definition(self):
        from src.eval import compute_ate

        rng = np.random.default_rng(5)
        gt = [se3_from_Rt(np.eye(3), p) for p in rng.uniform(0, 5, (10, 3))]
        est = [
            se3_from_Rt(np.eye(3), p + rng.normal(0, 0.05, 3))
            for p in rng.uniform(0, 5, (10, 3))
        ]
        ate_rmse = compute_ate(est, gt)[0]
        assert compute_trajectory_error(est, gt) == pytest.approx(ate_rmse, rel=1e-10)


# ---------------------------------------------------------------------------
# Relative pose from matches
# ---------------------------------------------------------------------------


class TestComputeRelativePose:
    def test_unit_translation_and_valid_rotation(self):
        rng = np.random.default_rng(11)
        pts3d = np.column_stack(
            [
                rng.uniform(-1.5, 1.5, 120),
                rng.uniform(-1.5, 1.5, 120),
                rng.uniform(2.0, 6.0, 120),
            ]
        )
        K = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]])

        angle = 0.15
        c, s = np.cos(angle), np.sin(angle)
        R_true = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        t_true = np.array([1.0, 0.0, 0.0])
        t_true /= np.linalg.norm(t_true)

        def project(pts, R, t):
            cam = pts @ R.T + t
            uv = (K @ cam.T).T
            return uv[:, :2] / uv[:, 2:3]

        p1 = project(pts3d, np.eye(3), np.zeros(3))
        p2 = project(pts3d, R_true, t_true)

        kp1 = [cv2.KeyPoint(float(x), float(y), 3) for x, y in p1]
        kp2 = [cv2.KeyPoint(float(x), float(y), 3) for x, y in p2]
        matches = [cv2.DMatch(i, i, 0.0) for i in range(len(pts3d))]

        R, t, mask = compute_relative_pose(kp1, kp2, matches, K)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(t) == pytest.approx(1.0, abs=1e-10)
        assert mask is not None


# ---------------------------------------------------------------------------
# Monocular VO (unit-scale)
# ---------------------------------------------------------------------------


def _render_two_plane_scene(tex, center, tex_extent=(4.0, 3.0)):
    """Render a two-depth-plane noise texture: world x < 0 at Z = 2,
    x >= 0 at Z = 3.5.  Raw noise gives ORB plenty of corners (descriptor
    matching involves no Taylor expansion, so smoothness is not needed)."""
    H, W = SIZE
    fx, fy, cx, cy = K_VO[0, 0], K_VO[1, 1], K_VO[0, 2], K_VO[1, 2]
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    dx = (us - cx) / fx
    dy = (vs - cy) / fy

    def sample(z_plane):
        t = z_plane - center[2]
        x = center[0] + t * dx
        y = center[1] + t * dy
        tx = (x + tex_extent[0]) / (2 * tex_extent[0]) * (tex.shape[1] - 1)
        ty = (y + tex_extent[1]) / (2 * tex_extent[1]) * (tex.shape[0] - 1)
        return cv2.remap(
            tex,
            tx.astype(np.float32),
            ty.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    boundary = center[0] + (2.0 - center[2]) * dx
    img = np.where(boundary < 0, sample(2.0), sample(3.5))
    return img.astype(np.uint8)


class TestMonocularVO:
    def test_lateral_motion_direction(self):
        rng = np.random.default_rng(21)
        tex = rng.integers(0, 256, (301, 401), dtype=np.uint8)
        vo = MonocularVO(K_VO)
        frame1 = _render_two_plane_scene(tex, (0.0, 0.0, 0.0))
        frame2 = _render_two_plane_scene(tex, (0.5, 0.0, 0.0))

        pose2 = vo.process_frame(frame1)
        assert np.allclose(pose2[:3, 3], 0.0)  # first frame anchors the origin
        pose3 = vo.process_frame(frame2)

        t = pose3[:3, 3]
        assert np.linalg.norm(t) == pytest.approx(1.0, abs=0.1)  # unit-norm gauge
        direction = t / np.linalg.norm(t)
        assert float(direction @ np.array([1.0, 0.0, 0.0])) > 0.9


# ---------------------------------------------------------------------------
# Stereo VO (metric scale)
# ---------------------------------------------------------------------------


def _make_blob_texture(rng, tex_size=(301, 401)):
    """Blob texture: high-contrast smooth-ish content SGBM can match."""
    blob = np.zeros(tex_size, dtype=np.float64)
    for _ in range(150):
        cy, cx = rng.integers(0, tex_size[0]), rng.integers(0, tex_size[1])
        r = int(rng.integers(4, 14))
        val = float(rng.integers(60, 256))
        cv2.circle(blob, (int(cx), int(cy)), r, val, -1)
    return blob.astype(np.uint8)


def _render_wavy_surface(tex, center):
    """Render a wavy surface z = 3 + 0.5y + 0.3 sin(0.8x) with *tex*.

    Depth varies across rows (SGBM's per-row DP can lock on) and the
    sin term makes the point cloud non-planar (PnP needs ≥3 dimensions
    of structure; coplanar points silently fail).
    """
    H, W = SIZE
    fx, fy, cx, cy = K_VO[0, 0], K_VO[1, 1], K_VO[0, 2], K_VO[1, 2]
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    dx = (us - cx) / fx
    dy = (vs - cy) / fy

    def surface(x, y):
        return 3.0 + 0.5 * y + 0.3 * np.sin(0.8 * x)

    t = np.full((H, W), 3.0)
    for _ in range(20):  # fixed-point solve of the ray–surface intersection
        t = surface(center[0] + t * dx, center[1] + t * dy) - center[2]
    x = center[0] + t * dx
    y = center[1] + t * dy
    tx = ((x + 4.0) / 8.0 * (tex.shape[1] - 1)).astype(np.float32)
    ty = ((y + 3.0) / 6.0 * (tex.shape[0] - 1)).astype(np.float32)
    img = cv2.remap(tex, tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img


class TestStereoVO:
    def test_metric_lateral_motion(self):
        rng = np.random.default_rng(7)
        tex = _make_blob_texture(rng)  # one shared texture for all views
        vo = StereoVO(K_VO, baseline=0.5)
        left1 = _render_wavy_surface(tex, (0.0, 0.0, 0.0))
        right1 = _render_wavy_surface(tex, (0.5, 0.0, 0.0))
        left2 = _render_wavy_surface(tex, (0.5, 0.0, 0.0))
        right2 = _render_wavy_surface(tex, (1.0, 0.0, 0.0))

        vo.process_frame(left1, right1)
        pose2 = vo.process_frame(left2, right2)

        t = pose2[:3, 3]
        # Metric scale: 0.5 m lateral motion (measured 0.46 on this scene).
        assert 0.35 < np.linalg.norm(t) < 0.65
        direction = t / np.linalg.norm(t)
        assert float(direction @ np.array([1.0, 0.0, 0.0])) > 0.95

        traj = vo.get_trajectory()
        assert traj.shape == (2, 3)
        np.testing.assert_allclose(traj[0], 0.0)
