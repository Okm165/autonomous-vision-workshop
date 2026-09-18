"""Tests for src/pointcloud.py — back-projection, transforms, ICP, normals.

ICP is validated with invariance-style tests: a perfect init must recover the
identity, and a synthetic rigid transform must be recovered from data.
"""

import numpy as np
import pytest
from scipy.spatial import KDTree

from src.pointcloud import (
    depth_to_pointcloud,
    estimate_normals,
    icp_align,
    merge_pointclouds,
    remove_outliers,
    transform_points,
    voxel_downsample,
)

K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
H, W = 48, 64


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TestDepthToPointCloud:
    def test_constant_depth_plane(self):
        depth = np.full((H, W), 2.0)
        pc = depth_to_pointcloud(depth, K)
        assert pc.shape == (H * W, 3)
        assert pc.dtype == np.float64
        # pixel (u=10, v=5): X = (u-cx)*d/fx, Y = (v-cy)*d/fy, Z = d
        p = pc[5 * W + 10]  # row-major order of the valid mask
        assert p[0] == pytest.approx((10 - 32) * 2.0 / 100.0)
        assert p[1] == pytest.approx((5 - 24) * 2.0 / 100.0)
        assert p[2] == pytest.approx(2.0)

    def test_invalid_depth_excluded(self):
        depth = np.full((H, W), 3.0)
        depth[0, 0] = 0.0
        depth[7, 13] = -1.0  # non-positive treated as invalid
        pc = depth_to_pointcloud(depth, K)
        assert pc.shape == (H * W - 2, 3)
        assert np.all(pc[:, 2] > 0)

    def test_all_invalid_depth(self):
        pc = depth_to_pointcloud(np.zeros((4, 4)), K)
        assert pc.shape == (0, 3)

    def test_with_color(self):
        rng = np.random.default_rng(0)
        depth = np.full((H, W), 1.0)
        color = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        pc = depth_to_pointcloud(depth, K, color=color)
        assert pc.shape == (H * W, 6)
        assert np.allclose(pc[:, 3:], color.reshape(-1, 3))
        assert np.all((pc[:, 3:] >= 0) & (pc[:, 3:] <= 255))


class TestTransformPoints:
    def test_identity(self, rng):
        pts = rng.normal(size=(20, 3))
        assert np.allclose(transform_points(pts, np.eye(4)), pts)

    def test_known_rigid_transform(self):
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        T = np.eye(4)
        T[:3, :3] = _rot_z(np.pi / 2)
        T[:3, 3] = [1.0, 2.0, 3.0]
        out = transform_points(pts, T)
        # Rz(90): (1,0,0) -> (0,1,0); (0,1,0) -> (-1,0,0); then + t
        assert np.allclose(out[0], [1.0, 3.0, 3.0])
        assert np.allclose(out[1], [0.0, 2.0, 3.0])

    def test_translation_only(self, rng):
        pts = rng.normal(size=(10, 3))
        T = np.eye(4)
        T[:3, 3] = [0.5, -0.5, 2.0]
        assert np.allclose(transform_points(pts, T), pts + T[:3, 3])

    def test_colors_preserved(self, rng):
        pts6 = np.column_stack([rng.normal(size=(10, 3)), rng.normal(size=(10, 3))])
        T = np.eye(4)
        T[:3, 3] = [1, 1, 1]
        out = transform_points(pts6, T)
        assert out.shape == (10, 6)
        assert np.allclose(out[:, 3:], pts6[:, 3:])  # colours untouched
        assert np.allclose(out[:, :3], pts6[:, :3] + 1.0)

    def test_se3_composition_property(self, rng):
        """transform_points(p, T2 @ T1) == transform_points(transform_points(p, T1), T2)."""
        from src.transforms import se3_exp

        pts = rng.normal(size=(30, 3))
        T1 = se3_exp(rng.normal(scale=0.5, size=6))
        T2 = se3_exp(rng.normal(scale=0.5, size=6))
        lhs = transform_points(pts, T2 @ T1)
        rhs = transform_points(transform_points(pts, T1), T2)
        assert np.allclose(lhs, rhs, atol=1e-10)


class TestMergePointClouds:
    def test_merge_two_clouds(self, rng):
        c1 = rng.normal(size=(5, 3))
        c2 = rng.normal(size=(7, 3))
        T_id = np.eye(4)
        T_shift = np.eye(4)
        T_shift[:3, 3] = [10.0, 0.0, 0.0]
        merged = merge_pointclouds([c1, c2], [T_id, T_shift])
        assert merged.shape == (12, 3)
        assert np.allclose(merged[:5], c1)
        expected_second = c2 + np.array([10.0, 0.0, 0.0])
        assert np.allclose(merged[5:], expected_second)

    def test_merge_with_colors(self, rng):
        c = np.column_stack([rng.normal(size=(4, 3)), np.ones((4, 3))])
        merged = merge_pointclouds([c], [np.eye(4)])
        assert merged.shape == (4, 6)


class TestVoxelDownsample:
    def test_points_in_same_voxel_merge_to_centroid(self):
        pts = np.array(
            [
                [0.01, 0.01, 0.01],
                [0.02, 0.02, 0.02],
                [0.09, 0.09, 0.09],  # voxel (0,0,0)
                [0.95, 0.95, 0.95],
            ]  # voxel (9,9,9)
        )
        out = voxel_downsample(pts, voxel_size=0.1)
        assert out.shape == (2, 3)
        assert np.allclose(out[0], [0.04, 0.04, 0.04])
        assert np.allclose(out[1], [0.95, 0.95, 0.95])

    def test_output_is_voxel_centroid(self, rng):
        pts = rng.uniform(-2, 2, size=(300, 3))
        out = voxel_downsample(pts, voxel_size=0.25)
        assert out.shape[0] <= pts.shape[0]
        # every output point must be the exact mean of the inputs in its voxel
        key_in = np.floor(pts / 0.25).astype(np.int64)
        for p in out:
            k = np.floor(p / 0.25).astype(np.int64)
            group = pts[np.all(key_in == k, axis=1)]
            assert len(group) > 0
            assert np.allclose(p, group.mean(axis=0), atol=1e-12)

    def test_no_collisions_in_hash_keys(self, rng):
        """Distinct voxels must never collapse into one centroid."""
        pts = rng.uniform(-5, 5, size=(500, 3))
        vs = 0.3
        n_expected = len(np.unique(np.floor(pts / vs).astype(np.int64), axis=0))
        out = voxel_downsample(pts, vs)
        assert out.shape[0] == n_expected

    def test_preserves_extra_columns_as_mean(self):
        pts = np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.05, 0.0, 0.0, 3.0]]  # xyz + intensity
        )
        out = voxel_downsample(pts, 0.1)
        assert out.shape == (1, 4)
        assert out[0, 3] == pytest.approx(2.0)

    def test_large_voxel_single_centroid(self, rng):
        pts = rng.uniform(0, 1, size=(50, 3))
        out = voxel_downsample(pts, voxel_size=10.0)
        assert out.shape == (1, 3)
        assert np.allclose(out[0], pts.mean(axis=0))


class TestEstimateNormals:
    def test_plane_normals_perpendicular(self, rng):
        """Points on a z=const plane -> normals parallel to z-axis."""
        n_pts = 200
        pts = np.column_stack(
            [
                rng.uniform(-1, 1, n_pts),
                rng.uniform(-1, 1, n_pts),
                np.full(n_pts, 3.0) + rng.normal(scale=1e-3, size=n_pts),
            ]
        )
        normals = estimate_normals(pts, k=10)
        assert normals.shape == (n_pts, 3)
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)
        dots = normals @ np.array([0.0, 0.0, 1.0])
        assert np.all(np.abs(dots) > 0.99)  # direction (sign) is a convention

    def test_sphere_normals_radial(self, rng):
        """Points on a unit sphere -> normals aligned with the radius."""
        v = rng.normal(size=(400, 3))
        pts = v / np.linalg.norm(v, axis=1, keepdims=True)
        normals = estimate_normals(pts, k=12)
        dots = np.sum(normals * pts, axis=1)
        assert np.mean(np.abs(dots) > 0.9) > 0.9

    def test_unit_norm(self, rng):
        pts = rng.normal(size=(80, 3)) * 0.1
        normals = estimate_normals(pts, k=8)
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)


def _random_se3(rng, max_angle=0.4, max_t=0.5):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.1, max_angle)
    from src.transforms import so3_exp

    T = np.eye(4)
    T[:3, :3] = so3_exp(axis * angle)
    T[:3, 3] = rng.uniform(-max_t, max_t, size=3)
    return T


class TestICP:
    def test_perfect_init_recovers_identity(self):
        rng = np.random.default_rng(4)
        pts = rng.uniform(0, 2, size=(150, 3))
        T, dists, iters = icp_align(pts, pts.copy())
        assert np.allclose(T, np.eye(4), atol=1e-9)
        assert np.max(dists) < 1e-9
        assert 1 <= iters <= 50

    def test_recovers_synthetic_rigid_transform(self):
        rng = np.random.default_rng(7)
        src = rng.uniform(0, 2, size=(300, 3))
        T_true = _random_se3(rng)
        tgt = transform_points(src, T_true)

        T_est, _dists, iters = icp_align(src, tgt, max_iterations=100, tolerance=1e-12)

        # final alignment quality: NN distance of aligned source to target
        aligned = transform_points(src, T_est)
        final_d = KDTree(tgt).query(aligned)[0]
        assert final_d.mean() < 1e-4
        assert iters <= 100

        # recovered transform close to ground truth
        R_err = T_est[:3, :3] @ T_true[:3, :3].T
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        assert angle_err < 1e-2
        assert np.allclose(T_est[:3, 3], T_true[:3, 3], atol=1e-2)

    def test_returns_valid_se3(self, rng):
        src = rng.uniform(0, 1, size=(100, 3))
        T_true = _random_se3(rng)
        tgt = transform_points(src, T_true)
        T, dists, _ = icp_align(src, tgt)
        assert T.shape == (4, 4)
        assert np.allclose(T[3], [0, 0, 0, 1])
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        assert dists.shape == (100,)
        assert np.all(dists >= 0)

    def test_translation_only_recovery(self):
        rng = np.random.default_rng(11)
        src = rng.uniform(0, 2, size=(200, 3))
        T_true = np.eye(4)
        T_true[:3, 3] = [0.5, -0.3, 0.2]
        tgt = transform_points(src, T_true)
        T, dists, _ = icp_align(src, tgt, max_iterations=100, tolerance=1e-12)
        assert np.allclose(T, T_true, atol=1e-3)
        assert dists.mean() < 1e-6


class TestRemoveOutliers:
    def test_isolated_outliers_removed(self, rng):
        inliers = rng.normal(scale=0.05, size=(200, 3))
        outliers = rng.uniform(8.0, 10.0, size=(10, 3))  # far away cluster
        pts = np.vstack([inliers, outliers])
        filtered, mask = remove_outliers(pts, k=10, std_ratio=2.0)
        assert mask.shape == (210,)
        assert mask.sum() == len(filtered)
        assert mask[-10:].sum() == 0  # all far outliers removed
        assert mask[:200].sum() >= 190  # nearly all inliers kept
        assert filtered.shape[1] == 3

    def test_mask_matches_filtered(self, rng):
        pts = rng.uniform(0, 1, size=(120, 4))  # extra column preserved
        filtered, mask = remove_outliers(pts, k=8, std_ratio=1.0)
        assert np.allclose(filtered, pts[mask])
        assert filtered.shape[1] == 4

    def test_high_threshold_keeps_everything(self, rng):
        """A huge std_ratio must not remove any point."""
        pts = rng.uniform(0, 1, size=(100, 3))
        filtered, mask = remove_outliers(pts, k=10, std_ratio=1e6)
        assert mask.all()
        assert len(filtered) == 100


class TestPoissonReconstruct:
    def test_smoke(self):
        pytest.importorskip("open3d")
        rng = np.random.default_rng(3)
        pts = np.column_stack(
            [rng.uniform(-1, 1, 150), rng.uniform(-1, 1, 150), np.zeros(150)]
        )
        normals = np.tile([0.0, 0.0, 1.0], (150, 1))
        mesh = pytest.importorskip("src.pointcloud").poisson_reconstruct(
            pts, normals, depth=4
        )
        assert len(mesh.vertices) > 0
        assert len(mesh.triangles) > 0
