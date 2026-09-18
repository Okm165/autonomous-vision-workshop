"""Tests for src/pipeline.py — LocalMapper orchestration.

These are integration tests: a synthetic scene (textured frame + flat depth
map of a wall at 2 m) is pushed through feature matching, pose estimation,
point cloud generation, TSDF fusion and occupancy mapping.
"""

import sys
import types

import numpy as np
import pytest

from src.pipeline import LocalMapper
from src.tsdf import TSDFVolume

K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
BOUNDS = np.array([[-2.0, 2.0], [-2.0, 2.0], [0.5, 3.5]])
VOXEL = 0.25
H = W = 64
DEPTH = 2.0  # fronto-parallel wall at 2 m


@pytest.fixture(scope="module")
def mapper():
    """A LocalMapper that has consumed two identical textured frames."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), DEPTH)
    m = LocalMapper(K, map_bounds=BOUNDS, voxel_size=VOXEL)
    m.process_frame(img, depth=depth)
    m.process_frame(img, depth=depth)
    return m


# ---------------------------------------------------------------------------
# Pose / VO integration
# ---------------------------------------------------------------------------


class TestPoses:
    def test_first_frame_is_identity(self, mapper):
        poses = mapper.get_poses()
        assert len(poses) == 2
        assert np.allclose(poses[0], np.eye(4))

    def test_second_frame_finite_and_valid_se3(self, mapper):
        T = mapper.get_poses()[1]
        assert T.shape == (4, 4)
        assert np.all(np.isfinite(T))
        assert np.allclose(T[3], [0, 0, 0, 1])
        R = T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)

    def test_trajectory_shape(self, mapper):
        traj = mapper.get_trajectory()
        assert traj.ndim == 2
        assert traj.shape[1] == 3
        assert traj.shape[0] >= 1  # at least the initial origin sample
        assert np.allclose(traj[0], 0.0)


# ---------------------------------------------------------------------------
# Point cloud / TSDF / occupancy state
# ---------------------------------------------------------------------------


class TestMapState:
    def test_merged_pointcloud_shape_and_colors(self, mapper):
        pc = mapper.get_pointcloud()
        assert pc.shape == (2 * H * W, 6)  # all depths valid → all pixels kept
        assert pc[:, 2].min() > 0
        assert np.all((pc[:, 3:] >= 0) & (pc[:, 3:] <= 255))

    def test_pointcloud_backprojection_correct(self, mapper):
        """Pixel (u, v) must map to ((u−cx)d/fx, (v−cy)d/fy, d) in cam frame;
        with identity pose the world frame coincides with the camera frame."""
        pc = mapper.get_pointcloud()
        u, v = 10, 5
        p = pc[v * W + u]  # row-major traversal of the depth image
        assert p[0] == pytest.approx((u - 32.0) * DEPTH / 100.0)
        assert p[1] == pytest.approx((v - 32.0) * DEPTH / 100.0)
        assert p[2] == pytest.approx(DEPTH)

    def test_mesh_lies_on_the_wall_plane(self, mapper):
        verts, faces, norms, colors = mapper.get_mesh()
        assert len(verts) > 0
        assert len(faces) > 0
        assert verts.shape[1] == 3
        assert norms.shape == verts.shape
        assert colors.shape == verts.shape
        # the observed surface is exactly the z = 2 m plane
        assert np.allclose(verts[:, 2], DEPTH, atol=1e-6)

    def test_mesh_faces_are_valid_indices(self, mapper):
        _, faces, _, _ = mapper.get_mesh()
        assert faces.min() >= 0

    def test_occupancy_log_odds_updated(self, mapper):
        occ = mapper.get_occupancy_grid()
        # rays traverse free space before the wall → negative log-odds,
        # the wall surface voxel → positive log-odds
        assert (occ.grid > 0).sum() > 0
        assert (occ.grid < 0).sum() > 0
        # clamping bounds are respected
        assert occ.grid.max() <= occ.log_odds_max + 1e-9
        assert occ.grid.min() >= -occ.log_odds_max - 1e-9

    def test_get_map_returns_tsdf(self, mapper):
        assert isinstance(mapper.get_map(), TSDFVolume)

    def test_occupancy_grid_bounds_match(self, mapper):
        occ = mapper.get_occupancy_grid()
        assert np.allclose(occ.bounds, BOUNDS)


# ---------------------------------------------------------------------------
# Fresh mapper (empty state) and construction options
# ---------------------------------------------------------------------------


class TestFreshMapper:
    def test_empty_accessors(self):
        m = LocalMapper(K, map_bounds=BOUNDS, voxel_size=VOXEL)
        assert m.get_pointcloud().shape == (0, 3)
        assert m.get_poses() == []
        assert m.get_trajectory().shape == (1, 3)
        verts, faces, _norms, _colors = m.get_mesh()
        assert len(verts) == 0
        assert len(faces) == 0

    def test_semantic_mode_disabled_by_default(self):
        m = LocalMapper(K, map_bounds=BOUNDS, voxel_size=VOXEL)
        assert m.get_semantic_mesh() is None

    def test_sift_detector_accepted(self):
        m = LocalMapper(K, map_bounds=BOUNDS, voxel_size=VOXEL, detector="sift")
        assert m._vo.detector_name == "sift"

    def test_unknown_detector_rejected(self):
        with pytest.raises(ValueError, match="detector"):
            LocalMapper(K, map_bounds=BOUNDS, voxel_size=VOXEL, detector="fast")


# ---------------------------------------------------------------------------
# Semantic mode
# ---------------------------------------------------------------------------


class TestSemanticMode:
    def test_semantic_mesh_after_frames(self):

        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        depth = np.full((H, W), DEPTH)
        m = LocalMapper(K, map_bounds=BOUNDS, voxel_size=VOXEL, use_semantic=True)
        m.process_frame(img, depth=depth)
        m.process_frame(img, depth=depth)
        result = m.get_semantic_mesh()
        assert result is not None
        verts, _faces, _norms, labels = result
        assert len(verts) > 0
        assert len(verts) == len(labels)
        assert np.all(verts[:, 2] == pytest.approx(DEPTH, abs=1e-6))


# ---------------------------------------------------------------------------
# depth=None path (neural estimation unavailable → frame still processed)
# ---------------------------------------------------------------------------


class TestDepthFallback:
    def test_missing_depth_module_skips_integration(self, monkeypatch):
        """If neural depth is unavailable, the pose is still returned and no
        points are fused (ImportError swallowed by _estimate_depth)."""
        stub = types.ModuleType("src.depth")  # no DepthEstimator attribute
        monkeypatch.setitem(sys.modules, "src.depth", stub)

        rng = np.random.default_rng(2)
        img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        m = LocalMapper(K, map_bounds=BOUNDS, voxel_size=VOXEL)
        T = m.process_frame(img, depth=None)

        assert T.shape == (4, 4)
        assert np.allclose(T, np.eye(4))
        assert len(m.get_poses()) == 1
        assert m.get_pointcloud().shape == (0, 3)
