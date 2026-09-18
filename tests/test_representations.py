"""Tests for src/representations.py — VoxelGrid, Octree, comparison stats."""

import numpy as np
import pytest

from src.representations import Octree, VoxelGrid, compare_representations

BOUNDS = np.array([[-2.0, 2.0], [-2.0, 2.0], [-1.0, 3.0]])
RES = 0.25


# ---------------------------------------------------------------------------
# VoxelGrid
# ---------------------------------------------------------------------------


class TestVoxelGrid:
    def test_dims_and_origin(self):
        vg = VoxelGrid(BOUNDS, RES)
        expected = np.ceil((BOUNDS[:, 1] - BOUNDS[:, 0]) / RES).astype(int)
        assert np.array_equal(vg._dims, expected)
        assert np.allclose(vg._origin, BOUNDS[:, 0])

    def test_insert_and_query_roundtrip(self, rng):
        vg = VoxelGrid(BOUNDS, RES)
        pts = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(200, 3))
        vg.insert(pts)
        for p in pts:
            assert vg.query(p) == 1.0
        # an empty voxel must read as free
        empty = BOUNDS[:, 0] + 1e-3  # corner voxel only occupied if inserted
        if np.linalg.norm(pts - empty, axis=1).min() > RES:
            assert vg.query(empty) == 0.0

    def test_insert_with_value_array(self):
        vg = VoxelGrid(BOUNDS, 1.0)
        pts = np.array([[0.0, 0.0, 0.0], [1.5, -1.5, 2.5]])
        vg.insert(pts, values=np.array([0.25, 0.75], dtype=np.float64))
        assert vg.query(pts[0]) == pytest.approx(0.25)
        assert vg.query(pts[1]) == pytest.approx(0.75)

    def test_query_is_voxel_quantised(self):
        """Two points inside the same voxel give identical readings."""
        vg = VoxelGrid(BOUNDS, RES)
        vg.insert(np.array([[0.03, -1.97, 0.51]], dtype=np.float64))
        # same voxel: indices ((0.03+2)/0.25, (−1.97+2)/0.25, (0.51+1)/0.25) = (8, 0, 6)
        assert vg.query(np.array([0.10, -1.90, 0.70], dtype=np.float64)) == 1.0
        assert (
            vg.query(np.array([0.30, -1.97, 0.51], dtype=np.float64)) == 0.0
        )  # x falls in voxel 9

    def test_out_of_bounds_clipped_not_crashing(self):
        vg = VoxelGrid(BOUNDS, RES)
        far_pts = np.array([[100.0, -100.0, 0.0]], dtype=np.float64)
        vg.insert(far_pts)  # far outside
        assert vg.query(far_pts[0]) == 1.0  # stored at clipped corner
        # raw index (408, −392, 4) clips to the corner voxel (15, 0, 4)
        assert vg._grid[15, 0, 4] == 1.0

    def test_to_pointcloud_returns_voxel_centers(self):
        vg = VoxelGrid(BOUNDS, RES)
        p = np.array([[0.10, -0.10, 0.60]])
        vg.insert(p)
        pc = vg.to_pointcloud()
        assert pc.shape == (1, 3)
        # voxel centre must be within resolution/2 of the inserted point
        assert np.all(np.abs(pc[0] - p[0]) <= RES / 2)
        # and the centre lies exactly on the half-resolution lattice
        idx = np.floor((pc[0] - BOUNDS[:, 0]) / RES)
        assert np.allclose((pc[0] - BOUNDS[:, 0]) - (idx + 0.5) * RES, 0, atol=1e-9)

    def test_to_pointcloud_matches_inserted_count(self, rng):
        vg = VoxelGrid(BOUNDS, RES)
        pts = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(150, 3))
        vg.insert(pts)
        n_occupied_voxels = len(
            np.unique(np.floor((pts - BOUNDS[:, 0]) / RES).astype(int), axis=0)
        )
        assert vg.to_pointcloud().shape[0] == n_occupied_voxels

    def test_memory_bytes(self):
        vg = VoxelGrid(BOUNDS, RES)
        assert vg.memory_bytes == vg._grid.nbytes
        assert vg.memory_bytes == 4 * int(np.prod(vg._dims))  # float32


# ---------------------------------------------------------------------------
# Octree
# ---------------------------------------------------------------------------


class TestOctree:
    def test_insert_and_query(self, rng):
        ot = Octree(center=np.zeros(3), half_size=4.0, max_depth=8)
        pts = rng.uniform(-3.5, 3.5, size=(300, 3))
        for p in pts:
            ot.insert(p)
        assert ot.n_points == 300
        for p in pts:
            assert ot.query(p)
        assert not ot.query(np.array([50.0, 50.0, 50.0], dtype=np.float64))
        assert not ot.query(pts[0] + 0.5)  # distinct nearby point

    def test_get_all_points_preserves_multiset(self, rng):
        ot = Octree(center=np.zeros(3), half_size=4.0, max_depth=8)
        pts = rng.uniform(-3.5, 3.5, size=(250, 3))
        for p in pts:
            ot.insert(p)
        got = ot.get_all_points()
        assert got.shape == (250, 3)
        assert np.allclose(np.sort(got, axis=0), np.sort(pts, axis=0), atol=1e-12)

    def test_subdivision_happens_for_dense_cloud(self, rng):
        ot = Octree(center=np.zeros(3), half_size=4.0, max_depth=8, max_leaf_points=4)
        pts = rng.uniform(-3.5, 3.5, size=(200, 3))
        for p in pts:
            ot.insert(p)
        assert ot.n_nodes > 1
        # total stored points must survive subdivision
        assert ot.get_all_points().shape[0] == 200

    def test_duplicate_points_terminate(self):
        """Duplicates map to the same octant forever — recursion must stop."""
        ot = Octree(center=np.zeros(3), half_size=2.0, max_depth=4, max_leaf_points=2)
        p = np.array([0.5, 0.5, 0.5])
        for _ in range(20):
            ot.insert(p)
        assert ot.n_points == 20
        assert ot.query(p)
        assert ot.get_all_points().shape[0] == 20

    def test_empty_tree(self):
        ot = Octree(center=np.zeros(3), half_size=2.0)
        assert ot.n_points == 0
        assert ot.get_all_points().shape == (0, 3)
        assert not ot.query(np.array([0.0, 0.0, 0.0], dtype=np.float64))

    def test_child_octants_partition_space(self):
        """Each octant centre must lie inside the parent cube, at ±half/2."""
        center = np.array([1.0, -1.0, 0.5])
        half = 2.0
        for octant in range(8):
            c = Octree._child_center(center, half / 2, octant)
            for axis, bit in enumerate((1, 2, 4)):
                sign = 1.0 if octant & bit else -1.0
                assert c[axis] == pytest.approx(center[axis] + sign * half / 2)
            assert np.all(np.abs(c - center) <= half)

    def test_points_outside_root_still_stored(self):
        """Out-of-bounds points fall into the root's boundary octant."""
        ot = Octree(center=np.zeros(3), half_size=1.0, max_depth=2, max_leaf_points=1)
        outside = np.array([5.0, 5.0, 5.0])
        ot.insert(outside)
        assert ot.query(outside)
        assert ot.get_all_points().shape[0] == 1


# ---------------------------------------------------------------------------
# compare_representations
# ---------------------------------------------------------------------------


class TestCompareRepresentations:
    def test_returns_expected_keys(self, rng):
        pts = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(200, 3))
        stats = compare_representations(pts, BOUNDS, voxel_size=RES)
        assert set(stats) == {"voxel_grid", "octree"}
        assert stats["voxel_grid"]["n_occupied"] > 0
        assert stats["octree"]["n_points"] == 200
        assert stats["voxel_grid"]["memory_bytes"] > 0
        for key in ("insert_time_s", "query_time_s"):
            assert stats["voxel_grid"][key] >= 0.0
            assert stats["octree"][key] >= 0.0

    def test_voxel_memory_matches_grid(self, rng):
        pts = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(100, 3))
        stats = compare_representations(pts, BOUNDS, voxel_size=RES)
        vg = VoxelGrid(BOUNDS, RES)
        assert stats["voxel_grid"]["memory_bytes"] == vg.memory_bytes

    def test_empty_cloud_does_not_crash(self):
        """Regression: division by zero on the query-timing probe."""
        pts = np.zeros((0, 3))
        stats = compare_representations(pts, BOUNDS, voxel_size=RES)
        assert stats["voxel_grid"]["n_occupied"] == 0
        assert stats["octree"]["n_points"] == 0
