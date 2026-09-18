"""Tests for src/occupancy.py — log-odds occupancy grid mapping."""

import numpy as np
import pytest

from src.occupancy import OccupancyGrid

BOUNDS = np.array([[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]], dtype=np.float64)
RES = 0.5
CAM_POS = np.array([0.25, 0.25, 0.25], dtype=np.float64)


def _single_ray_grid() -> OccupancyGrid:
    """Grid fed one depth pixel whose ray runs along +z through voxel (0, 0, *).

    Camera at world (0.25, 0.25, 0.25) looking down +z; the surface point
    lands at depth 3.5 -> world z = 3.75 -> voxel (0, 0, 7).
    """
    grid = OccupancyGrid(BOUNDS, resolution=RES)
    depth = np.array([[3.5]])
    K = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, 3] = [0.25, 0.25, 0.25]
    grid.update(depth, K, T)
    return grid


class TestConstructor:
    def test_valid_bounds(self):
        grid = OccupancyGrid(BOUNDS, resolution=RES)
        assert grid.grid_size.tolist() == [8, 8, 8]
        assert grid.grid.shape == (8, 8, 8)
        np.testing.assert_array_equal(grid.grid, 0.0)

    def test_transposed_bounds_accepted(self):
        grid = OccupancyGrid(
            np.array([[0.0, 4.0]] * 3, dtype=np.float64), resolution=RES
        )
        assert grid.bounds.shape == (3, 2)

        swapped = OccupancyGrid(np.array(BOUNDS).T, resolution=RES)
        assert swapped.grid.shape == (8, 8, 8)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match=r"bounds must have shape \(3, 2\)"):
            OccupancyGrid(np.zeros((4, 2)))

    def test_empty_bounds_raise(self):
        with pytest.raises(ValueError, match="bounds max must exceed min"):
            OccupancyGrid(
                np.array([[1.0, 1.0], [0.0, 4.0], [0.0, 4.0]], dtype=np.float64)
            )

    def test_grid_size_ceil(self):
        grid = OccupancyGrid(
            np.array([[0.0, 1.1], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
            resolution=0.5,
        )
        assert grid.grid_size.tolist() == [3, 2, 2]


class TestSingleRayUpdate:
    def test_log_odds_values(self):
        grid = _single_ray_grid()
        # Free voxels along the ray, occupied hit voxel.
        np.testing.assert_allclose(grid.grid[0, 0, :7], -0.4)
        assert grid.grid[0, 0, 7] == pytest.approx(0.85)
        # Untouched voxels stay at zero.
        assert grid.grid[1, 1, 1] == 0.0

    def test_probability_conversion(self):
        grid = _single_ray_grid()
        prob = grid.to_probability()
        np.testing.assert_allclose(prob[0, 0, :7], 1.0 / (1.0 + np.exp(0.4)))
        assert prob[0, 0, 7] == pytest.approx(1.0 / (1.0 + np.exp(-0.85)))

    def test_accumulation_and_clamping(self):
        grid = OccupancyGrid(BOUNDS, resolution=RES)
        depth = np.array([[3.5]])
        K = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, 3] = [0.25, 0.25, 0.25]
        for _ in range(5):
            grid.update(depth, K, T)

        np.testing.assert_allclose(grid.grid[0, 0, :7], -2.0)  # 5 * -0.4
        assert grid.grid[0, 0, 7] == pytest.approx(3.5)  # 4.25 clipped to max

    def test_invalid_depth_ignored(self):
        grid = OccupancyGrid(BOUNDS, resolution=RES)
        K = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        grid.update(np.array([[0.0]]), K, T)
        grid.update(np.array([[-1.0]]), K, T)
        np.testing.assert_array_equal(grid.grid, 0.0)

    def test_out_of_bounds_endpoint_ignored(self):
        grid = OccupancyGrid(BOUNDS, resolution=RES)
        # Depth 10 m -> surface far outside the grid -> whole ray skipped.
        K = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, 3] = [0.25, 0.25, 0.25]
        grid.update(np.array([[10.0]]), K, T)
        np.testing.assert_array_equal(grid.grid, 0.0)


class TestQueries:
    def test_occupied_points(self):
        grid = _single_ray_grid()
        pts = grid.get_occupied_points(threshold=0.6)
        assert pts.shape == (1, 3)
        # Voxel (0, 0, 7) centre.
        np.testing.assert_allclose(pts[0], [0.25, 0.25, 3.75])

    def test_free_points(self):
        grid = _single_ray_grid()
        pts = grid.get_free_points(threshold=0.45)
        assert pts.shape == (7, 3)
        expected_z = 0.25 + 0.5 * np.arange(7)
        np.testing.assert_allclose(pts[:, 2], expected_z)
        np.testing.assert_allclose(pts[:, :2], 0.25)

    def test_is_occupied(self):
        grid = _single_ray_grid()
        assert grid.is_occupied(np.array([0.25, 0.25, 3.75], dtype=np.float64)) is True
        assert grid.is_occupied(np.array([0.25, 0.25, 1.75], dtype=np.float64)) is False
        assert grid.is_occupied(np.array([50.0, 50.0, 50.0], dtype=np.float64)) is False


class TestRaycast:
    def test_hit(self):
        grid = _single_ray_grid()
        hit, dist = grid.raycast_3d(
            CAM_POS, np.array([0.0, 0.0, 1.0], dtype=np.float64), 4.0
        )
        assert hit is not None
        np.testing.assert_allclose(hit, [0.25, 0.25, 3.5])
        assert dist == pytest.approx(3.25)

    def test_miss_returns_max_range(self):
        grid = _single_ray_grid()
        hit, dist = grid.raycast_3d(
            CAM_POS, np.array([0.0, 0.0, -1.0], dtype=np.float64), 4.0
        )
        assert hit is None
        assert dist == pytest.approx(4.0)


class TestBresenham3D:
    def test_axis_aligned(self):
        grid = OccupancyGrid(BOUNDS, resolution=RES)
        line = grid._bresenham_3d((0, 0, 0), (5, 0, 0))
        assert len(line) == 6
        assert line[0] == (0, 0, 0)
        assert line[-1] == (5, 0, 0)

    def test_diagonal(self):
        grid = OccupancyGrid(BOUNDS, resolution=RES)
        line = grid._bresenham_3d((0, 0, 0), (3, 3, 3))
        assert [tuple(p) for p in line] == [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]

    def test_26_connectivity(self, rng):
        grid = OccupancyGrid(BOUNDS, resolution=RES)
        for _ in range(20):
            start = tuple(rng.integers(0, 8, size=3))
            end = tuple(rng.integers(0, 8, size=3))
            line = grid._bresenham_3d(start, end)
            diffs = np.diff(np.asarray(line), axis=0)
            assert np.all(np.abs(diffs) <= 1)
            # Every step moves (no repeated voxels), but may move in any
            # subset of axes (26-connectivity).
            assert np.all(np.any(diffs != 0, axis=1))
