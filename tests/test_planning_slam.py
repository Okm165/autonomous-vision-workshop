"""Path planning (A*, RRT*) and occupancy log-odds mapping tests."""

import numpy as np
import pytest

from src.occupancy import OccupancyGrid
from src.path_planning import (
    OccupancyGrid3D,
    astar_3d,
    create_random_obstacles,
    rrt_star,
)

# ---------------------------------------------------------------------------
# Occupancy grid log-odds (2-D facade over the 3-D voxel grid is not part of
# the API; we test the real 3-D log-odds semantics end to end).
# ---------------------------------------------------------------------------

BOUNDS = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])


def _simple_depth_scene():
    """Depth image of a wall at 3 m seen by a camera at the origin."""
    K = np.array([[100.0, 0, 64.0], [0, 100.0, 64.0], [0, 0, 1.0]])
    depth = np.full((128, 128), 3.0)
    return depth, K, np.eye(4)


class TestOccupancyLogOdds:
    def test_update_marks_surface_occupied_and_rays_free(self):
        grid = OccupancyGrid(BOUNDS, resolution=0.2)
        depth, K, T = _simple_depth_scene()
        grid.update(depth, K, T)
        occupied = grid.get_occupied_points(0.7)
        free = grid.get_free_points(0.3)
        assert len(occupied) > 0
        assert len(free) > 0
        # the wall sits at z ~ 3 m in the camera (= world) frame
        assert np.abs(occupied[:, 2] - 3.0).max() < 0.3
        # rays travelled towards the wall -> free voxels strictly before it
        assert free[:, 2].max() < 3.0 + 1e-9

    def test_log_odds_clamped(self):
        grid = OccupancyGrid(BOUNDS, resolution=0.2)
        depth, K, T = _simple_depth_scene()
        for _ in range(50):
            grid.update(depth, K, T)
        assert np.abs(grid.grid).max() <= grid.log_odds_max + 1e-9

    def test_repeated_free_observation_drives_to_free(self):
        grid = OccupancyGrid(BOUNDS, resolution=0.2)
        depth, K, T = _simple_depth_scene()
        for _ in range(20):
            grid.update(depth, K, T)
        # a voxel on the optical axis at z = 1 m was traversed many times
        idx = grid._world_to_voxel(np.array([0.64, 0.64, 1.0]))  # u=v=64 principal
        assert grid.grid[idx] < 0.0


def _empty_grid(resolution=0.5):
    grid = np.zeros((20, 20, 20), dtype=bool)
    return OccupancyGrid3D(grid, resolution, np.zeros(3))


class TestAStar3D:
    def test_straight_line_in_free_space(self):
        grid = _empty_grid()
        start, goal = np.array([1.0, 1.0, 1.0]), np.array([9.0, 9.0, 9.0])
        path = astar_3d(grid, start, goal)
        assert path is not None
        assert np.allclose(path[0], start, atol=1e-9)
        assert np.allclose(path[-1], goal, atol=1e-9)

    def test_respects_obstacles(self):
        grid = create_random_obstacles(
            (20, 20, 20),
            n_obstacles=8,
            obstacle_size_range=(2, 4),
            resolution=0.5,
            seed=3,
        )
        start, goal = np.array([0.5, 0.5, 0.5]), np.array([9.5, 9.5, 9.5])
        path = astar_3d(grid, start, goal)
        if path is None:
            pytest.skip("random obstacles disconnected start from goal")
        for p in path:
            assert grid.is_free_world(p), f"path point {p} inside an obstacle"

    def test_unreachable_returns_none(self):
        grid = _empty_grid()
        # wall of obstacles separating x < 5 from x > 5
        grid.grid[10, :, :] = True
        path = astar_3d(grid, np.array([1.0, 5.0, 5.0]), np.array([9.0, 5.0, 5.0]))
        assert path is None


class TestRRTStar:
    def test_reaches_goal_in_free_space(self):
        grid = _empty_grid()
        path = rrt_star(
            grid,
            np.array([1.0, 1.0, 1.0]),
            np.array([9.0, 9.0, 9.0]),
            max_iterations=5000,
            seed=42,
        )
        assert path is not None
        assert np.allclose(path[0], [1.0, 1.0, 1.0], atol=1e-6)
        assert np.linalg.norm(path[-1] - np.array([9.0, 9.0, 9.0])) < 0.35

    def test_rrt_star_shorter_than_or_equal_to_rrt(self):
        from src.path_planning import path_length, rrt

        grid = _empty_grid()
        start, goal = np.array([1.0, 1.0, 1.0]), np.array([9.0, 9.0, 9.0])
        p_rrt = rrt(grid, start, goal, max_iterations=3000, seed=1)
        p_star = rrt_star(grid, start, goal, max_iterations=3000, seed=1)
        if p_rrt is None or p_star is None:
            pytest.skip("planner failed to find a path with this seed")
        assert path_length(p_star) <= path_length(p_rrt) + 1e-6
