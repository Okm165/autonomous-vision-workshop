"""Smoke tests for src/viz.py — every plot function runs headless (Agg)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import viz


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


K = np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])


def _poses(n=4):
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[:3, 3] = [i * 0.5, 0.0, 0.0]
        poses.append(T)
    return poses


class TestCamerasAndScenes:
    def test_plot_cameras_3d_returns_axes(self):
        ax = viz.plot_cameras_3d(_poses(), K=K)
        assert ax is not None

    def test_plot_pointcloud_3d(self):
        pts = np.random.default_rng(0).uniform(-1, 1, (500, 3))
        colors = np.random.default_rng(1).uniform(0, 1, (500, 3))
        ax = viz.plot_pointcloud_3d(pts, colors=colors)
        assert ax is not None

    def test_render_3d_scene_combines_points_and_poses(self):
        pts = np.random.default_rng(2).uniform(-1, 1, (200, 3))
        ax = viz.render_3d_scene(points=pts, poses=_poses(), K=K)
        assert ax is not None

    def test_render_3d_scene_empty(self):
        ax = viz.render_3d_scene()
        assert ax is not None


class TestImagePlots:
    def test_plot_depth_map(self):
        depth = np.linspace(1, 10, 64 * 48).reshape(48, 64)
        fig = viz.plot_depth_map(depth)
        assert fig is not None

    def test_plot_matches(self):
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        kp1 = np.array([[10.0, 10.0], [40.0, 30.0]])
        kp2 = np.array([[12.0, 11.0], [41.0, 29.0]])
        ax = viz.plot_matches(img, kp1, img, kp2, matches=[(0, 0), (1, 1)])
        assert ax is not None

    def test_side_by_side_comparison(self):
        a = np.zeros((32, 32))
        b = np.ones((32, 32, 3))
        fig = viz.side_by_side_comparison([a, b], ["gray", "rgb"])
        assert fig is not None

    def test_depth_colormap(self):
        depth = np.linspace(0, 5, 48).reshape(6, 8)
        out = viz.depth_colormap(depth)
        assert out.dtype == np.uint8
        assert out.shape == (6, 8, 3)
        # invalid (0) pixels are black
        assert np.all(out[0, 0] == 0)
        assert out[1:, 1:].max() > 0

    def test_flow_to_color_zero_flow_is_neutral(self):
        flow = np.zeros((16, 16, 2))
        out = viz.flow_to_color(flow)
        assert out.dtype == np.uint8
        assert out.shape == (16, 16, 3)
        # zero magnitude -> saturation 0 -> grayscale value 255
        assert np.all(out == 255)

    def test_flow_to_color_direction_encoding(self):
        h, w = 8, 8
        _yy, _xx = np.mgrid[:h, :w]
        flow = np.zeros((h, w, 2))
        flow[..., 0] = 1.0  # pure +x flow everywhere
        out = viz.flow_to_color(flow, max_flow=1.0)
        assert out.shape == (h, w, 3)
        # uniform field -> uniform color
        assert np.all(out == out[0, 0])


class TestTrajectoryAndGeometry:
    def test_plot_trajectory_with_gt(self):
        est, gt = _poses(6), _poses(6)
        for T in gt:
            T[0, 3] += 0.01
        ax = viz.plot_trajectory(est, gt_poses=gt)
        assert ax is not None

    def test_plot_covariance_ellipse(self):
        ax = plt.gca()
        ax = viz.plot_covariance_ellipse(
            mean=np.array([1.0, 2.0]),
            cov=np.array([[1.0, 0.3], [0.3, 0.5]]),
            ax=ax,
        )
        assert ax is not None

    def test_create_3d_axes(self):
        ax = viz.create_3d_axes()
        assert ax is not None

    def test_make_video_frames_pads_and_converts(self):
        imgs = [
            np.zeros((30, 20), dtype=np.uint8),
            np.ones((25, 40, 3), dtype=np.uint8),
        ]
        frames = viz.make_video_frames(imgs, fps=10)
        assert len(frames) == 2
        assert all(f.shape == (30, 40, 3) for f in frames)
        assert all(f.dtype == np.uint8 for f in frames)

    def test_make_video_frames_empty(self):
        assert viz.make_video_frames([]) == []
