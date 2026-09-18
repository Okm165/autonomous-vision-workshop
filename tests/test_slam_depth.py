"""Tests for robust kernels (src/slam.py), depth alignment (src/depth.py),
the pose-graph optimiser, and src/viz.py smoke coverage."""

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from src.depth import align_depth_to_ground_truth
from src.slam import (
    PoseGraph,
    cauchy_kernel,
    geman_mcclure_kernel,
    huber_kernel,
)
from src.transforms import se3_compose, se3_from_Rt, se3_inverse, se3_log

# ---------------------------------------------------------------------------
# Robust kernels
# ---------------------------------------------------------------------------


class TestHuberKernel:
    def test_quadratic_region(self):
        cost, weight = huber_kernel(0.5, delta=1.0)
        assert cost == pytest.approx(0.125)
        assert weight == 1.0

    def test_linear_region(self):
        cost, weight = huber_kernel(2.0, delta=1.0)
        assert cost == pytest.approx(1.0 * (2.0 - 0.5))  # δ(|r| − δ/2)
        assert weight == pytest.approx(1.0 / 2.0)  # δ / |r|

    def test_seam_continuity(self):
        # Both branches agree exactly at |r| = δ.
        delta = 1.7
        cost_in, _ = huber_kernel(delta, delta)
        cost_out, _ = huber_kernel(delta + 1e-12, delta)
        assert cost_out == pytest.approx(cost_in, rel=1e-9)
        assert cost_in == pytest.approx(0.5 * delta**2)

    def test_symmetry(self):
        c1, w1 = huber_kernel(1.3, 0.8)
        c2, w2 = huber_kernel(-1.3, 0.8)
        assert c1 == pytest.approx(c2)
        assert w1 == pytest.approx(w2)


class TestCauchyKernel:
    def test_at_zero(self):
        cost, weight = cauchy_kernel(0.0, c=1.0)
        assert cost == 0.0
        assert weight == 1.0

    def test_analytic_value(self):
        r, c = 2.0, 1.0
        cost, weight = cauchy_kernel(r, c)
        assert cost == pytest.approx(0.5 * c**2 * np.log1p((r / c) ** 2))
        assert weight == pytest.approx(1.0 / (1.0 + (r / c) ** 2))

    def test_redescending_influence(self):
        # Weight decays to zero for large residuals.
        assert cauchy_kernel(100.0, c=1.0)[1] < 1e-3


class TestGemanMcClureKernel:
    def test_at_zero(self):
        cost, weight = geman_mcclure_kernel(0.0, c=1.0)
        assert cost == 0.0
        assert weight == pytest.approx(1.0)

    def test_analytic_value(self):
        r, c = 2.0, 1.0
        cost, weight = geman_mcclure_kernel(r, c)
        assert cost == pytest.approx(0.5 * r**2 / (c**2 + r**2))
        assert weight == pytest.approx(c**2 / (c**2 + r**2) ** 2)

    def test_bounded_cost(self):
        # rho < 1/2 for any residual when c = 1.
        assert geman_mcclure_kernel(1e6, c=1.0)[0] < 0.5


# ---------------------------------------------------------------------------
# Depth alignment
# ---------------------------------------------------------------------------


class TestAlignDepthToGroundTruth:
    def test_exact_linear_relationship(self):
        pred = np.arange(16, dtype=np.float64).reshape(4, 4) + 1.0
        gt = 2.0 * pred + 3.0
        scale, shift, aligned = align_depth_to_ground_truth(pred, gt)
        assert scale == pytest.approx(2.0, abs=1e-9)
        assert shift == pytest.approx(3.0, abs=1e-9)
        assert aligned.dtype == np.float32
        np.testing.assert_allclose(aligned, gt, rtol=1e-6)

    def test_mask_restricts_fit(self):
        pred = np.arange(16, dtype=np.float64).reshape(4, 4) + 1.0
        gt = 3.0 * pred + 1.0
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2] = True  # fit only on half the pixels
        scale, shift, aligned = align_depth_to_ground_truth(pred, gt, mask=mask)
        assert scale == pytest.approx(3.0, abs=1e-9)
        assert shift == pytest.approx(1.0, abs=1e-9)
        # Alignment applies to the whole map, not just the mask.
        np.testing.assert_allclose(aligned, gt, rtol=1e-6)

    def test_too_few_valid_pixels(self):
        pred = np.full((2, 2), 4.0)
        gt = np.array([[1.0, 0.0], [0.0, 0.0]])  # single valid pixel
        scale, shift, aligned = align_depth_to_ground_truth(pred, gt)
        assert scale == 1.0
        assert shift == 0.0
        np.testing.assert_array_equal(aligned, pred)


# ---------------------------------------------------------------------------
# Pose graph
# ---------------------------------------------------------------------------


def _square_poses():
    """Four poses driving a 1 m CCW square in the xy-plane, facing travel."""
    poses = []
    headings = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    positions = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    for pos, th in zip(positions, headings, strict=False):
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        poses.append(se3_from_Rt(R, np.array(pos, dtype=np.float64)))
    return poses


class TestPoseGraphAPI:
    def test_add_nodes_and_edges(self):
        graph = PoseGraph()
        assert graph.add_node(np.eye(4)) == 0
        assert graph.add_node(np.eye(4)) == 1
        graph.add_edge(0, 1, np.eye(4))
        assert len(graph.edges) == 1
        i, j, _t_ij, info = graph.edges[0]
        assert (i, j) == (0, 1)
        np.testing.assert_array_equal(info, np.eye(6))

    def test_loop_closure_default_information(self):
        graph = PoseGraph()
        graph.add_node(np.eye(4))
        graph.add_node(np.eye(4))
        graph.add_loop_closure(0, 1, np.eye(4))
        np.testing.assert_array_equal(graph.edges[0][3], np.eye(6) * 100)


class TestPoseGraphOptimisation:
    def test_drifted_square_corrected_by_loop_closure(self):
        poses = _square_poses()
        graph = PoseGraph()
        for p in poses:
            graph.add_node(p)

        drift = 1.05  # odometry over-estimates each step by 5%
        for i in range(3):
            t_ij = se3_compose(se3_inverse(poses[i]), poses[i + 1])
            t_ij = t_ij.copy()
            t_ij[:3, 3] *= drift
            graph.add_edge(i, i + 1, t_ij)

        # Exact loop closure with high information.
        t_30 = se3_compose(se3_inverse(poses[3]), poses[0])
        graph.add_loop_closure(3, 0, t_30, information=np.eye(6) * 100.0)

        # Initialise by composing drifted odometry.
        init = [poses[0]]
        for i in range(3):
            t_ij = graph.edges[i][2]
            init.append(se3_compose(init[-1], t_ij))
        for k, p in enumerate(init):
            graph.poses[k] = p.copy()

        def mean_position_error(node_poses):
            gt = np.array([p[:3, 3] for p in poses])
            est = np.array([p[:3, 3] for p in node_poses])
            return float(np.linalg.norm(est - gt, axis=1).mean())

        err_before = mean_position_error(graph.poses)
        assert err_before > 0.04  # drift is present

        graph.optimize(n_iterations=30, fixed_nodes=[0])
        err_after = mean_position_error(graph.poses)
        assert err_after < err_before

        # The real convergence criteria: the weighted least-squares cost
        # collapses and the high-information loop closure is enforced.
        def graph_cost(node_poses):
            probe = PoseGraph()
            probe.poses = [p.copy() for p in node_poses]
            probe.edges = graph.edges
            return sum(
                float(e @ info @ e)
                for i, j, t_ij, info in graph.edges
                for e in [probe._compute_error(i, j, t_ij)]
            )

        cost_before = graph_cost(init)
        cost_after = graph_cost(graph.poses)
        assert cost_after < 0.01 * cost_before

        t_30 = graph.edges[3][2]
        loop_after = float(np.linalg.norm(graph._compute_error(3, 0, t_30)))
        assert loop_after < 0.01

    def test_consistent_graph_stays_put(self):
        poses = _square_poses()
        graph = PoseGraph()
        for p in poses:
            graph.add_node(p)
        for i in range(3):
            graph.add_edge(i, i + 1, se3_compose(se3_inverse(poses[i]), poses[i + 1]))

        optimized = graph.optimize(n_iterations=10, fixed_nodes=[0])
        for est, gt in zip(optimized, poses, strict=False):
            np.testing.assert_allclose(est, gt, atol=1e-6)

    def test_edge_error_is_zero_for_consistent_edge(self):
        poses = _square_poses()
        graph = PoseGraph()
        graph.add_node(poses[0])
        graph.add_node(poses[1])
        t_01 = se3_compose(se3_inverse(poses[0]), poses[1])
        graph.add_edge(0, 1, t_01)
        np.testing.assert_allclose(graph._compute_error(0, 1, t_01), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# viz smoke tests (Agg backend, figures closed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def viz():
    import matplotlib.pyplot as plt

    import src.viz

    yield src.viz
    plt.close("all")


class TestVizSmoke:
    def test_create_3d_axes(self, viz):
        import matplotlib.pyplot as plt

        ax = viz.create_3d_axes(title="t")
        assert ax is not None
        plt.close("all")

    def test_plot_cameras_3d(self, viz):
        import matplotlib.pyplot as plt

        poses = _square_poses()
        ax = viz.plot_cameras_3d(poses, scale=0.2)
        assert ax is not None
        plt.close("all")

    def test_plot_pointcloud_3d(self, viz):
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        pts = rng.uniform(-1, 1, (500, 3))
        ax = viz.plot_pointcloud_3d(pts)
        assert ax is not None
        plt.close("all")

    def test_plot_depth_map(self, viz):
        import matplotlib.pyplot as plt

        depth = np.linspace(0.5, 5.0, 64).reshape(8, 8)
        ax = viz.plot_depth_map(depth)
        assert ax is not None
        plt.close("all")

    def test_plot_trajectory(self, viz):
        import matplotlib.pyplot as plt

        poses = _square_poses()
        ax = viz.plot_trajectory(poses, gt_poses=poses)
        assert ax is not None
        plt.close("all")

    def test_depth_colormap(self, viz):
        depth = np.linspace(0.5, 5.0, 64).reshape(8, 8)
        colored = viz.depth_colormap(depth)
        assert colored.dtype == np.uint8
        assert colored.shape == (8, 8, 3)

    def test_side_by_side_comparison(self, viz):
        import matplotlib.pyplot as plt

        a = np.random.default_rng(1).uniform(0, 1, (16, 16))
        b = np.random.default_rng(2).uniform(0, 1, (16, 16))
        fig = viz.side_by_side_comparison([a, b], ["a", "b"])
        assert fig is not None
        plt.close("all")

    def test_render_3d_scene(self, viz):
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(3)
        pts = rng.uniform(-1, 1, (300, 3))
        fig = viz.render_3d_scene(points=pts, poses=_square_poses())
        assert fig is not None
        plt.close("all")

    def test_se3_log_roundtrip(self):
        # Sanity check of the SE(3) helpers the pose graph relies on.
        xi = se3_log(np.eye(4))
        np.testing.assert_allclose(xi, 0.0, atol=1e-12)
