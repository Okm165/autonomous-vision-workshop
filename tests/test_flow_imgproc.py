"""Tests for src/imgproc.py and src/flow.py."""

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from src.flow import (
    compute_scene_flow,
    flow_to_color,
    flow_to_depth,
    lucas_kanade,
    pyramidal_lk,
    warp_image_flow,
)
from src.imgproc import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    canny_edges,
    convolve2d,
    gaussian_kernel,
    histogram_equalize,
    reconstruct_from_laplacian,
    sobel_gradients,
)

# ---------------------------------------------------------------------------
# imgproc — convolution
# ---------------------------------------------------------------------------


class TestConvolve2d:
    def test_delta_kernel_is_identity(self):
        img = np.arange(35.0).reshape(5, 7)
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1.0
        np.testing.assert_allclose(convolve2d(img, kernel), img, atol=1e-12)

    def test_delta_at_corner_shifts(self):
        # True convolution flips the kernel: a delta at (0, 0) means
        # out[y, x] = img[y + 1, x + 1] on the interior.
        img = np.arange(30.0).reshape(5, 6)
        kernel = np.zeros((3, 3))
        kernel[0, 0] = 1.0
        out = convolve2d(img, kernel)
        np.testing.assert_allclose(out[:-1, :-1], img[1:, 1:], atol=1e-12)

    def test_box_kernel_sums_neighbourhood(self):
        img = np.ones((5, 5))
        out = convolve2d(img, np.ones((3, 3)))
        np.testing.assert_allclose(out[2:-2, 2:-2], 9.0)


# ---------------------------------------------------------------------------
# imgproc — gaussian kernel
# ---------------------------------------------------------------------------


class TestGaussianKernel:
    def test_sums_to_one(self):
        for size, sigma in [(3, 0.8), (5, 1.0), (11, 2.5)]:
            assert gaussian_kernel(size, sigma).sum() == pytest.approx(1.0)

    def test_even_size_raises(self):
        with pytest.raises(ValueError, match="Kernel size must be odd"):
            gaussian_kernel(4, 1.0)

    def test_separable(self):
        size, sigma = 7, 1.3
        kernel = gaussian_kernel(size, sigma)
        x = np.arange(size) - size // 2
        g1d = np.exp(-(x**2) / (2.0 * sigma**2))
        g1d /= g1d.sum()
        np.testing.assert_allclose(kernel, np.outer(g1d, g1d), rtol=1e-12)


# ---------------------------------------------------------------------------
# imgproc — sobel (sign-corrected correlation-form kernels)
# ---------------------------------------------------------------------------


class TestSobel:
    def test_horizontal_ramp(self):
        a = 3.0
        img = np.tile(np.arange(12.0) * a, (10, 1))
        gx, gy, mag, direction = sobel_gradients(img)
        np.testing.assert_allclose(gx[2:-2, 2:-2], 8.0 * a)
        np.testing.assert_allclose(gy[2:-2, 2:-2], 0.0, atol=1e-12)
        np.testing.assert_allclose(mag[2:-2, 2:-2], 8.0 * a)
        np.testing.assert_allclose(direction[2:-2, 2:-2], 0.0, atol=1e-12)

    def test_vertical_ramp(self):
        b = 2.0
        img = np.tile((np.arange(10) * b)[:, None], (1, 12))
        gx, gy, _mag, direction = sobel_gradients(img)
        np.testing.assert_allclose(gx[2:-2, 2:-2], 0.0, atol=1e-12)
        np.testing.assert_allclose(gy[2:-2, 2:-2], 8.0 * b)
        np.testing.assert_allclose(direction[2:-2, 2:-2], np.pi / 2.0, atol=1e-12)


# ---------------------------------------------------------------------------
# imgproc — canny / pyramids / equalisation
# ---------------------------------------------------------------------------


class TestCanny:
    def test_bright_square(self):
        img = np.zeros((64, 64))
        img[20:44, 20:44] = 255.0
        edges = canny_edges(img, 50.0, 150.0)
        assert edges.dtype == np.uint8
        assert set(np.unique(edges)) <= {0, 255}
        assert edges.sum() > 0
        assert edges[32, 32] == 0  # flat interior produces no edges


class TestLaplacianPyramid:
    def test_reconstruction_is_exact(self, rng):
        img = rng.uniform(0.0, 1.0, (64, 64))
        levels = 3
        laps = build_laplacian_pyramid(img, levels)
        assert len(laps) == levels
        gaussian = build_gaussian_pyramid(img, levels)
        np.testing.assert_allclose(laps[-1], gaussian[-1], atol=1e-12)
        reconstructed = reconstruct_from_laplacian(laps)
        assert reconstructed.shape == img.shape
        np.testing.assert_allclose(reconstructed, img, atol=1e-8)


class TestHistogramEqualize:
    def test_two_levels_span_full_range(self):
        out = histogram_equalize(np.array([[100, 110]], dtype=np.uint8))
        np.testing.assert_array_equal(out, np.array([[0, 255]], dtype=np.uint8))


# ---------------------------------------------------------------------------
# flow — Lucas–Kanade
# ---------------------------------------------------------------------------


def _texture(seed, size=256, sigma=3.0) -> np.ndarray:
    """Smooth random texture — LK's Taylor expansion needs C1-continuous
    content; raw iid noise breaks the derivation entirely, and structure
    finer than the tracked displacement creates ambiguous optima."""
    gen = np.random.default_rng(seed)  # local seed: independent of fixture order
    return gaussian_filter(gen.normal(0.0, 1.0, (size, size)), sigma=sigma)


def _grid_points(lo, hi, step):
    xs = np.arange(lo, hi, step, dtype=np.float64)
    return np.array([[x, y] for y in xs for x in xs])


class TestLucasKanade:
    def test_integer_shift(self):
        prev = _texture(seed=2024)
        curr = np.zeros_like(prev)
        curr[:-2, 3:] = prev[2:, :-3]  # content moves (+3, -2)
        pts = _grid_points(40, 217, 40)
        new, status = lucas_kanade(prev, curr, pts, window_size=15)
        assert status.all()
        np.testing.assert_allclose(new[:, 0] - pts[:, 0], 3.0, atol=0.75)
        np.testing.assert_allclose(new[:, 1] - pts[:, 1], -2.0, atol=0.75)

    def test_subpixel_shift(self):
        prev = _texture(seed=2024).astype(np.float32)
        shift = (2.3, -1.4)
        m = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float64)
        curr = cv2.warpAffine(prev, m, (prev.shape[1], prev.shape[0]))
        pts = _grid_points(40, 217, 40)
        new, status = lucas_kanade(prev, curr, pts, window_size=15)
        assert status.all()
        np.testing.assert_allclose(new[:, 0] - pts[:, 0], shift[0], atol=0.75)
        np.testing.assert_allclose(new[:, 1] - pts[:, 1], shift[1], atol=0.75)

    def test_flat_region_fails(self):
        img = np.full((64, 64), 0.5)
        new, status = lucas_kanade(img, img, np.array([[32.0, 32.0]]), window_size=15)
        assert not status[0]
        np.testing.assert_allclose(new[0], [32.0, 32.0])

    def test_border_point_fails(self):
        prev = _texture(seed=7, size=64)
        curr = prev.copy()
        new, status = lucas_kanade(prev, curr, np.array([[2.0, 2.0]]), window_size=15)
        assert not status[0]
        np.testing.assert_allclose(new[0], [2.0, 2.0])


class TestPyramidalLK:
    def test_large_displacement(self):
        # 9 px exceeds the single-scale search range (~half the 15 px
        # window) yet is recovered exactly thanks to coarse-to-fine.
        prev = _texture(seed=2024)
        curr = np.zeros_like(prev)
        curr[:-6, 9:] = prev[6:, :-9]  # content moves (+9, -6)
        # Interior grid: every point's window stays in bounds at the
        # coarsest level (where the image is 64x64) even after shifting.
        pts = _grid_points(48, 209, 32)
        new, status = pyramidal_lk(prev, curr, pts, levels=3, window_size=15)
        assert status.all()
        np.testing.assert_allclose(new[:, 0] - pts[:, 0], 9.0, atol=0.75)
        np.testing.assert_allclose(new[:, 1] - pts[:, 1], -6.0, atol=0.75)


# ---------------------------------------------------------------------------
# flow — colour wheel / depth / scene flow / warping
# ---------------------------------------------------------------------------


class TestFlowToColor:
    def test_positive_x_is_red(self):
        flow = np.zeros((1, 1, 2))
        flow[0, 0] = [1.0, 0.0]
        bgr = flow_to_color(flow)
        assert bgr.dtype == np.uint8
        np.testing.assert_array_equal(bgr[0, 0], [0, 0, 255])

    def test_negative_x_is_cyan(self):
        flow = np.zeros((1, 1, 2))
        flow[0, 0] = [-1.0, 0.0]
        bgr = flow_to_color(flow)
        np.testing.assert_array_equal(bgr[0, 0], [255, 255, 0])

    def test_zero_flow_is_black(self):
        flow = np.zeros((2, 2, 2))
        bgr = flow_to_color(flow)
        np.testing.assert_array_equal(bgr, 0)


class TestFlowToDepth:
    K: NDArray[np.float64] = np.array(
        [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]
    )

    def _true_depth(self, h=4, w=5):
        us, vs = np.meshgrid(
            np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)
        )
        return 3.0 + 0.5 * us + 0.25 * vs, us, vs

    def test_pure_x_translation_exact(self):
        z_true, _, _ = self._true_depth()
        flow = np.zeros((*z_true.shape, 2))
        flow[..., 0] = self.K[0, 0] * 0.2 / z_true
        depth = flow_to_depth(flow, self.K, np.eye(3), np.array([0.2, 0.0, 0.0]))
        np.testing.assert_allclose(depth, z_true, rtol=1e-10)

    def test_translation_with_z_component(self):
        # With t = (tx, 0, tz) the recovered depth is the *relative* depth
        # (Z + tz) / (1 + tz) — an exact algebraic identity of the
        # decomposition Z = ||f_trans|| / ||f - f_rot||.
        tx, tz = 0.2, 0.3
        z_true, us, vs = self._true_depth()
        denom = z_true + tz
        flow = np.zeros((*z_true.shape, 2))
        flow[..., 0] = (self.K[0, 0] * tx - us * tz) / denom
        flow[..., 1] = -vs * tz / denom
        depth = flow_to_depth(flow, self.K, np.eye(3), np.array([tx, 0.0, tz]))
        np.testing.assert_allclose(depth, (z_true + tz) / (1.0 + tz), rtol=1e-10)


class TestSceneFlow:
    K: NDArray[np.float64] = np.array(
        [[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 1.0]]
    )

    def test_static_scene_zero_flow(self):
        depth = np.full((3, 3), 5.0)
        flow = np.zeros((3, 3, 2))
        sf = compute_scene_flow(flow, depth, depth, self.K)
        np.testing.assert_allclose(sf, 0.0, atol=1e-12)

    def test_depth_step_is_ray_direction(self):
        depth1 = np.full((3, 3), 5.0)
        depth2 = depth1 + 1.0
        flow = np.zeros((3, 3, 2))
        sf = compute_scene_flow(flow, depth1, depth2, self.K)
        ray = np.linalg.inv(self.K) @ np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(sf[1, 1], ray, atol=1e-12)

    def test_ego_motion_subtraction(self):
        depth = np.full((3, 3), 5.0)
        flow = np.zeros((3, 3, 2))
        t = np.array([0.1, 0.0, 0.0])
        t12 = np.eye(4)
        t12[:3, 3] = t
        sf = compute_scene_flow(flow, depth, depth, self.K, T_12=t12)
        # A static world observed through a translating camera yields
        # scene flow = -t at every pixel.
        np.testing.assert_allclose(sf, np.broadcast_to(-t, sf.shape), atol=1e-12)


class TestWarpImageFlow:
    def test_constant_flow(self):
        img = np.arange(48.0).reshape(6, 8)
        flow = np.zeros((6, 8, 2))
        flow[..., 0] = 3.0
        warped = warp_image_flow(img, flow)
        # Backward warp: out[y, x] = img[y, x + 3].
        np.testing.assert_allclose(warped[:, :5], img[:, 3:])
        # Border replication fills the remainder.
        np.testing.assert_allclose(warped[:, -1], img[:, -1])
