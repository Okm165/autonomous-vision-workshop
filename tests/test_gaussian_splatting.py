"""Tests for src/gaussian_splatting.py — 3D Gaussian Splatting forward model."""

import numpy as np
import pytest

from src.gaussian_splatting import (
    Gaussian3D,
    build_covariance_3d,
    compute_loss,
    create_random_gaussians,
    gaussian_2d_pdf,
    project_gaussian_to_2d,
    render_gaussians,
    volume_render_equation,
)

K_BASIC = np.array(
    [
        [100.0, 0.0, 64.0],
        [0.0, 100.0, 64.0],
        [0.0, 0.0, 1.0],
    ]
)


def _identity_cam() -> np.ndarray:
    return np.eye(4)


# ======================================================================
# Covariance construction
# ======================================================================


class TestBuildCovariance3D:
    def test_identity_rotation_gives_diagonal(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        scale = np.log([0.5, 1.0, 2.0])
        S = build_covariance_3d(scale, q)
        assert np.allclose(S, np.diag([0.25, 1.0, 4.0]), atol=1e-12)

    def test_random_rotations_are_psd_and_symmetric(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            scale = rng.uniform(-3.0, 0.5, size=3)
            S = build_covariance_3d(scale, q)
            assert np.allclose(S, S.T, atol=1e-12)
            assert np.linalg.eigvalsh(S).min() >= -1e-12

    def test_matches_direct_RSSR_construction(self):
        rng = np.random.default_rng(1)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        s = np.exp(rng.uniform(-2.0, 0.0, size=3))
        # quaternion [w, x, y, z] -> rotation matrix (closed form)
        w, x, y, z = q
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        expected = R @ np.diag(s**2) @ R.T
        assert np.allclose(build_covariance_3d(np.log(s), q), expected, atol=1e-10)

    def test_gaussian3d_scales_property(self):
        g = Gaussian3D(
            position=np.zeros(3),
            scale=np.log([0.5, 1.0, 2.0]),
            quaternion=np.array([1.0, 0, 0, 0]),
            color=np.zeros(3),
            opacity=1.0,
        )
        assert np.allclose(g.scales, [0.5, 1.0, 2.0])
        assert np.allclose(g.covariance_3d, np.diag([0.25, 1.0, 4.0]), atol=1e-12)


# ======================================================================
# 2D Gaussian PDF
# ======================================================================


class TestGaussian2DPdf:
    """NOTE: by design this returns the *unnormalized* Gaussian
    G = exp(-1/2 dᵀ Σ⁻¹ d) — the prefactor cancels against opacity during
    compositing (documented in the function docstring).
    """

    def test_peak_at_mean_is_one(self):
        cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        pdf = gaussian_2d_pdf(1.0, 2.0, np.array([1.0, 2.0]), cov)
        assert pdf == pytest.approx(1.0)

    def test_integrates_to_two_pi_sqrt_det(self):
        cov = np.array([[2.0, 0.4], [0.4, 1.0]])
        mu = np.array([0.5, -0.25])
        xs = np.linspace(-6, 7, 1201)
        grid_x, grid_y = np.meshgrid(xs, xs)
        pdf = gaussian_2d_pdf(grid_x, grid_y, mu, cov)
        total = np.trapezoid(np.trapezoid(pdf, xs, axis=1), xs)
        assert total == pytest.approx(2 * np.pi * np.sqrt(np.linalg.det(cov)), rel=1e-4)

    def test_second_moments_match_covariance(self):
        cov = np.array([[3.0, 0.8], [0.8, 1.5]])
        mu = np.array([0.0, 0.0])
        xs = np.linspace(-12, 12, 1601)
        grid_x, grid_y = np.meshgrid(xs, xs)
        pdf = gaussian_2d_pdf(grid_x, grid_y, mu, cov)
        norm = np.trapezoid(np.trapezoid(pdf, xs, axis=1), xs)
        exx = np.trapezoid(np.trapezoid(grid_x**2 * pdf, xs, axis=1), xs) / norm
        eyy = np.trapezoid(np.trapezoid(grid_y**2 * pdf, xs, axis=1), xs) / norm
        exy = np.trapezoid(np.trapezoid(grid_x * grid_y * pdf, xs, axis=1), xs) / norm
        assert exx == pytest.approx(cov[0, 0], rel=1e-3)
        assert eyy == pytest.approx(cov[1, 1], rel=1e-3)
        assert exy == pytest.approx(cov[0, 1], rel=1e-3)

    def test_degenerate_covariance_returns_zero(self):
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # det = 0
        out = gaussian_2d_pdf(0.0, 0.0, np.zeros(2), cov)
        assert np.all(out == 0.0)


# ======================================================================
# Projection (EWA)
# ======================================================================


class TestProjectGaussianTo2D:
    def _gaussian(self, position):
        return Gaussian3D(
            position=np.asarray(position, dtype=np.float64),
            scale=np.log([0.1, 0.1, 0.1]),
            quaternion=np.array([1.0, 0, 0, 0]),
            color=np.ones(3),
            opacity=1.0,
        )

    def test_center_projects_to_principal_point(self):
        mu2d, _cov2d, depth = project_gaussian_to_2d(
            self._gaussian([0.0, 0.0, 5.0]), _identity_cam(), K_BASIC
        )
        assert depth == pytest.approx(5.0)
        assert mu2d == pytest.approx([64.0, 64.0])

    def test_lateral_offset_scales_with_fx_over_z(self):
        # x = 1.0 at z = 4.0 -> pixel offset = fx * x / z = 25 px
        mu2d, _, _ = project_gaussian_to_2d(
            self._gaussian([1.0, -0.5, 4.0]), _identity_cam(), K_BASIC
        )
        assert mu2d[0] == pytest.approx(64.0 + 100.0 * 1.0 / 4.0)
        assert mu2d[1] == pytest.approx(64.0 - 100.0 * 0.5 / 4.0)

    def test_projected_covariance_is_psd(self):
        rng = np.random.default_rng(2)
        for _ in range(10):
            g = Gaussian3D(
                position=rng.uniform(-0.5, 0.5, 3) + np.array([0, 0, 4.0]),
                scale=rng.uniform(-2.0, -0.5, 3),
                quaternion=rng.normal(size=4),
                color=np.ones(3),
                opacity=1.0,
            )
            _, cov2d, _ = project_gaussian_to_2d(g, _identity_cam(), K_BASIC)
            assert np.allclose(cov2d, cov2d.T, atol=1e-12)
            assert np.linalg.eigvalsh(cov2d).min() >= -1e-9


# ======================================================================
# Rendering
# ======================================================================


class TestRenderGaussians:
    def _one(self, position, color, opacity=0.9):
        return Gaussian3D(
            position=np.asarray(position, dtype=np.float64),
            scale=np.log([0.2, 0.2, 0.2]),
            quaternion=np.array([1.0, 0, 0, 0]),
            color=np.asarray(color, dtype=np.float64),
            opacity=opacity,
        )

    def test_single_gaussian_renders_bright_center(self):
        g = self._one([0.0, 0.0, 5.0], [1.0, 0.0, 0.0])
        img = render_gaussians([g], _identity_cam(), K_BASIC, (128, 128))
        assert img.shape == (128, 128, 3)
        center = img[64, 64]
        assert center[0] > 0.5  # strong red response at the splat centre
        assert center[1] < 1e-6
        assert center[2] < 1e-6
        # monotone falloff away from centre
        assert img[64, 70, 0] < center[0]
        assert img[70, 64, 0] < center[0]

    def test_image_values_in_unit_range(self):
        rng = np.random.default_rng(3)
        gs = create_random_gaussians(15, rng=rng)
        for g in gs:
            g.position += np.array([0, 0, 5.0])
        img = render_gaussians(gs, _identity_cam(), K_BASIC, (96, 96))
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_front_gaussian_occludes_behind(self):
        front = self._one([0.0, 0.0, 3.0], [1.0, 0.0, 0.0], opacity=1.0)
        back = self._one([0.0, 0.0, 8.0], [0.0, 0.0, 1.0], opacity=1.0)
        both = render_gaussians([front, back], _identity_cam(), K_BASIC, (128, 128))
        only_front = render_gaussians([front], _identity_cam(), K_BASIC, (128, 128))
        # centre fully covered by the opaque front splat
        assert np.allclose(both[64, 64], only_front[64, 64])
        assert both[64, 64, 2] < 1e-6  # no blue bleeds through

    def test_gaussian_behind_camera_is_ignored(self):
        g = self._one([0.0, 0.0, -5.0], [1.0, 1.0, 1.0], opacity=1.0)
        img = render_gaussians([g], _identity_cam(), K_BASIC, (64, 64))
        assert img.max() == 0.0

    def test_depth_sorting_independent_of_input_order(self):
        front = self._one([0.0, 0.0, 3.0], [1.0, 0.0, 0.0], opacity=0.6)
        back = self._one([0.0, 0.0, 8.0], [0.0, 0.0, 1.0], opacity=0.6)
        a = render_gaussians([front, back], _identity_cam(), K_BASIC, (128, 128))
        b = render_gaussians([back, front], _identity_cam(), K_BASIC, (128, 128))
        assert np.allclose(a, b, atol=1e-12)


# ======================================================================
# Losses
# ======================================================================


class TestComputeLoss:
    def test_identical_images_zero_loss(self):
        img = np.random.default_rng(4).uniform(0, 1, (48, 48, 3))
        losses = compute_loss(img, img.copy())
        assert losses["l1"] == pytest.approx(0.0, abs=1e-12)
        assert losses["total"] == pytest.approx(0.0, abs=1e-9)

    def test_known_l1_offset(self):
        rng = np.random.default_rng(5)
        target = rng.uniform(0.2, 0.8, (32, 32, 3))
        rendered = np.clip(target + 0.1, 0, 1)
        losses = compute_loss(rendered, target)
        assert losses["l1"] == pytest.approx(0.1, rel=1e-6)
        # total = 0.8 * l1 + 0.2 * (1 - ssim), 0 <= 1 - ssim <= 1
        assert 0.8 * 0.1 <= losses["total"] <= 0.8 * 0.1 + 0.2 + 1e-9

    def test_ssim_of_identical_images_is_one(self):
        from src.gaussian_splatting import _compute_ssim

        img = np.random.default_rng(6).uniform(0, 1, (40, 40, 3))
        assert _compute_ssim(img, img.copy()) == pytest.approx(1.0, abs=1e-6)

    def test_anticorrelated_images_have_negative_ssim(self):
        from src.gaussian_splatting import _compute_ssim

        img = np.repeat(
            np.linspace(0, 1, 64, dtype=np.float64)[None, :, None], 64, axis=0
        )
        ssim = _compute_ssim(img, 1.0 - img)
        assert ssim < 0.0


# ======================================================================
# create_random_gaussians
# ======================================================================


class TestCreateRandomGaussians:
    def test_count_and_ranges(self):
        gs = create_random_gaussians(30, bounds=(-2.0, 3.0))
        assert len(gs) == 30
        for g in gs:
            assert np.all(g.position >= -2.0)
            assert np.all(g.position <= 3.0)
            assert np.all(g.scale >= -2.0)
            assert np.all(g.scale <= 0.0)
            assert 0.3 <= g.opacity <= 1.0
            assert np.all((g.color >= 0.0) & (g.color <= 1.0))
            assert g.quaternion.shape == (4,)
            assert np.linalg.norm(g.quaternion) == pytest.approx(1.0, abs=1e-9)

    def test_rng_reproducibility(self):
        a = create_random_gaussians(10, rng=np.random.default_rng(42))
        b = create_random_gaussians(10, rng=np.random.default_rng(42))
        for ga, gb in zip(a, b, strict=False):
            assert np.array_equal(ga.position, gb.position)
            assert np.array_equal(ga.scale, gb.scale)
            assert np.array_equal(ga.quaternion, gb.quaternion)
            assert np.array_equal(ga.color, gb.color)
            assert ga.opacity == gb.opacity

    def test_two_draws_differ(self):
        a = create_random_gaussians(5, rng=np.random.default_rng(1))
        b = create_random_gaussians(5, rng=np.random.default_rng(2))
        assert not np.array_equal(a[0].position, b[0].position)


# ======================================================================
# Volume rendering equation (NeRF connection)
# ======================================================================


class TestVolumeRenderEquation:
    def test_single_opaque_sample(self):
        c = volume_render_equation(
            sigmas=np.array([1e6], dtype=np.float64),
            colors=np.array([[0.2, 0.5, 0.8]], dtype=np.float64),
            deltas=np.array([1.0], dtype=np.float64),
        )
        assert c == pytest.approx([0.2, 0.5, 0.8], abs=1e-6)

    def test_empty_space_returns_black(self):
        c = volume_render_equation(
            sigmas=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
            deltas=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        )
        assert np.allclose(c, 0.0)

    def test_uniform_absorbing_medium_takes_sample_color(self):
        # constant color: C = (1 - T_end) * c; total optical depth = 50*0.2 = 10
        c = volume_render_equation(
            sigmas=np.full(50, 1.0),
            colors=np.tile([0.3, 0.6, 0.9], (50, 1)),
            deltas=np.full(50, 0.2),
        )
        absorbed = 1.0 - np.exp(-10.0)
        assert c == pytest.approx(absorbed * np.array([0.3, 0.6, 0.9]), rel=1e-6)

    def test_weights_sum_bounded_by_one(self):
        rng = np.random.default_rng(7)
        sigmas = rng.uniform(0, 3, 40)
        deltas = rng.uniform(0.05, 0.3, 40)
        alpha = 1.0 - np.exp(-sigmas * deltas)
        tau = np.concatenate([[0.0], np.cumsum(sigmas[:-1] * deltas[:-1])])
        weights = alpha * np.exp(-tau)
        assert weights.sum() <= 1.0 + 1e-12
        assert weights.min() >= 0.0

    def test_order_matters(self):
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        sigmas = np.array([5.0, 5.0])
        deltas = np.array([1.0, 1.0])
        red_first = volume_render_equation(sigmas, colors, deltas)
        blue_first = volume_render_equation(sigmas, colors[::-1], deltas)
        assert red_first[0] > red_first[2]  # red in front dominates
        assert blue_first[2] > blue_first[0]

    def test_input_shape_validation(self):
        colors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        deltas = np.array([1.0], dtype=np.float64)
        with pytest.raises(AssertionError):
            volume_render_equation(np.array([1.0], dtype=np.float64), colors, deltas)
        with pytest.raises(AssertionError):
            volume_render_equation(
                np.array([1.0, 2.0], dtype=np.float64), colors, deltas
            )
