"""Tests for trajectory / depth / reconstruction evaluation metrics (src/eval.py)."""

import numpy as np
import pytest

from src.eval import (
    align_trajectories_umeyama,
    compute_ate,
    compute_depth_metrics,
    compute_reconstruction_metrics,
    compute_rpe,
    compute_scale_error,
)
from src.transforms import rodrigues


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Umeyama Sim(3) alignment
# ---------------------------------------------------------------------------


class TestUmeyama:
    def test_exact_sim3_recovery(self, rng):
        """target = s·R@src + t exactly  →  parameters recovered exactly."""
        src = rng.uniform(-3, 3, (40, 3))
        angle = 0.7
        R_true = rodrigues(np.array([0.3, -0.5, 0.8]), angle)
        s_true = 2.5
        t_true = np.array([4.0, -2.0, 1.0])
        tgt = s_true * (src @ R_true.T) + t_true

        s, R, t = align_trajectories_umeyama(src, tgt)
        assert s == pytest.approx(s_true)
        np.testing.assert_allclose(R, R_true, atol=1e-12)
        np.testing.assert_allclose(t, t_true, atol=1e-12)

    def test_rotation_has_positive_determinant(self, rng):
        """A mirrored point set must still yield a proper rotation (det +1)."""
        src = rng.uniform(-1, 1, (30, 3))
        tgt = src @ np.diag([-1.0, 1.0, 1.0]).T + np.array([1.0, 2.0, 3.0])
        _, R, _ = align_trajectories_umeyama(src, tgt)
        assert np.linalg.det(R) > 0.999

    def test_translation_only(self):
        src = np.random.default_rng(1).uniform(0, 1, (10, 3))
        tgt = src + np.array([5.0, -3.0, 2.0])
        s, R, t = align_trajectories_umeyama(src, tgt)
        assert s == pytest.approx(1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(t, [5.0, -3.0, 2.0], atol=1e-12)


# ---------------------------------------------------------------------------
# ATE
# ---------------------------------------------------------------------------


class TestATE:
    def _trajectory(self, n=20, step=np.array([0.5, 0.1, 0.0])):
        return [_pose(np.eye(3), i * step) for i in range(n)]

    def test_identical_trajectories_zero_error(self):
        poses = self._trajectory()
        rmse, mean, median, aligned, s = compute_ate(poses, poses)
        assert rmse < 1e-12
        assert mean < 1e-12
        assert median < 1e-12
        assert s == pytest.approx(1.0)
        np.testing.assert_allclose(
            aligned, np.array([p[:3, 3] for p in poses]), atol=1e-9
        )

    def test_scale_recovery(self):
        """Estimated trajectory at half metric scale → Umeyama scale = 2."""
        gt = self._trajectory()
        est = [_pose(np.eye(3), 0.5 * p[:3, 3]) for p in gt]
        rmse, _, _, _, s = compute_ate(est, gt)
        assert s == pytest.approx(2.0)
        assert rmse < 1e-9

    def test_noise_gives_expected_rmse(self, rng):
        """IID position noise with σ → aligned ATE RMSE ≈ σ·√(2 (1 − 1/N))-ish.

        For small noise the Umeyama alignment removes ~3 dof, so the RMSE
        of the *aligned* residuals is slightly below σ; we assert the
        coarse range around σ.
        """
        gt = self._trajectory(50)
        sigma = 0.05
        est = [_pose(np.eye(3), p[:3, 3] + rng.normal(0, sigma, 3)) for p in gt]
        rmse, _, _, _, s = compute_ate(est, gt)
        assert 0.3 * sigma < rmse < 2.0 * sigma
        assert 0.9 < s < 1.1

    def test_rotation_offset_recovered(self, rng):
        """Global rotation + translation of the estimate is aligned away."""
        gt = self._trajectory(30)
        R = rodrigues(np.array([0.1, 0.4, -0.2]), 0.9)
        t0 = np.array([10.0, -5.0, 2.0])
        est = [_pose(R @ np.eye(3), R @ p[:3, 3] + t0) for p in gt]
        rmse, _, _, _, _ = compute_ate(est, gt)
        assert rmse < 1e-9


# ---------------------------------------------------------------------------
# RPE
# ---------------------------------------------------------------------------


class TestRPE:
    def test_identical_zero(self):
        poses = [_pose(np.eye(3), np.array([i * 0.3, 0.0, 0.0])) for i in range(15)]
        trans_rmse, _trans_mean, rot_rmse, _rot_mean = compute_rpe(poses, poses)
        assert trans_rmse == 0.0
        assert rot_rmse == 0.0

    def test_constant_offset_is_invisible_to_rpe(self):
        """RPE is relative: a constant drift offset cancels exactly."""
        gt = [_pose(np.eye(3), np.array([i * 1.0, 0.0, 0.0])) for i in range(12)]
        off = np.array([3.0, -2.0, 1.0])
        est = [_pose(np.eye(3), p[:3, 3] + off) for p in gt]
        trans_rmse, _, rot_rmse, _ = compute_rpe(est, gt)
        assert trans_rmse < 1e-9
        assert rot_rmse < 1e-9

    def test_scale_error_per_step(self):
        """est = 2·gt: each relative step doubles → per-step error = ‖step‖ = 1."""
        gt = [_pose(np.eye(3), np.array([i * 1.0, 0.0, 0.0])) for i in range(12)]
        est = [_pose(np.eye(3), 2.0 * p[:3, 3]) for p in gt]
        trans_rmse, trans_mean, rot_rmse, _ = compute_rpe(est, gt)
        assert trans_rmse == 1.0
        assert trans_mean == 1.0
        assert rot_rmse == 0.0

    def test_pure_rotation_increment_error(self):
        """Rotation-only increments: per-step rotation error = |Δangle|."""
        alpha = 0.1
        e = 0.02

        def rot_pose(theta):
            return _pose(rodrigues(np.array([0.0, 0.0, 1.0]), theta), np.zeros(3))

        gt = [rot_pose(i * alpha) for i in range(20)]
        est = [rot_pose(i * (alpha + e)) for i in range(20)]
        _, _, rot_rmse, rot_mean = compute_rpe(est, gt)
        np.testing.assert_allclose(rot_rmse, e, atol=1e-9)
        np.testing.assert_allclose(rot_mean, e, atol=1e-9)

    def test_delta_parameter_skips_frames(self):
        """With alternating per-step error e, delta=2 steps see zero error."""
        gt = [_pose(np.eye(3), np.array([i * 1.0, 0.0, 0.0])) for i in range(11)]
        e = 0.05
        est = []
        for i in range(11):
            offset = e if i % 2 else 0.0
            est.append(_pose(np.eye(3), np.array([i * 1.0, offset, 0.0])))

        rmse_d1, _, _, _ = compute_rpe(est, gt, delta=1)
        rmse_d2, _, _, _ = compute_rpe(est, gt, delta=2)
        np.testing.assert_allclose(rmse_d1, e, atol=1e-12)
        assert rmse_d2 < 1e-12


# ---------------------------------------------------------------------------
# Scale error
# ---------------------------------------------------------------------------


def test_scale_error():
    err, pct = compute_scale_error(1.1)
    np.testing.assert_allclose(err, 0.1)
    np.testing.assert_allclose(pct, 10.0)
    err, pct = compute_scale_error(0.92)
    np.testing.assert_allclose(err, 0.08)
    np.testing.assert_allclose(pct, 8.0)


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------


class TestDepthMetrics:
    def test_hand_computed_values(self):
        gt = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = np.array([[1.25, 2.5], [3.0, 4.0]])  # +25 % on two pixels

        m = compute_depth_metrics(pred, gt)

        diffs = np.array([0.25, 0.5, 0.0, 0.0])
        assert m["abs_rel"] == np.mean(np.array([0.25, 0.25, 0.0, 0.0]))
        assert m["abs_rel"] == 0.125
        # sq_rel = mean(diff² / gt)
        np.testing.assert_allclose(
            m["sq_rel"], np.mean([0.25**2 / 1.0, 0.5**2 / 2.0, 0.0, 0.0])
        )
        np.testing.assert_allclose(m["rmse"], np.sqrt(np.mean(diffs**2)))
        # both perturbed pixels have ratio 1.25 → mean over 4 pixels
        np.testing.assert_allclose(m["rmse_log"], np.log(1.25) / np.sqrt(2))
        # ratio = 1.25 exactly on two pixels → strict '<' excludes them
        assert m["delta_1"] == 0.5
        assert m["delta_2"] == 1.0
        assert m["delta_3"] == 1.0

    def test_identical_maps(self):
        gt = np.random.default_rng(3).uniform(1, 10, (8, 8))
        m = compute_depth_metrics(gt, gt)
        assert m["abs_rel"] == 0.0
        assert m["rmse"] == 0.0
        assert m["delta_1"] == 1.0

    def test_mask_selects_pixels(self):
        gt = np.array([[1.0, 100.0]])
        pred = np.array([[2.0, 100.0]])
        m_all = compute_depth_metrics(pred, gt)
        m_masked = compute_depth_metrics(pred, gt, mask=np.array([[True, False]]))
        assert m_all["abs_rel"] == 0.5
        assert m_masked["abs_rel"] == 1.0

    def test_empty_mask_returns_zeros(self):
        gt = np.zeros((4, 4))
        pred = np.ones((4, 4))
        m = compute_depth_metrics(pred, gt)
        assert all(v == 0.0 for v in m.values())


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------


class TestReconstructionMetrics:
    def test_identical_clouds(self, rng):
        pts = rng.uniform(-1, 1, (300, 3))
        m = compute_reconstruction_metrics(pts, pts)
        assert m["chamfer"] == 0.0
        assert m["accuracy"] == 0.0
        assert m["completeness"] == 0.0
        assert m["f_score"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0

    def test_uniform_shift(self, rng):
        pts = np.column_stack([np.arange(20, dtype=float), np.zeros(20), np.zeros(20)])
        shifted = pts + np.array([0.1, 0.0, 0.0])
        m = compute_reconstruction_metrics(shifted, pts)
        np.testing.assert_allclose(m["accuracy"], 0.1)
        np.testing.assert_allclose(m["completeness"], 0.1)
        np.testing.assert_allclose(m["chamfer"], 0.2)

    def test_half_coverage_recall(self, rng):
        """Prediction covers only half of the ground truth → recall = 0.5."""
        gt = np.vstack(
            [
                np.column_stack(
                    [np.zeros(10), np.arange(10, dtype=float), np.zeros(10)]
                ),
                np.column_stack(
                    [np.ones(10) * 5, np.arange(10, dtype=float), np.zeros(10)]
                ),
            ]
        )
        pred = gt[:10]  # only the first segment
        m = compute_reconstruction_metrics(pred, gt, f_score_threshold=0.5)
        assert m["precision"] == 1.0
        assert m["recall"] == 0.5
        np.testing.assert_allclose(m["f_score"], 2 * 1.0 * 0.5 / (1.0 + 0.5))
