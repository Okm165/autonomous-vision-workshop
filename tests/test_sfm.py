"""Tests for src/sfm.py — IncrementalSfM on synthetic multi-view data.

Also covers src/projective.py (homography DLT/RANSAC/warp, cross-ratio,
vanishing points, single-view metrology).
"""

import numpy as np
import pytest

from src.projective import (
    compute_homography_dlt,
    compute_homography_ransac,
    cross_ratio,
    find_vanishing_line,
    find_vanishing_point,
    from_homogeneous,
    hartley_normalize,
    measure_height_single_view,
    to_homogeneous,
    warp_image,
)
from src.sfm import IncrementalSfM
from src.transforms import rodrigues, se3_from_Rt, se3_inverse

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _rotation(axis, angle):
    return rodrigues(np.asarray(axis, dtype=np.float64), angle)


def _make_scene(rng, n_points=120):
    """Random 3-D points in a slab in front of the origin camera."""
    pts = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, n_points),
            rng.uniform(-1.5, 1.5, n_points),
            rng.uniform(4.0, 8.0, n_points),
        ]
    )
    return pts


def _camera_pose(rng, center, look_at=np.array([0.0, 0.0, 6.0])):
    """Simple camera pose: position *center*, z-axis toward *look_at*."""
    z = look_at - center
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.column_stack([x, y, z])
    return se3_from_Rt(R_wc, center)


def _project(X, T_w2c, K):
    Xc = (T_w2c[:3, :3] @ X.T).T + T_w2c[:3, 3]
    valid = Xc[:, 2] > 0.1
    uv = np.full((len(X), 2), np.nan)
    p = (K @ Xc[valid].T).T
    uv[valid] = p[:, :2] / p[:, 2:3]
    inside = (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < K[0, 2] * 2)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < K[1, 2] * 2)
    )
    return uv, inside


K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def three_view_scene():
    rng = np.random.default_rng(7)
    pts = _make_scene(rng)
    centers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.6, 0.0, 0.2]),
        np.array([-0.4, 0.3, 0.5]),
    ]
    T_w2c = []
    for c in centers:
        T_c2w = _camera_pose(rng, c)
        T_w2c.append(se3_inverse(T_c2w))
    uvs, visibles = zip(*[_project(pts, T, K) for T in T_w2c], strict=False)
    return pts, T_w2c, uvs, visibles


class TestIncrementalSfM:
    def test_two_view_initialization_recovers_pose_and_points(self, three_view_scene):
        pts, T_w2c, uvs, vis = three_view_scene
        sfm = IncrementalSfM(K)
        both = vis[0] & vis[1]
        sfm.add_image(0, uvs[0][vis[0]])
        sfm.add_image(1, uvs[1][vis[1]])

        # matches: map kp index in the "both" subset back to full indices
        idx0_full = np.where(vis[0])[0]
        idx1_full = np.where(vis[1])[0]
        pos0 = {int(i): k for k, i in enumerate(idx0_full)}
        pos1 = {int(i): k for k, i in enumerate(idx1_full)}
        matches = [(pos0[int(i)], pos1[int(i)]) for i in np.where(both)[0]]
        sfm.add_image_pair(0, 1, matches, uvs[0], uvs[1])

        assert sfm.initialize(0, 1) is True

        # camera 2 pose (camera-to-world).  E-decomposition fixes the gauge to
        # ||t|| = 1, so the reconstruction equals GT up to a global scale
        # lambda = 1 / ||t_gt||.
        T_est = sfm.cameras[1]
        T_gt = se3_inverse(T_w2c[1])
        lam = 1.0 / np.linalg.norm(T_gt[:3, 3])
        np.testing.assert_allclose(T_est[:3, :3], T_gt[:3, :3], atol=1e-6)
        np.testing.assert_allclose(T_est[:3, 3], T_gt[:3, 3] * lam, atol=1e-6)

        # triangulated points should match GT in the same gauge (noise-free);
        # the unit-baseline gauge inflates the scene by 1/||t_gt||
        pts_arr = np.array(sfm.points_3d)
        assert len(pts_arr) >= 0.9 * both.sum()
        d = np.linalg.norm(
            (pts_arr * np.linalg.norm(T_gt[:3, 3]))[:, None, :] - pts[None, :, :],
            axis=2,
        )
        nn = d.min(axis=1)
        assert np.median(nn) < 1e-3

    def test_three_view_registration_and_bundle_adjust(self, three_view_scene):
        pts, T_w2c, uvs, vis = three_view_scene
        sfm = IncrementalSfM(K)
        both01 = vis[0] & vis[1]
        both12 = vis[1] & vis[2]
        sfm.add_image(0, uvs[0])
        sfm.add_image(1, uvs[1])
        sfm.add_image(2, uvs[2])

        pos = [np.where(v)[0] for v in vis]
        lookup = [{int(i): k for k, i in enumerate(p)} for p in pos]

        m01 = [(lookup[0][int(i)], lookup[1][int(i)]) for i in np.where(both01)[0]]
        m12 = [(lookup[1][int(i)], lookup[2][int(i)]) for i in np.where(both12)[0]]
        sfm.add_image_pair(0, 1, m01, uvs[0], uvs[1])
        sfm.add_image_pair(1, 2, m12, uvs[1], uvs[2])

        assert sfm.initialize(0, 1) is True
        assert sfm.register_image(2) is True

        # the reconstruction lives in the unit-first-baseline gauge; compare
        # against GT after undoing the known scale lambda
        T_gt1 = se3_inverse(T_w2c[1])
        lam = 1.0 / np.linalg.norm(T_gt1[:3, 3])

        for cid in (0, 1, 2):
            T_est = sfm.cameras[cid]
            T_gt = se3_inverse(T_w2c[cid])
            t_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3] * lam)
            r_err = np.linalg.norm(T_est[:3, :3] - T_gt[:3, :3])
            assert t_err < 0.1, f"camera {cid} translation off by {t_err}"
            assert r_err < 0.02, f"camera {cid} rotation off by {r_err}"

        err = sfm.bundle_adjust(max_iterations=30)
        # noise-free data: BA should drive reprojection to ~0
        assert err < 0.5

        # after BA, all three camera poses must still match ground truth
        for cid in (0, 1, 2):
            T_est = sfm.cameras[cid]
            T_gt = se3_inverse(T_w2c[cid])
            t_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3] * lam)
            r_err = np.linalg.norm(T_est[:3, :3] - T_gt[:3, :3])
            assert t_err < 0.1, f"camera {cid} translation off by {t_err}"
            assert r_err < 0.02, f"camera {cid} rotation off by {r_err}"

        # reconstructed points close to GT after BA (same gauge)
        pts_arr = np.array(sfm.points_3d) * np.linalg.norm(T_gt1[:3, 3])
        d = np.linalg.norm(pts_arr[:, None, :] - pts[None, :, :], axis=2)
        assert np.median(d.min(axis=1)) < 0.05


# ======================================================================
#  src/projective.py
# ======================================================================


def _apply_h(H, pts):
    return from_homogeneous((H @ to_homogeneous(pts).T).T)


class TestHomographyDLT:
    def test_exact_recovery(self):
        rng = np.random.default_rng(3)
        H_true = np.array(
            [
                [1.2, 0.1, -30.0],
                [-0.05, 0.9, 12.0],
                [0.0003, 0.0002, 1.0],
            ]
        )
        src = rng.uniform(0, 200, (30, 2))
        dst = _apply_h(H_true, src)
        H = compute_homography_dlt(src, dst)
        np.testing.assert_allclose(H, H_true / H_true[2, 2], atol=1e-8)

    def test_rotation_scale_translation_case(self):
        # similarity H = s R + t must be recovered exactly
        R = rodrigues(np.array([0.0, 0.0, 1.0]), 0.4)
        H = np.eye(3)
        H[:2, :2] = 1.7 * R[:2, :2]
        H[:2, 2] = [15.0, -8.0]
        rng = np.random.default_rng(4)
        src = rng.uniform(-50, 50, (12, 2))
        dst = _apply_h(H, src)
        H_est = compute_homography_dlt(src, dst)
        np.testing.assert_allclose(H_est, H, atol=1e-8)

    def test_needs_four_points(self):
        with pytest.raises(ValueError, match="At least 4 point correspondences"):
            compute_homography_dlt(np.zeros((3, 2)), np.zeros((3, 2)))


class TestHomographyRANSAC:
    def test_rejects_outliers(self):
        rng = np.random.default_rng(11)
        H_true = np.array([[1.0, 0.05, 10.0], [-0.03, 0.95, -5.0], [2e-4, 1e-4, 1.0]])
        src = rng.uniform(0, 300, (80, 2))
        dst = _apply_h(H_true, src)
        dst_clean = dst.copy()
        n_out = 20
        dst[:n_out] = rng.uniform(0, 300, (n_out, 2))  # outliers

        H, mask = compute_homography_ransac(
            src, dst, thresh=3.0, max_iters=3000, rng=rng
        )
        assert mask[n_out:].sum() >= 55  # most true inliers found
        assert mask[:n_out].sum() == 0  # no outliers accepted
        # refit on inliers should be accurate
        err = np.linalg.norm(_apply_h(H, src[mask]) - dst_clean[mask], axis=1)
        assert np.mean(err) < 1e-3


class TestHartleyNormalize:
    def test_centroid_and_mean_distance(self):
        rng = np.random.default_rng(5)
        pts = rng.uniform(100, 400, (25, 2))
        norm, T = hartley_normalize(pts)
        np.testing.assert_allclose(norm.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            np.sqrt((norm**2).sum(axis=1)).mean(), np.sqrt(2.0), atol=1e-12
        )
        # T applied to homogeneous inputs reproduces `norm`
        thr = (T @ to_homogeneous(pts).T).T
        np.testing.assert_allclose(from_homogeneous(thr), norm, atol=1e-12)


class TestWarpImage:
    def test_identity(self):
        rng = np.random.default_rng(6)
        img = rng.integers(0, 255, (40, 50)).astype(np.uint8)
        out = warp_image(img, np.eye(3), (40, 50))
        np.testing.assert_array_equal(out, img)

    def test_translation(self):
        rng = np.random.default_rng(7)
        img = rng.integers(0, 255, (30, 40)).astype(np.uint8)
        H = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        out = warp_image(img, H, (30, 40))
        np.testing.assert_array_equal(out[1:, 2:], img[:-1, :-2])
        # shifted-in borders are black
        assert (out[0, :] == 0).all()
        assert (out[:, :2] == 0).all()

    def test_colour_image(self):
        rng = np.random.default_rng(8)
        img = rng.integers(0, 255, (20, 24, 3)).astype(np.uint8)
        out = warp_image(img, np.eye(3), (20, 24))
        np.testing.assert_array_equal(out, img)


class TestCrossRatio:
    def test_known_value(self):
        p = [np.array([0.0]), np.array([1.0]), np.array([3.0]), np.array([7.0])]
        cr = cross_ratio(*p)
        np.testing.assert_allclose(cr, 9.0 / 7.0, rtol=1e-12)

    def test_projective_invariance(self):
        # 1-D projective map x -> (a x + b) / (c x + d) must preserve CR
        a, b, c, d = 2.0, 1.0, 0.3, 1.0
        xs = [0.0, 1.0, 3.0, 7.0]
        mapped = [np.array([(a * x + b) / (c * x + d)]) for x in xs]
        cr_orig = cross_ratio(*[np.array([x]) for x in xs])
        cr_map = cross_ratio(*mapped)
        np.testing.assert_allclose(cr_map, cr_orig, rtol=1e-10)

    def test_2d_collinear(self):
        # points along a diagonal line, arbitrary spacing
        pts = [np.array([t, 2.0 * t + 1.0]) for t in (-2.0, 0.5, 4.0, 11.0)]
        cr = cross_ratio(*pts)
        xs = [p[0] for p in pts]
        cr_ref = cross_ratio(*[np.array([x]) for x in xs])
        np.testing.assert_allclose(cr, cr_ref, rtol=1e-10)


class TestVanishingPoints:
    def test_concurrent_lines(self):
        v = np.array([340.0, 210.0, 1.0])  # finite vanishing point
        rng = np.random.default_rng(9)
        lines = []
        for _ in range(8):
            q = np.append(rng.uniform(0, 100, 2), 1.0)
            lines.append(np.cross(v, q))
        vp = find_vanishing_point(np.array(lines))
        np.testing.assert_allclose(vp / vp[2], v, atol=1e-6)

    def test_parallel_lines_vp_at_infinity(self):
        # horizontal lines y = c  ->  l = (0, 1, -c); VP is (1, 0, 0)
        lines = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0], [0.0, 1.0, -3.0]])
        vp = find_vanishing_point(lines)
        np.testing.assert_allclose(np.abs(vp), np.array([1.0, 0.0, 0.0]), atol=1e-8)

    def test_vanishing_line_through_two_vps(self):
        v1 = np.array([500.0, 220.0, 1.0])
        v2 = np.array([-80.0, 260.0, 1.0])
        vl = find_vanishing_line(v1, v2)
        assert abs(vl @ v1) < 1e-6
        assert abs(vl @ v2) < 1e-6


class TestMeasureHeight:
    def test_criminisi_synthetic_view(self):
        """Full synthetic pinhole scene: reference pole 1.6 m, query pole 2.4 m."""
        f, W, H = 500.0, 640.0, 480.0
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        pitch = np.deg2rad(12.0)  # camera pitched slightly down
        Rx = rodrigues(np.array([1.0, 0.0, 0.0]), pitch)
        # camera centre at 1.5 m height, world y up, camera looks along +z
        C = np.array([0.0, 1.5, 0.0])
        R_wc = Rx  # cam x right, z forward (tilted down), y up-ish
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = R_wc.T
        T_w2c[:3, 3] = -R_wc.T @ C

        def project(X):
            Xc = T_w2c[:3, :3] @ X + T_w2c[:3, 3]
            p = K @ Xc
            return p[:2] / p[2]

        # vertical vanishing point: image of the direction (0, 1, 0)
        d_cam = R_wc.T @ np.array([0.0, 1.0, 0.0])
        vp_vert_h = K @ d_cam
        # two in-plane directions -> ground-plane vanishing line
        d1 = R_wc.T @ np.array([1.0, 0.0, 0.0])
        d2 = R_wc.T @ np.array([0.0, 0.0, 1.0])
        v1, v2 = K @ d1, K @ d2
        vl = np.cross(v1, v2)
        vl /= np.linalg.norm(vl[:2])

        depth = 6.0
        H_ref, H_query = 1.6, 2.4
        ref_base = project(np.array([1.2, 0.0, depth]))
        ref_top = project(np.array([1.2, H_ref, depth]))
        q_base = project(np.array([-0.8, 0.0, depth]))
        q_top = project(np.array([-0.8, H_query, depth]))

        h = measure_height_single_view(
            q_base, q_top, H_ref, ref_base, ref_top, vl, vp_vert_h
        )
        np.testing.assert_allclose(h, H_query, rtol=1e-6)

    def test_criminisi_randomized(self):
        """Machine-precision accuracy across randomised perspective scenes.

        Random cameras (varying pitch/roll/yaw/intrinsics/height) viewing
        two vertical poles at *different depths* — the configuration that
        requires the full 4-factor Criminisi formula.
        """
        rng = np.random.default_rng(42)
        for _ in range(30):
            f = rng.uniform(400, 900)
            W_img, H_img = 1280.0, 960.0
            K = np.array(
                [
                    [f, 0, W_img / 2 + rng.uniform(-40, 40)],
                    [0, rng.uniform(400, 900), H_img / 2 + rng.uniform(-30, 30)],
                    [0, 0, 1.0],
                ]
            )
            pitch = np.deg2rad(rng.uniform(3, 28))
            roll = np.deg2rad(rng.uniform(-4, 4))
            yaw = np.deg2rad(rng.uniform(-10, 10))
            R_wc = (
                rodrigues(np.array([1.0, 0.0, 0.0]), pitch)
                @ rodrigues(np.array([0.0, 0.0, 1.0]), yaw)
                @ rodrigues(np.array([0.0, 1.0, 0.0]), roll)
            )
            C = np.array([rng.uniform(-5, 5), rng.uniform(1.0, 3.0), rng.uniform(0, 5)])
            T_w2c = np.eye(4)
            T_w2c[:3, :3] = R_wc.T
            T_w2c[:3, 3] = -R_wc.T @ C

            def project(X, T=T_w2c, K=K):
                Xc = T[:3, :3] @ X + T[:3, 3]
                p = K @ Xc
                return p[:2] / p[2]

            # vanishing line of the ground plane y=0 and vertical VP
            n_c = T_w2c[:3, :3] @ np.array([0.0, 1.0, 0.0])
            vl = np.linalg.inv(K).T @ n_c
            vp_vert = K @ n_c

            H_ref = rng.uniform(1.0, 3.0)
            H_query = rng.uniform(0.5, 15.0)
            up = np.array([0.0, 1.0, 0.0])
            foot_r = np.array([rng.uniform(-6, 6), 0.0, rng.uniform(4, 20)])
            foot_q = np.array([rng.uniform(-6, 6), 0.0, rng.uniform(4, 20)])

            h = measure_height_single_view(
                project(foot_q),
                project(foot_q + H_query * up),
                H_ref,
                project(foot_r),
                project(foot_r + H_ref * up),
                vl,
                vp_vert,
            )
            np.testing.assert_allclose(h, H_query, rtol=1e-8)
