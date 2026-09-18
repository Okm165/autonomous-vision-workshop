"""Camera models, calibration, and multi-view geometry.

This module provides mathematically rigorous implementations of:
- Pinhole camera projection / back-projection with lens distortion
- Camera calibration (DLT + non-linear refinement)
- Perspective-n-Point pose estimation
- Fundamental / Essential matrix computation and decomposition
- Linear and midpoint triangulation

Coordinate conventions
----------------------
- **World frame**: right-handed, arbitrary origin.
- **Camera frame**: X right, Y down, Z forward (OpenCV convention).
- **Image frame**: u right (column), v down (row), origin at top-left.

References
----------
[1] Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., 2004.
[2] Zhang, "A Flexible New Technique for Camera Calibration", TPAMI 2000.
[3] Nistér, "An Efficient Solution to the Five-Point Relative Pose Problem", TPAMI 2004.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Self

import cv2
import numpy as np

from ._cv import solve_pnp_ransac as _solve_pnp_ransac

#: Generic numerical tolerance for rank/denominator tests.
_EPS = 1e-12

# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------# Brown-Conrady distortion helpers (module-level, shared by undistortion)
# ---------------------------------------------------------------------------
def _distort_forward(
    x: np.ndarray,
    y: np.ndarray,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the forward Brown-Conrady model to normalised coordinates."""
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def _forward_residual(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual of the forward model at a candidate undistorted guess."""
    fx, fy = _distort_forward(x, y, k1, k2, p1, p2, k3)
    return fx - xd, fy - yd


def _newton_invert(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
    x0: np.ndarray,
    y0: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Damped Newton on the 2-D forward model with an analytic Jacobian.

    Each point is solved independently (vectorised); a backtracking
    half-stepping guards against divergence, and the best iterate seen so
    far is kept, so the result is never worse than the initial guess.
    """
    x, y = x0.copy(), y0.copy()
    fx, fy = _forward_residual(x, y, xd, yd, k1, k2, p1, p2, k3)
    best_norm = np.maximum(np.abs(fx), np.abs(fy))
    best_x, best_y = x.copy(), y.copy()

    for _ in range(max_iter):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        drad = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2**2  # d(rho)/d(r2)

        # Jacobian of F(x, y) = (x*rho + t1, y*rho + t2)
        j00 = radial + 2.0 * x * x * drad + 2.0 * p1 * y + 6.0 * p2 * x
        j01 = 2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y
        j11 = radial + 2.0 * y * y * drad + 6.0 * p1 * y + 2.0 * p2 * x

        res_x, res_y = _forward_residual(x, y, xd, yd, k1, k2, p1, p2, k3)

        det = j00 * j11 - j01 * j01
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        dx = -(j11 * res_x - j01 * res_y) / det
        dy = -(j00 * res_y - j01 * res_x) / det

        # backtracking: halve the step while the residual norm grows
        step = np.ones_like(x)
        # initialisers only satisfy the type checker; range(30) always runs
        xn, yn = x, y
        norm = best_norm
        improved = np.ones_like(x, dtype=bool)
        for _bt in range(30):
            xn, yn = x + step * dx, y + step * dy
            fnx, fny = _forward_residual(xn, yn, xd, yd, k1, k2, p1, p2, k3)
            norm = np.maximum(np.abs(fnx), np.abs(fny))
            improved = norm < best_norm
            if improved.all():
                break
            step = np.where(improved, step, step * 0.5)

        x, y = xn, yn
        best_norm = np.where(improved, norm, best_norm)
        best_x = np.where(improved, xn, best_x)
        best_y = np.where(improved, yn, best_y)
        if (best_norm < 1e3 * 1e-12).all():
            break

    return best_x, best_y


class CameraIntrinsics:
    r"""Pinhole camera model with optional radial-tangential distortion.

    The intrinsic (calibration) matrix is

    .. math::

        K = \begin{bmatrix}
            f_x & 0   & c_x \\
            0   & f_y & c_y \\
            0   & 0   & 1
        \end{bmatrix}

    where :math:`(f_x, f_y)` are the focal lengths in pixel units and
    :math:`(c_x, c_y)` is the principal point.

    Distortion follows the OpenCV Brown–Conrady model with up to five
    coefficients :math:`(k_1, k_2, p_1, p_2, k_3)`.

    Parameters
    ----------
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal-point coordinates in pixels.
    dist_coeffs : array-like or None
        Distortion coefficients ``(k1, k2, p1, p2[, k3])``.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    _K: np.ndarray

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_coeffs: np.ndarray | None = None,
    ) -> None:
        """Initialise intrinsics from focal lengths, principal point, and optional distortion."""
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        if dist_coeffs is not None:
            dc = np.asarray(dist_coeffs, dtype=np.float64).ravel()
            if dc.size < 5:
                dc = np.pad(dc, (0, 5 - dc.size))
            self.dist_coeffs: np.ndarray | None = dc[:5]
        else:
            self.dist_coeffs = None

        self._K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self._K_inv: np.ndarray = np.linalg.inv(self._K)

    # -- properties ---------------------------------------------------------

    @property
    def K(self) -> np.ndarray:
        """Return the 3x3 intrinsic matrix (read-only copy)."""
        return self._K.copy()

    @property
    def K_inv(self) -> np.ndarray:
        """Return the inverse of the intrinsic matrix."""
        return self._K_inv.copy()

    # -- projection ---------------------------------------------------------

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        r"""Project 3-D camera-frame points onto the image plane.

        Pinhole model
        ~~~~~~~~~~~~~
        Given a 3-D point :math:`\mathbf{P} = (X, Y, Z)^T` expressed in the
        camera frame, the ideal (undistorted) projection is:

        .. math::

            \begin{aligned}
            x_n &= X / Z, \\
            y_n &= Y / Z, \\
            u   &= f_x \, x_n + c_x, \\
            v   &= f_y \, y_n + c_y.
            \end{aligned}

        In matrix form: :math:`\lambda\,\tilde{\mathbf{p}} = K\,\mathbf{P}`
        with :math:`\lambda = Z`.

        Distortion model (Brown–Conrady)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        After computing normalised coordinates :math:`(x_n, y_n)`:

        .. math::

            r^2 &= x_n^2 + y_n^2 \\
            x_d &= x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
                  + 2 p_1 x_n y_n + p_2 (r^2 + 2 x_n^2) \\
            y_d &= y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
                  + p_1 (r^2 + 2 y_n^2) + 2 p_2 x_n y_n

        Then the pixel coordinates are
        :math:`u = f_x x_d + c_x,\; v = f_y y_d + c_y`.

        Parameters
        ----------
        points_3d : ndarray, shape (N, 3)
            Points in camera coordinates.

        Returns
        -------
        points_2d : ndarray, shape (N, 2)
            Projected pixel coordinates.  Points with :math:`Z \le 0` are
            mapped to ``(NaN, NaN)``.
        """
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        assert pts.shape[1] == 3, "Expected (N, 3) array"

        Z = pts[:, 2]
        valid = Z > 0.0

        xn = np.full(len(pts), np.nan)
        yn = np.full(len(pts), np.nan)
        xn[valid] = pts[valid, 0] / Z[valid]
        yn[valid] = pts[valid, 1] / Z[valid]

        if self.dist_coeffs is not None:
            k1, k2, p1, p2, k3 = self.dist_coeffs
            r2 = xn**2 + yn**2
            radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn**2)
            yd = yn * radial + p1 * (r2 + 2.0 * yn**2) + 2.0 * p2 * xn * yn
        else:
            xd, yd = xn, yn

        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy

        return np.column_stack([u, v])

    # -- back-projection ----------------------------------------------------

    def unproject(self, points_2d: np.ndarray, depth: np.ndarray) -> np.ndarray:
        r"""Back-project pixel coordinates to 3-D camera-frame points.

        .. math::

            \mathbf{P} = Z \, K^{-1}
            \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}
            = \begin{pmatrix}
                (u - c_x)\,Z / f_x \\
                (v - c_y)\,Z / f_y \\
                Z
            \end{pmatrix}

        Parameters
        ----------
        points_2d : ndarray, shape (N, 2)
            Pixel coordinates ``(u, v)``.
        depth : ndarray, shape (N,)
            Depth values :math:`Z > 0` for each point.

        Returns
        -------
        points_3d : ndarray, shape (N, 3)
        """
        pts = np.asarray(points_2d, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        Z = np.asarray(depth, dtype=np.float64).ravel()

        ones = np.ones(len(pts))
        hom = np.column_stack([pts, ones])  # (N, 3)
        rays = (self._K_inv @ hom.T).T  # (N, 3)
        return rays * Z[:, np.newaxis]

    backproject: Callable[[Self, np.ndarray, np.ndarray], np.ndarray] = unproject

    # -- factory methods ----------------------------------------------------

    @classmethod
    def from_fov(
        cls,
        fov_deg: float,
        image_size: tuple[int, int],
        *,
        dist_coeffs: np.ndarray | None = None,
    ) -> CameraIntrinsics:
        r"""Create intrinsics from horizontal field-of-view angle.

        .. math::

            f_x = \frac{W}{2\,\tan(\text{FoV}/2)}

        Parameters
        ----------
        fov_deg : horizontal field of view in degrees
        image_size : (width, height) in pixels
        dist_coeffs : optional distortion coefficients

        Returns
        -------
        CameraIntrinsics
        """
        W, H = image_size
        fx = W / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
        fy = fx
        cx = W / 2.0
        cy = H / 2.0
        return cls(fx, fy, cx, cy, dist_coeffs)

    # -- undistortion -------------------------------------------------------

    def undistort_points(
        self,
        points_2d: np.ndarray,
        *,
        max_iter: int = 50,
        tol: float = 1e-12,
    ) -> np.ndarray:
        r"""Iteratively remove lens distortion from pixel coordinates.

        Given distorted pixel coordinates, first normalise them to
        :math:`(x_d, y_d)` and then seek the undistorted :math:`(x_n, y_n)`
        such that the forward Brown-Conrady model maps
        :math:`(x_n, y_n) \mapsto (x_d, y_d)`.

        The forward model is

        .. math::

            x_d = x_n\,\rho(r^2) + 2p_1 x_n y_n + p_2(r^2 + 2x_n^2), \\
            y_d = y_n\,\rho(r^2) + p_1(r^2 + 2y_n^2) + 2p_2 x_n y_n,

        with :math:`r^2 = x_n^2 + y_n^2` and
        :math:`\rho(r^2) = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6`.

        The inverse is solved with OpenCV's classic **undistortion iteration**
        (a damped fixed-point scheme).  Each round estimates the tangential
        displacement at the current guess and removes it before dividing by
        the radial factor:

        .. math::

            \mathbf{d}_{\text{tang}}
              = \begin{pmatrix} 2p_1 x_n y_n + p_2(r^2 + 2x_n^2) \\
                                 p_1(r^2 + 2y_n^2) + 2p_2 x_n y_n\end{pmatrix},
            \qquad
            \mathbf{x}_{i+1} = \frac{\mathbf{x}_d - \mathbf{d}_{\text{tang}}(\mathbf{x}_i)}
                                       {\rho(r_i^2)}.

        The map being inverted is a radial scaling of the plane, so the
        iteration is a contraction whenever
        :math:`|\mathrm{d}(r\,\rho)/\mathrm{d}r| < 1` on the path between the
        initial guess and the fixed point — exactly the region where the
        polynomial model is physically meaningful (a real single-valued
        lens).  Convergence is linear with rate roughly
        :math:`|\mathrm{d}(r\rho)/\mathrm{d}r|`, giving ~10–40 iterations for
        typical lenses; 50 iterations drive it to machine precision for
        :math:`k_1 \approx -0.35` and below
        :math:`10^{-5}\,\mathrm{px}` even for extreme
        :math:`k_1 \approx -0.75` wide-angle lenses.

        Parameters
        ----------
        points_2d : ndarray, shape (N, 2)
            Distorted pixel coordinates.
        max_iter : int
            Maximum number of fixed-point iterations.
        tol : float
            Convergence threshold on the max normalised-coordinate change.

        Returns
        -------
        undistorted : ndarray, shape (N, 2)
            Undistorted **pixel** coordinates.  If no distortion coefficients
            are set the input is returned unchanged.
        """
        pts = np.asarray(points_2d, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]

        if self.dist_coeffs is None:
            return pts.copy()

        k1, k2, p1, p2, k3 = self.dist_coeffs

        xd = (pts[:, 0] - self.cx) / self.fx
        yd = (pts[:, 1] - self.cy) / self.fy

        # Initial guess: the distorted coordinates themselves (the fixed-point
        # map is a contraction from there for any physically valid lens).
        x = xd.copy()
        y = yd.copy()

        for _ in range(max_iter):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            tang_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            tang_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

            x_new = (xd - tang_x) / radial
            y_new = (yd - tang_y) / radial

            delta = np.max(np.abs(x_new - x)) + np.max(np.abs(y_new - y))
            x, y = x_new, y_new
            if delta < tol:
                break

        # --- Verify + Newton fallback -------------------------------------
        # The fixed-point scheme is a contraction only inside the lens'
        # physical validity region.  Points far outside it (e.g. synthetic
        # coordinates well beyond the sensor) can oscillate; OpenCV's own
        # iteration fails on the same inputs.  For those, refine with a
        # damped Newton on the full forward model, which converges to the
        # unique preimage whenever the radial map is monotonic.
        res_x, res_y = _forward_residual(x, y, xd, yd, k1, k2, p1, p2, k3)
        bad = np.maximum(np.abs(res_x), np.abs(res_y)) > 1e3 * tol
        if bad.any():
            x[bad], y[bad] = _newton_invert(
                xd[bad], yd[bad], k1, k2, p1, p2, k3, x[bad], y[bad], max_iter
            )

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.column_stack([u, v])

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image using the stored distortion coefficients."""
        if self.dist_coeffs is None:
            return image
        h, w = image.shape[:2]
        new_K, _roi = cv2.getOptimalNewCameraMatrix(
            self._K,
            self.dist_coeffs,
            (w, h),
            1,
            (w, h),
        )
        return cv2.undistort(image, self._K, self.dist_coeffs, None, new_K)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_camera(
    object_points_list: Sequence[np.ndarray],
    image_points_list: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    r"""Calibrate a camera from checkerboard correspondences.

    This is a thin wrapper around :pyfunc:`cv2.calibrateCamera` that returns
    NumPy arrays directly.  The underlying algorithm proceeds in two stages,
    summarised below.

    Stage 1 — Direct Linear Transform (DLT)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Each 3-D ↔ 2-D correspondence
    :math:`\mathbf{X}_i \leftrightarrow \mathbf{x}_i` gives two rows of the
    *measurement matrix* :math:`\mathbf{A}`:

    .. math::

        \begin{bmatrix}
        \mathbf{0}^T & -w_i\,\tilde{\mathbf{X}}_i^T
            & y_i\,\tilde{\mathbf{X}}_i^T \\
        w_i\,\tilde{\mathbf{X}}_i^T & \mathbf{0}^T
            & -x_i\,\tilde{\mathbf{X}}_i^T
        \end{bmatrix}

    where :math:`\tilde{\mathbf{X}}_i = (X,Y,Z,1)^T` and
    :math:`\tilde{\mathbf{x}}_i = (x_i, y_i, w_i)^T`.

    The :math:`2n \times 12` system :math:`\mathbf{A}\mathbf{p} = \mathbf{0}`
    is solved by taking the right singular vector of :math:`\mathbf{A}`
    corresponding to its smallest singular value (SVD).  The 12-vector
    :math:`\mathbf{p}` is reshaped into the :math:`3 \times 4` projection
    matrix :math:`P = K[R \mid \mathbf{t}]`.

    :math:`K` and :math:`R` are recovered via RQ decomposition of the left
    :math:`3 \times 3` block of :math:`P`.

    Stage 2 — Non-linear refinement
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The DLT solution is refined by minimising the total reprojection error

    .. math::

        \sum_{i,j} \lVert \mathbf{x}_{ij}
            - \hat\pi(K, k, R_i, \mathbf{t}_i, \mathbf{X}_j) \rVert^2

    over all intrinsic parameters :math:`(f_x, f_y, c_x, c_y, k_1 \ldots)`
    and per-image extrinsics :math:`(R_i, \mathbf{t}_i)` using
    Levenberg–Marquardt.

    Parameters
    ----------
    object_points_list : sequence of ndarray, each (M, 3)
        3-D checkerboard corner coordinates for each calibration image
        (typically :math:`Z=0`).
    image_points_list : sequence of ndarray, each (M, 2)
        Detected 2-D corners for each image.
    image_size : (width, height)
        Image resolution in pixels.

    Returns
    -------
    K : ndarray, shape (3, 3)
        Intrinsic matrix.
    dist_coeffs : ndarray, shape (5,)
        Distortion coefficients ``(k1, k2, p1, p2, k3)``.
    rvecs : list of ndarray, each (3, 1)
        Rodrigues rotation vectors, one per image.
    tvecs : list of ndarray, each (3, 1)
        Translation vectors, one per image.
    """
    obj_pts = [np.asarray(o, dtype=np.float32) for o in object_points_list]
    img_pts = [np.asarray(i, dtype=np.float32) for i in image_points_list]

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts,
        img_pts,
        image_size,
        None,
        None,
    )
    if not ret:
        raise RuntimeError("cv2.calibrateCamera failed")

    return (
        K.astype(np.float64),
        dist.ravel()[:5].astype(np.float64),
        [r.astype(np.float64) for r in rvecs],
        [t.astype(np.float64) for t in tvecs],
    )


calibrate_from_checkerboard = calibrate_camera


# ---------------------------------------------------------------------------
# PnP
# ---------------------------------------------------------------------------


def solve_pnp(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> tuple[bool, np.ndarray, np.ndarray]:
    r"""Estimate camera pose from 3-D ↔ 2-D correspondences (PnP).

    Perspective-n-Point (PnP) recovers the rotation :math:`R` and translation
    :math:`\mathbf{t}` that relate a set of 3-D world points to their 2-D
    projections via:

    .. math::

        \lambda_i\,\tilde{\mathbf{x}}_i
            = K\,[R \mid \mathbf{t}]\,\tilde{\mathbf{X}}_i.

    Minimal case — P3P (3 points)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    With exactly 3 correspondences, the problem reduces to finding the
    distances :math:`d_1, d_2, d_3` from the camera centre to the 3-D
    points.  Applying the cosine rule in the triangle formed by each pair of
    back-projected rays yields a system of three equations in three unknowns:

    .. math::

        d_i^2 + d_j^2 - 2\,d_i\,d_j\,\cos\theta_{ij}
            = \lVert \mathbf{X}_i - \mathbf{X}_j \rVert^2

    This quartic system admits up to 4 real solutions; a 4th point is used to
    disambiguate.

    Parameters
    ----------
    points_3d : ndarray, shape (N, 3)
    points_2d : ndarray, shape (N, 2)
    K : ndarray, shape (3, 3)
    dist_coeffs : ndarray or None

    Returns
    -------
    success : bool
    rvec : ndarray, shape (3, 1)
        Rodrigues rotation vector.
    tvec : ndarray, shape (3, 1)
    """
    obj = np.ascontiguousarray(points_3d, dtype=np.float64)
    img = np.ascontiguousarray(points_2d, dtype=np.float64)
    dc = (
        np.zeros(5, dtype=np.float64)
        if dist_coeffs is None
        else np.asarray(
            dist_coeffs,
            dtype=np.float64,
        ).ravel()
    )

    ok, rvec, tvec = cv2.solvePnP(obj, img, K.astype(np.float64), dc)
    return bool(ok), rvec, tvec


def solve_pnp_ransac(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    reproj_thresh: float = 3.0,
    iterations: int = 1000,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray | None]:
    r"""RANSAC-robust PnP pose estimation.

    Repeatedly samples minimal subsets (4 points), solves PnP, and scores
    each hypothesis by counting inliers whose reprojection error is below
    *reproj_thresh*.  The largest-consensus model is refined on all inliers.

    Parameters
    ----------
    points_3d : ndarray, shape (N, 3)
    points_2d : ndarray, shape (N, 2)
    K : ndarray, shape (3, 3)
    dist_coeffs : ndarray or None
    reproj_thresh : float
        Inlier threshold in pixels.
    iterations : int
        RANSAC iterations.

    Returns
    -------
    success : bool
    rvec : ndarray, shape (3, 1)
    tvec : ndarray, shape (3, 1)
    inliers : ndarray of int or None
        Indices of inlier correspondences.
    """
    obj = np.ascontiguousarray(points_3d, dtype=np.float64)
    img = np.ascontiguousarray(points_2d, dtype=np.float64)
    dc = (
        np.zeros(5, dtype=np.float64)
        if dist_coeffs is None
        else np.asarray(
            dist_coeffs,
            dtype=np.float64,
        ).ravel()
    )

    ok, rvec, tvec, inliers = _solve_pnp_ransac(
        obj,
        img,
        np.asarray(K, dtype=np.float64),
        dc,
        reprojection_error=reproj_thresh,
        iterations=iterations,
    )
    return bool(ok), rvec, tvec, inliers


# ---------------------------------------------------------------------------
# Projection matrix composition / decomposition
# ---------------------------------------------------------------------------


def compute_P(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    r"""Build the 3×4 projection matrix :math:`P = K\,[R \mid \mathbf{t}]`.

    .. math::

        P = K \begin{bmatrix} R & \mathbf{t} \end{bmatrix}
          \in \mathbb{R}^{3 \times 4}

    Parameters
    ----------
    K : ndarray, shape (3, 3)
    R : ndarray, shape (3, 3)
    t : ndarray, shape (3,) or (3, 1)

    Returns
    -------
    P : ndarray, shape (3, 4)
    """
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    Rt = np.hstack([R, t])
    return K @ Rt


def decompose_P(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Decompose a projection matrix into intrinsic and extrinsic parts.

    Given :math:`P = K\,[R \mid \mathbf{t}]`, we recover :math:`K`, :math:`R`,
    :math:`\mathbf{t}` using RQ decomposition on the left 3×3 sub-matrix
    :math:`M = KR`:

    1. Reverse the rows and columns of :math:`M` to get :math:`M'`.
    2. QR-factorise :math:`M' = Q'R'`.
    3. Then :math:`K = {R'}^T` reversed, :math:`R = {Q'}^T` reversed.
    4. Enforce positive diagonal in :math:`K` by flipping signs as needed.
    5. :math:`\mathbf{t} = K^{-1} P_{:,3}`.

    This is the classical RQ decomposition approach of Hartley & Zisserman
    (Algorithm 4.1 in [1]).

    Parameters
    ----------
    P : ndarray, shape (3, 4)

    Returns
    -------
    K : ndarray, shape (3, 3)
        Upper-triangular intrinsic matrix with positive diagonal.
    R : ndarray, shape (3, 3)
        Rotation matrix (:math:`\det R = +1`).
    t : ndarray, shape (3, 1)
        Translation vector.
    """
    P = np.asarray(P, dtype=np.float64)
    M = P[:, :3]

    J = np.flipud(np.eye(3))

    # RQ via reversed QR.  Writing J for the exchange matrix (J² = I),
    #     J M J = (J K J)(J R J),
    # where J K J is *lower* triangular.  Transposing turns this into an
    # ordinary QR factorisation,
    #     (J M J)ᵀ = (J Rᵀ J)(J Kᵀ J),
    # whose upper-triangular right factor J Kᵀ J yields K and whose
    # orthogonal left factor J Rᵀ J yields R, both after one more flip.
    M_flip = J @ M @ J
    Q_prime, R_prime = np.linalg.qr(M_flip.T)

    K = J @ R_prime.T @ J
    R = J @ Q_prime.T @ J

    # Fix the sign ambiguity column-wise: require diag(K) > 0, compensating
    # in R so that K @ R is unchanged.
    D = np.diag(np.sign(np.diag(K)) + (np.diag(K) == 0.0))
    K = K @ D
    R = D @ R

    # Normalise so K[2,2] = 1
    K = K / K[2, 2]

    # Ensure det(R) = +1 (R is already orthogonal to machine precision)
    if np.linalg.det(R) < 0:
        R = -R
        K = -K

    t = np.linalg.solve(K, P[:, 3:4])

    return K, R, t


# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------


def reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    r"""Per-point reprojection error.

    For each pair :math:`(\mathbf{X}_i, \mathbf{x}_i)`:

    .. math::

        e_i = \lVert \mathbf{x}_i - \pi(K, R, \mathbf{t}, \mathbf{X}_i)
              \rVert_2

    where

    .. math::

        \pi(\cdot) = \text{dehomog}\!\bigl(K\,[R \mid \mathbf{t}]\,
            \tilde{\mathbf{X}}_i\bigr).

    Parameters
    ----------
    points_3d : ndarray, shape (N, 3)
    points_2d : ndarray, shape (N, 2)
    K : ndarray, shape (3, 3)
    R : ndarray, shape (3, 3)
    t : ndarray, shape (3,) or (3, 1)

    Returns
    -------
    errors : ndarray, shape (N,)
        Euclidean reprojection error for each correspondence.
    """
    pts3 = np.asarray(points_3d, dtype=np.float64)
    pts2 = np.asarray(points_2d, dtype=np.float64)
    P = compute_P(K, R, t)

    hom = np.hstack([pts3, np.ones((len(pts3), 1))])  # (N, 4)
    proj = (P @ hom.T).T  # (N, 3)
    proj_2d = proj[:, :2] / proj[:, 2:3]

    return np.linalg.norm(pts2 - proj_2d, axis=1)


# ---------------------------------------------------------------------------
# Fundamental & Essential matrices
# ---------------------------------------------------------------------------


def _hartley_normalise(
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Hartley normalisation (isotropic scaling).

    Translate so centroid is at the origin and scale so the average distance
    from the origin is :math:`\sqrt{2}`:

    .. math::

        T = \begin{bmatrix}
            s & 0 & -s\,\bar u \\
            0 & s & -s\,\bar v \\
            0 & 0 & 1
        \end{bmatrix},
        \qquad
        s = \frac{\sqrt{2}}{\bar d}

    where :math:`\bar d` is the mean distance to centroid
    :math:`(\bar u, \bar v)`.

    Returns (normalised_pts_homogeneous, T).
    """
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    if mean_dist < 1e-12:
        mean_dist = 1e-12
    s = np.sqrt(2.0) / mean_dist
    T = np.array(
        [
            [s, 0.0, -s * centroid[0]],
            [0.0, s, -s * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    hom = np.column_stack([pts, np.ones(len(pts))])
    normed = (T @ hom.T).T
    return normed[:, :2], T


def compute_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    r"""Compute the fundamental matrix using the normalised 8-point algorithm.

    Epipolar constraint
    ~~~~~~~~~~~~~~~~~~~
    For every pair of corresponding points
    :math:`\mathbf{x} \leftrightarrow \mathbf{x}'`:

    .. math::

        {\mathbf{x}'}^T\,F\,\mathbf{x} = 0

    Expanded with :math:`\mathbf{x} = (x, y, 1)^T`,
    :math:`\mathbf{x}' = (x', y', 1)^T`:

    .. math::

        x' x\,f_{11} + x' y\,f_{12} + x'\,f_{13}
        + y' x\,f_{21} + y' y\,f_{22} + y'\,f_{23}
        + x\,f_{31} + y\,f_{32} + f_{33} = 0

    Algorithm
    ~~~~~~~~~
    1. **Normalise** both point sets with Hartley transforms :math:`T_1, T_2`.
    2. Build the :math:`N \times 9` constraint matrix :math:`A` where each row
       is :math:`[x'x,\; x'y,\; x',\; y'x,\; y'y,\; y',\; x,\; y,\; 1]`.
    3. **SVD** of :math:`A = U \Sigma V^T`.  The last column of :math:`V`
       (corresponding to the smallest singular value) gives :math:`\mathbf{f}`;
       reshape to :math:`3 \times 3` to get :math:`\hat F`.
    4. **Rank-2 enforcement**: SVD of :math:`\hat F = U_F \Sigma_F V_F^T`,
       zero the smallest singular value, reconstruct
       :math:`F' = U_F\,\text{diag}(\sigma_1, \sigma_2, 0)\,V_F^T`.
    5. **Denormalise**: :math:`F = T_2^T F' T_1`.

    Parameters
    ----------
    pts1, pts2 : ndarray, shape (N, 2)
        Corresponding points in images 1 and 2.  :math:`N \ge 8`.

    Returns
    -------
    F : ndarray, shape (3, 3)
        Fundamental matrix satisfying
        :math:`\mathbf{x}_2^T F \mathbf{x}_1 = 0`.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    assert len(pts1) >= 8, "At least 8 point correspondences required"

    n1, T1 = _hartley_normalise(pts1)
    n2, T2 = _hartley_normalise(pts2)

    x1, y1 = n1[:, 0], n1[:, 1]
    x2, y2 = n2[:, 0], n2[:, 1]

    A = np.column_stack(
        [
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            np.ones(len(pts1)),
        ]
    )

    _, _, Vt = np.linalg.svd(A)
    F_hat = Vt[-1].reshape(3, 3)

    # Enforce rank 2
    Uf, Sf, Vft = np.linalg.svd(F_hat)
    Sf[2] = 0.0
    F_hat = Uf @ np.diag(Sf) @ Vft

    F = T2.T @ F_hat @ T1
    F = F / F[2, 2] if abs(F[2, 2]) > 1e-12 else F / np.linalg.norm(F)
    return F


def compute_essential_matrix(
    F: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
) -> np.ndarray:
    r"""Compute the essential matrix from the fundamental matrix.

    .. math::

        E = K_2^T \, F \, K_1

    The essential matrix encodes the same epipolar geometry as :math:`F` but
    in normalised (calibrated) coordinates.  It has the special structure
    :math:`E = [\mathbf{t}]_\times R` and exactly two equal non-zero singular
    values.

    Parameters
    ----------
    F : ndarray, shape (3, 3)
    K1 : ndarray, shape (3, 3) — intrinsics of camera 1.
    K2 : ndarray, shape (3, 3) — intrinsics of camera 2.

    Returns
    -------
    E : ndarray, shape (3, 3)
    """
    E = K2.T @ F @ K1
    # Re-project onto the essential space: two equal singular values
    U, S, Vt = np.linalg.svd(E)
    s = (S[0] + S[1]) / 2.0
    E = U @ np.diag([s, s, 0.0]) @ Vt
    return E


def decompose_essential_matrix(
    E: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    r"""Extract the four possible :math:`(R, \mathbf{t})` from :math:`E`.

    Decomposition (Hartley & Zisserman §9.6.2)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Compute SVD :math:`E = U \operatorname{diag}(\sigma, \sigma, 0) V^T`.
    Define the orthogonal matrix

    .. math::

        W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1
            \end{bmatrix}.

    The two candidate rotations and two candidate translations are:

    .. math::

        R_1 = U\,W\,V^T, \qquad R_2 = U\,W^T\,V^T, \qquad
        \mathbf{t} = \pm U_{:,2}.

    If :math:`\det(R) = -1`, negate both :math:`R` and :math:`\mathbf{t}` so
    that :math:`R \in SO(3)`.

    Returns
    -------
    solutions : list of 4 (R, t) tuples
        Each :math:`R` is 3×3, each :math:`\mathbf{t}` is shape (3,).
    """
    U, _, Vt = np.linalg.svd(E)

    # Ensure proper rotations
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t),
    ]
    return solutions


def choose_pose_by_cheirality(
    R_list: list[np.ndarray],
    t_list: list[np.ndarray],
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Select the correct :math:`(R, \mathbf{t})` via the cheirality check.

    For each candidate pose, triangulate all correspondences and count how
    many reconstructed points have **positive depth** in *both* cameras:

    .. math::

        Z_1 = \mathbf{P}_{1,\text{row3}} \cdot \tilde{\mathbf{X}} > 0,
        \qquad
        Z_2 = \mathbf{P}_{2,\text{row3}} \cdot \tilde{\mathbf{X}} > 0.

    The physically correct solution is the one that places the scene in front
    of both cameras (maximum number of points with positive depth).

    Parameters
    ----------
    R_list : list of ndarray, shape (3, 3)
    t_list : list of ndarray, shape (3,)
    pts1, pts2 : ndarray, shape (N, 2)
    K : ndarray, shape (3, 3)

    Returns
    -------
    R : ndarray, shape (3, 3)
    t : ndarray, shape (3,)
    """
    P1 = compute_P(K, np.eye(3), np.zeros(3))

    best_count = -1
    best_R, best_t = R_list[0], t_list[0]

    for R, t_vec in zip(R_list, t_list, strict=False):
        P2 = compute_P(K, R, t_vec)
        X = triangulate_points(pts1, pts2, P1, P2)

        hom = np.hstack([X, np.ones((len(X), 1))])
        depth1 = (P1 @ hom.T)[2]
        depth2 = (P2 @ hom.T)[2]
        n_positive = int(np.sum((depth1 > 0) & (depth2 > 0)))

        if n_positive > best_count:
            best_count = n_positive
            best_R, best_t = R, t_vec

    return best_R, best_t


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    r"""Triangulate 3-D points using linear DLT.

    For each correspondence :math:`\mathbf{x}_1 \leftrightarrow \mathbf{x}_2`
    we form the linear system :math:`A\,\mathbf{X} = \mathbf{0}` where

    .. math::

        A = \begin{bmatrix}
            x_1 \mathbf{p}_1^{3T} - \mathbf{p}_1^{1T} \\
            y_1 \mathbf{p}_1^{3T} - \mathbf{p}_1^{2T} \\
            x_2 \mathbf{p}_2^{3T} - \mathbf{p}_2^{1T} \\
            y_2 \mathbf{p}_2^{3T} - \mathbf{p}_2^{2T}
        \end{bmatrix}
        \in \mathbb{R}^{4 \times 4}

    and :math:`\mathbf{p}_k^{iT}` is the :math:`i`-th row of :math:`P_k`.

    The solution is the right singular vector of :math:`A` corresponding to
    its smallest singular value.  The homogeneous 4-vector is converted to
    Euclidean coordinates by dividing by its last component.

    Parameters
    ----------
    pts1, pts2 : ndarray, shape (N, 2)
    P1, P2 : ndarray, shape (3, 4)

    Returns
    -------
    points_3d : ndarray, shape (N, 3)
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)

    N = len(pts1)
    X = np.empty((N, 3), dtype=np.float64)

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array(
            [
                x1 * P1[2] - P1[0],
                y1 * P1[2] - P1[1],
                x2 * P2[2] - P2[0],
                y2 * P2[2] - P2[1],
            ]
        )

        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        X[i] = Xh[:3] / Xh[3]

    return X


def _backproject_to_ray(pt_2d: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """Back-project a 2-D point to a unit-norm ray direction in camera frame."""
    h = np.array([pt_2d[0], pt_2d[1], 1.0])
    d = K_inv @ h
    return d / np.linalg.norm(d)


def _ray_midpoint(
    o1: np.ndarray,
    d1: np.ndarray,
    o2: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray:
    r"""Compute the midpoint of closest approach between two 3-D rays.

    Minimises :math:`\|(\mathbf{o}_1 + s\mathbf{d}_1) -
    (\mathbf{o}_2 + t\mathbf{d}_2)\|^2` over :math:`(s, t)`.  Setting the
    gradient to zero gives the 2×2 normal equations

    .. math::

        \begin{bmatrix} a & -b \\ b & -c \end{bmatrix}
        \begin{pmatrix} s \\ t \end{pmatrix}
        = \begin{pmatrix} -d \\ e \end{pmatrix},
        \qquad
        \begin{aligned}
          a &= \mathbf{d}_1\cdot\mathbf{d}_1, \\
          b &= \mathbf{d}_1\cdot\mathbf{d}_2, \\
          c &= \mathbf{d}_2\cdot\mathbf{d}_2, \\
          d &= \mathbf{d}_1\cdot\mathbf{w}, \\
          e &= \mathbf{d}_2\cdot\mathbf{w},
        \end{aligned}

    with :math:`\mathbf{w} = \mathbf{o}_2 - \mathbf{o}_1`.  Solving by
    Cramer's rule (determinant :math:`ac - b^2`) yields

    .. math::

        s = \frac{c\,d - b\,e}{ac - b^2}, \qquad
        t = \frac{b\,d - a\,e}{ac - b^2}.
    """
    w = o2 - o1
    a = d1 @ d1
    b = d1 @ d2
    c = d2 @ d2
    d = d1 @ w
    e = d2 @ w
    denom = a * c - b * b

    if abs(denom) < _EPS:
        return o1 + d1 * (d / a) if a > _EPS else o1.copy()

    s = (c * d - b * e) / denom
    t_param = (b * d - a * e) / denom

    closest1 = o1 + s * d1
    closest2 = o2 + t_param * d2
    return 0.5 * (closest1 + closest2)


def triangulate_midpoint(
    pts1: np.ndarray,
    pts2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    r"""Triangulate using the midpoint method.

    Given camera 1 at the origin and camera 2 at :math:`(R, \mathbf{t})`,
    back-project each 2-D point into a ray:

    .. math::

        \mathbf{r}_1 = K^{-1}\,\tilde{\mathbf{x}}_1, \qquad
        \mathbf{r}_2 = R^T\,(K^{-1}\,\tilde{\mathbf{x}}_2 - \mathbf{t}).

    The closest point between the two (generally skew) rays

    .. math::

        \mathbf{o}_1 + s\,\mathbf{d}_1
        \qquad\text{and}\qquad
        \mathbf{o}_2 + t\,\mathbf{d}_2

    is found by solving:

    .. math::

        \begin{bmatrix}
            \mathbf{d}_1 \cdot \mathbf{d}_1 &
           -\mathbf{d}_1 \cdot \mathbf{d}_2 \\
            \mathbf{d}_2 \cdot \mathbf{d}_1 &
           -\mathbf{d}_2 \cdot \mathbf{d}_2
        \end{bmatrix}
        \begin{pmatrix} s \\ t \end{pmatrix}
        =
        \begin{pmatrix}
            (\mathbf{o}_2 - \mathbf{o}_1) \cdot \mathbf{d}_1 \\
            (\mathbf{o}_2 - \mathbf{o}_1) \cdot \mathbf{d}_2
        \end{pmatrix}

    The 3-D point is the midpoint of the segment connecting the two closest
    points on each ray.

    Parameters
    ----------
    pts1, pts2 : ndarray, shape (N, 2)
    R : ndarray, shape (3, 3)
    t : ndarray, shape (3,) or (3, 1)
    K : ndarray, shape (3, 3)

    Returns
    -------
    points_3d : ndarray, shape (N, 3)
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    K_inv = np.linalg.inv(K)

    o1 = np.zeros(3)
    o2 = -R.T @ t  # camera-2 centre in camera-1 frame

    N = len(pts1)
    X = np.empty((N, 3), dtype=np.float64)

    for i in range(N):
        d1 = _backproject_to_ray(pts1[i], K_inv)
        d2 = R.T @ _backproject_to_ray(pts2[i], K_inv)

        X[i] = _ray_midpoint(o1, d1, o2, d2)

    return X
