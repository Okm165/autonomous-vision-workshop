"""
Optical Flow, Point Tracking & Scene Flow
==========================================

Implementations of classical optical-flow algorithms and their 3-D
extensions for a vision & 3-D mapping workshop.

Contents
--------
* **Lucas–Kanade sparse flow** — from-scratch normal-equations solver.
* **Pyramidal Lucas–Kanade** — coarse-to-fine for large displacements.
* **Dense flow (Farnebäck)** — OpenCV wrapper with sensible defaults.
* **Flow visualisation** — Middlebury HSV colour convention.
* **Flow-to-depth** — epipolar flow decomposition for known ego-motion.
* **3-D scene flow** — back-project + differencing with optional
  ego-motion subtraction.
* **Flow warping** — bilinear backward-warp of an image by a flow field.

Coordinate conventions
----------------------
* Image frame: *u* right (column), *v* down (row), origin top-left.
* Flow vectors ``(dx, dy)`` follow the image-frame convention.
* Depth maps store metric depth *Z* (distance along camera *Z*-axis).

References
----------
[1] Lucas & Kanade, "An Iterative Image Registration Technique with an
    Application to Stereo Vision", IJCAI 1981.
[2] Bouguet, "Pyramidal Implementation of the Lucas Kanade Feature
    Tracker — Description of the algorithm", Intel Corp., 2001.
[3] Farnebäck, "Two-Frame Motion Estimation Based on Polynomial
    Expansion", SCIA 2003.
[4] Vedula, Baker, Rander, Collins, Kanade, "Three-Dimensional Scene
    Flow", TPAMI 2005.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_gray_float(img: np.ndarray) -> np.ndarray:
    """Convert *img* to ``float64`` grayscale in [0, 1]."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def _image_gradients(
    img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Central-difference spatial gradients.

    .. math::

        I_x(x, y) \approx \frac{I(x+1, y) - I(x-1, y)}{2}

    Returns ``(Ix, Iy)`` each with the same shape as *img*.
    """
    kx = np.array([[-1, 0, 1]], dtype=np.float64) / 2.0
    ky = kx.T
    Ix = cv2.filter2D(img, cv2.CV_64F, kx)
    Iy = cv2.filter2D(img, cv2.CV_64F, ky)
    return Ix, Iy


def _build_gaussian_pyramid(
    img: np.ndarray, levels: int
) -> list[np.ndarray]:
    """Build a Gaussian pyramid with *levels* entries (index 0 = original)."""
    pyramid: list[np.ndarray] = [img]
    for _ in range(1, levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


# ---------------------------------------------------------------------------
# 1. Lucas–Kanade sparse optical flow (from scratch)
# ---------------------------------------------------------------------------


def _compute_gradient_windows(
    Ix: np.ndarray, Iy: np.ndarray, ix: int, iy: int, half: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract gradient sub-windows centred on pixel (ix, iy)."""
    Ix_w = Ix[iy - half : iy + half + 1, ix - half : ix + half + 1]
    Iy_w = Iy[iy - half : iy + half + 1, ix - half : ix + half + 1]
    return Ix_w, Iy_w


def _build_structure_tensor(
    Ix_w: np.ndarray, Iy_w: np.ndarray,
) -> np.ndarray:
    """Build the 2x2 structure tensor (Harris matrix) from gradient windows."""
    sum_IxIx = np.sum(Ix_w * Ix_w)
    sum_IxIy = np.sum(Ix_w * Iy_w)
    sum_IyIy = np.sum(Iy_w * Iy_w)
    return np.array([[sum_IxIx, sum_IxIy],
                     [sum_IxIy, sum_IyIy]], dtype=np.float64)


def _lk_iterative_refine(
    prev: np.ndarray,
    curr: np.ndarray,
    Ix_w: np.ndarray,
    Iy_w: np.ndarray,
    ATA_inv: np.ndarray,
    px: float,
    py: float,
    half: int,
    max_iterations: int,
    epsilon: float,
) -> Tuple[float, float, bool]:
    """Run iterative refinement for one tracked point, returning (dx, dy, ok)."""
    H, W = prev.shape
    ix, iy = int(round(px)), int(round(py))
    dx_total, dy_total = 0.0, 0.0
    ok = True
    for _ in range(max_iterations):
        cx = px + dx_total
        cy = py + dy_total
        cix, ciy = int(round(cx)), int(round(cy))

        if (cix - half < 0 or cix + half >= W
                or ciy - half < 0 or ciy + half >= H):
            ok = False
            break

        curr_patch = curr[ciy - half : ciy + half + 1,
                          cix - half : cix + half + 1]
        prev_patch = prev[iy - half : iy + half + 1,
                          ix - half : ix + half + 1]
        It = curr_patch - prev_patch

        b = np.array([-np.sum(Ix_w * It),
                      -np.sum(Iy_w * It)], dtype=np.float64)

        dd = ATA_inv @ b
        dx_total += dd[0]
        dy_total += dd[1]

        if np.linalg.norm(dd) < epsilon:
            break

    return dx_total, dy_total, ok


def _track_single_point(
    prev: np.ndarray,
    curr: np.ndarray,
    Ix: np.ndarray,
    Iy: np.ndarray,
    px: float,
    py: float,
    half: int,
    H: int,
    W: int,
    min_eigenvalue: float,
    max_iterations: int,
    epsilon: float,
) -> Tuple[np.ndarray, bool]:
    """Track one point from *prev* to *curr* via Lucas-Kanade, returning (new_xy, ok)."""
    ix, iy = int(round(px)), int(round(py))

    if ix - half < 0 or ix + half >= W or iy - half < 0 or iy + half >= H:
        return np.array([px, py]), False

    Ix_w, Iy_w = _compute_gradient_windows(Ix, Iy, ix, iy, half)
    ATA = _build_structure_tensor(Ix_w, Iy_w)

    eigvals = np.linalg.eigvalsh(ATA)
    if eigvals[0] < min_eigenvalue:
        return np.array([px, py]), False

    ATA_inv = np.linalg.inv(ATA)

    dx_total, dy_total, ok = _lk_iterative_refine(
        prev, curr, Ix_w, Iy_w, ATA_inv,
        px, py, half, max_iterations, epsilon,
    )

    new_xy = np.array([px + dx_total, py + dy_total])
    if new_xy[0] < 0 or new_xy[0] >= W or new_xy[1] < 0 or new_xy[1] >= H:
        ok = False

    return new_xy, ok


def lucas_kanade(
    prev: np.ndarray,
    curr: np.ndarray,
    points: np.ndarray,
    window_size: int = 15,
    min_eigenvalue: float = 1e-4,
    max_iterations: int = 10,
    epsilon: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Lucas–Kanade optical flow for sparse points.

    **Brightness-constancy assumption**

    .. math::

        I(x, y, t) = I(x + \delta x,\; y + \delta y,\; t + \delta t)

    A first-order Taylor expansion yields the *optical-flow constraint
    equation* (OFCE):

    .. math::

        I_x u + I_y v + I_t = 0

    where :math:`I_x, I_y` are spatial gradients, :math:`I_t` is the
    temporal gradient, and :math:`\mathbf{d} = (u, v)^T` is the
    displacement we seek.  One equation, two unknowns — this is the
    **aperture problem**.

    **Lucas–Kanade solution** — assume constant flow inside a window
    :math:`W` of size ``window_size × window_size``.  For every pixel
    :math:`(x_i, y_i) \in W` write one instance of the OFCE, giving an
    over-determined system :math:`A \mathbf{d} = \mathbf{b}`:

    .. math::

        A = \begin{pmatrix}
                I_x(x_1,y_1) & I_y(x_1,y_1) \\
                \vdots        & \vdots        \\
                I_x(x_n,y_n) & I_y(x_n,y_n)
            \end{pmatrix},
        \quad
        \mathbf{b} = \begin{pmatrix}
                          -I_t(x_1,y_1) \\
                          \vdots          \\
                          -I_t(x_n,y_n)
                      \end{pmatrix}

    Least-squares via the **normal equations**:

    .. math::

        A^T A \;\mathbf{d} = A^T \mathbf{b}
        \;\;\Longrightarrow\;\;
        \begin{pmatrix}
            \sum I_x^2    & \sum I_x I_y \\
            \sum I_x I_y  & \sum I_y^2
        \end{pmatrix}
        \begin{pmatrix} u \\ v \end{pmatrix}
        =
        \begin{pmatrix}
            -\sum I_x I_t \\
            -\sum I_y I_t
        \end{pmatrix}

    :math:`A^T A` is the **structure tensor** (Harris matrix) — its
    eigenvalues must both be large for a unique solution, which is exactly
    the Harris corner criterion.  Hence "good features to track" coincide
    with good corners.

    An **iterative refinement** loop warps the search window by the
    current estimate after each step, stopping when the update
    :math:`\|\Delta\mathbf{d}\| < \varepsilon` or *max_iterations* is
    reached.

    Parameters
    ----------
    prev, curr : (H, W) grayscale float images.
    points : (N, 2) array of ``(x, y)`` positions to track.
    window_size : Side length of the square tracking window (must be odd).
    min_eigenvalue : Reject a point when :math:`\lambda_{\min}(A^T A)`
        falls below this value (poorly conditioned).
    max_iterations : Newton-style refinement iterations per point.
    epsilon : Convergence threshold on the displacement update norm.

    Returns
    -------
    new_points : (N, 2) updated positions in *curr*.
    status : (N,) boolean array — ``True`` where tracking succeeded.
    """
    prev = _ensure_gray_float(prev)
    curr = _ensure_gray_float(curr)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 2)

    H, W = prev.shape
    half = window_size // 2
    Ix, Iy = _image_gradients(prev)

    N = len(points)
    new_points = np.copy(points)
    status = np.ones(N, dtype=bool)

    for i in range(N):
        new_points[i], status[i] = _track_single_point(
            prev, curr, Ix, Iy,
            points[i, 0], points[i, 1],
            half, H, W,
            min_eigenvalue, max_iterations, epsilon,
        )

    return new_points, status


# ---------------------------------------------------------------------------
# 2. Pyramidal Lucas–Kanade
# ---------------------------------------------------------------------------

def pyramidal_lk(
    prev: np.ndarray,
    curr: np.ndarray,
    points: np.ndarray,
    levels: int = 3,
    window_size: int = 15,
    max_iterations: int = 10,
    epsilon: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Pyramidal Lucas–Kanade for handling large displacements.

    Plain LK cannot cope with motions larger than about half the window
    size because the brightness-constancy linearisation breaks down.  The
    **coarse-to-fine** (pyramidal) strategy solves this:

    1. Build Gaussian pyramids :math:`\{I^L\}_{L=0}^{L_m}` for both
       frames (level 0 = original resolution).
    2. At the **coarsest** level :math:`L_m`, run LK with initial guess
       :math:`\mathbf{g}^{L_m} = \mathbf{0}`.
    3. At each finer level :math:`L`:

       a. Up-scale the previous estimate:
          :math:`\mathbf{g}^L = 2\,\mathbf{d}^{L+1}`.
       b. Run LK centred at the guessed position to obtain a residual
          :math:`\delta\mathbf{d}^L`.
       c. Set :math:`\mathbf{d}^L = \mathbf{g}^L + \delta\mathbf{d}^L`.

    The effective search range grows by a factor of :math:`2^{L_m}`, so
    even motions of tens of pixels are resolved.

    Parameters
    ----------
    prev, curr : (H, W) grayscale float images.
    points : (N, 2) array of ``(x, y)`` positions in *prev*.
    levels : Number of pyramid levels (including the original).
    window_size : Tracking-window side length at each level.
    max_iterations : Refinement iterations per level.
    epsilon : Convergence threshold per level.

    Returns
    -------
    new_points : (N, 2) tracked positions in *curr*.
    status : (N,) boolean — ``True`` where tracking succeeded at every
        level.

    See Also
    --------
    lucas_kanade : Single-scale Lucas–Kanade solver.

    References
    ----------
    [2] Bouguet, "Pyramidal Implementation of the Lucas Kanade Feature
        Tracker", Intel Corp., 2001.
    """
    prev = _ensure_gray_float(prev)
    curr = _ensure_gray_float(curr)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    pyr_prev = _build_gaussian_pyramid(prev, levels)
    pyr_curr = _build_gaussian_pyramid(curr, levels)

    N = len(points)
    # Scale points to the coarsest level
    scale = 2.0 ** (levels - 1)
    scaled_pts = points / scale

    cumulative_flow = np.zeros((N, 2), dtype=np.float64)
    status = np.ones(N, dtype=bool)

    for lvl in range(levels - 1, -1, -1):
        guess_pts = scaled_pts + cumulative_flow

        lvl_new, lvl_status = lucas_kanade(
            pyr_prev[lvl],
            pyr_curr[lvl],
            guess_pts,
            window_size=window_size,
            max_iterations=max_iterations,
            epsilon=epsilon,
        )

        residual = lvl_new - guess_pts
        cumulative_flow += residual
        status &= lvl_status

        if lvl > 0:
            # Move to the next finer level: double coords and flow
            scaled_pts *= 2.0
            cumulative_flow *= 2.0

    new_points = scaled_pts + cumulative_flow
    return new_points, status


# ---------------------------------------------------------------------------
# 3. Dense optical flow — Farnebäck wrapper
# ---------------------------------------------------------------------------

def dense_flow_farneback(
    prev: np.ndarray,
    curr: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
) -> np.ndarray:
    r"""Dense optical flow via Farnebäck's polynomial-expansion method.

    Each pixel neighbourhood is approximated by a quadratic polynomial:

    .. math::

        I(x) \approx x^T A\,x + b^T x + c

    where :math:`A` is a 2 × 2 matrix, :math:`b` a 2-vector, and
    :math:`c` a scalar, all estimated by a weighted least-squares fit
    inside a window of size *poly_n*.

    Under a global displacement model the displacement :math:`d` between
    two frames satisfies:

    .. math::

        A\,d = -\tfrac{1}{2}(b_2 - b_1), \quad A = \tfrac{1}{2}(A_1 + A_2)

    This is solved iteratively within an image-pyramid loop for robustness
    to large motions.

    Parameters
    ----------
    prev, curr : (H, W) grayscale images (uint8 or float).
    pyr_scale : Pyramid down-scale ratio (< 1).
    levels : Pyramid levels.
    winsize : Averaging window for polynomial expansion.
    iterations : Solver iterations per pyramid level.
    poly_n : Pixel neighbourhood size for polynomial approximation
        (5 or 7 recommended).
    poly_sigma : Gaussian std-dev for the polynomial-fit window
        (1.1 for *poly_n* = 5, 1.5 for 7).

    Returns
    -------
    flow : (H, W, 2) float32 — ``flow[..., 0]`` is *dx*,
        ``flow[..., 1]`` is *dy*.

    References
    ----------
    [3] Farnebäck, "Two-Frame Motion Estimation Based on Polynomial
        Expansion", SCIA 2003.
    """
    prev_g = _ensure_gray_float(prev)
    curr_g = _ensure_gray_float(curr)
    prev_u8 = (prev_g * 255).clip(0, 255).astype(np.uint8)
    curr_u8 = (curr_g * 255).clip(0, 255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        prev_u8,
        curr_u8,
        flow=None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=0,
    )
    return flow


# ---------------------------------------------------------------------------
# 4. Flow visualisation (Middlebury colour convention)
# ---------------------------------------------------------------------------

def flow_to_color(
    flow: np.ndarray,
    max_flow: Optional[float] = None,
) -> np.ndarray:
    r"""Convert a dense optical-flow field to an HSV colour image.

    **Middlebury colour convention**

    * **Hue** encodes flow direction:

      .. math::

          H = \operatorname{atan2}(dy,\; dx) \;\bmod\; 2\pi

      mapped to the ``[0, 180)`` range that OpenCV's 8-bit HSV uses.

    * **Saturation** is fixed at maximum (255).

    * **Value** encodes normalised flow magnitude:

      .. math::

          V = 255 \;\cdot\; \min\!\Bigl(\frac{\sqrt{dx^2 + dy^2}}
              {f_{\max}},\; 1\Bigr)

      where :math:`f_{\max}` is *max_flow* (auto-detected if ``None``).

    Parameters
    ----------
    flow : (H, W, 2) float array with ``flow[..., 0]`` = *dx*,
        ``flow[..., 1]`` = *dy*.
    max_flow : Clamp magnitude for normalisation.  ``None`` → use the
        maximum magnitude present in *flow*.

    Returns
    -------
    rgb : (H, W, 3) ``uint8`` BGR image suitable for ``cv2.imshow``.
    """
    dx = flow[..., 0]
    dy = flow[..., 1]

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)  # radians in [-π, π]

    if max_flow is None:
        max_flow = max(magnitude.max(), 1e-8)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    # OpenCV HSV: H in [0,180), S and V in [0,255]
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(magnitude / max_flow, 0.0, 1.0) * 255).astype(
        np.uint8
    )

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


# ---------------------------------------------------------------------------
# 5. Flow → depth (epipolar flow decomposition)
# ---------------------------------------------------------------------------


def _build_homogeneous_pixel_grid(
    H: int, W: int,
) -> np.ndarray:
    """Construct an (H, W, 3) grid of homogeneous pixel coordinates."""
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    return np.stack([us, vs, np.ones_like(us)], axis=-1)


def _compute_rotation_only_flow(
    K: np.ndarray,
    K_inv: np.ndarray,
    R: np.ndarray,
    uv1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation-only projected positions and flow (depth-independent).

    Returns (flow_rot, rotated_rays) where *rotated_rays* = R K^{-1} u
    is reused for the translation step.
    """
    rays = np.einsum("ij,...j->...i", K_inv, uv1)
    rotated = np.einsum("ij,...j->...i", R, rays)
    proj_rot = np.einsum("ij,...j->...i", K, rotated)
    z_proj = proj_rot[..., 2:3].clip(min=1e-12)
    uv_rot = proj_rot[..., :2] / z_proj
    flow_rot = uv_rot - uv1[..., :2]
    return flow_rot, rotated


def _compute_unit_translation_flow(
    K: np.ndarray,
    rotated: np.ndarray,
    t: np.ndarray,
    flow_rot: np.ndarray,
    uv1: np.ndarray,
) -> np.ndarray:
    """Translational flow component at unit depth (Z = 1)."""
    trans_3d = rotated + t[np.newaxis, np.newaxis, :]
    proj_full = np.einsum("ij,...j->...i", K, trans_3d)
    z_full = proj_full[..., 2:3].clip(min=1e-12)
    uv_full = proj_full[..., :2] / z_full
    uv_rot = flow_rot + uv1[..., :2]
    return uv_full - uv_rot


def _recover_depth_from_flow_decomposition(
    flow: np.ndarray,
    flow_rot: np.ndarray,
    flow_trans_unit: np.ndarray,
    min_flow_mag: float,
) -> np.ndarray:
    """Solve Z = ||f_trans(Z=1)|| / ||f_observed - f_rot||."""
    residual = flow[..., :2] - flow_rot
    mag_residual = np.sqrt(np.sum(residual ** 2, axis=-1)).clip(min=1e-12)
    mag_trans_unit = np.sqrt(np.sum(flow_trans_unit ** 2, axis=-1))
    return np.where(mag_trans_unit > min_flow_mag,
                    mag_trans_unit / mag_residual, 0.0)


def flow_to_depth(
    flow: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    min_flow_mag: float = 1e-3,
) -> np.ndarray:
    r"""Recover per-pixel depth from optical flow of a rigid/static scene.

    For a calibrated camera undergoing rotation :math:`R` and translation
    :math:`\mathbf{t}`, the *flow* of a static 3-D point at depth
    :math:`Z` decomposes into a rotational and a translational part.

    **Derivation** — the 3-D point :math:`\mathbf{P} = Z\,K^{-1}
    \tilde{\mathbf{u}}` (back-projected homogeneous pixel) is seen in
    frame 2 at

    .. math::

        \tilde{\mathbf{u}}_2
        \sim K\bigl(R\,K^{-1}\tilde{\mathbf{u}}_1
                     + \tfrac{1}{Z}\,\mathbf{t}\bigr)

    Splitting flow :math:`\mathbf{f} = \mathbf{u}_2 - \mathbf{u}_1`:

    .. math::

        \mathbf{f} = \underbrace{\mathbf{f}_{\text{rot}}}_{\text{rotation-only}}
                    + \underbrace{\frac{1}{Z}\;\mathbf{f}_{\text{trans}}}
                                  _{\text{translation-only, scaled by } 1/Z}

    Rotational flow depends only on :math:`R` and is depth-independent;
    translational flow is inversely proportional to depth.  Therefore:

    .. math::

        Z = \frac{\|\mathbf{f}_{\text{trans}}\|}
                 {\|\mathbf{f} - \mathbf{f}_{\text{rot}}\|}

    Because :math:`\mathbf{f}_{\text{trans}}` still contains the unknown
    :math:`Z`, we instead compute it for a *unit depth* reference and
    recover the actual depth from the ratio.

    Parameters
    ----------
    flow : (H, W, 2) measured optical flow ``(dx, dy)``.
    K : (3, 3) camera intrinsic matrix.
    R : (3, 3) rotation from frame 1 to frame 2.
    t : (3,) or (3, 1) translation from frame 1 to frame 2 (in camera-1
        coordinates).
    min_flow_mag : Pixels with translational flow magnitude below this
        threshold receive ``depth = 0`` (degenerate).

    Returns
    -------
    depth : (H, W) estimated depth map.  Pixels where the translational
        flow is negligible (e.g. along the epipole direction) are set
        to 0.
    """
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()

    H, W = flow.shape[:2]
    K_inv = np.linalg.inv(K)

    uv1 = _build_homogeneous_pixel_grid(H, W)
    flow_rot, rotated = _compute_rotation_only_flow(K, K_inv, R, uv1)
    flow_trans_unit = _compute_unit_translation_flow(K, rotated, t, flow_rot, uv1)

    return _recover_depth_from_flow_decomposition(
        flow, flow_rot, flow_trans_unit, min_flow_mag,
    )


# ---------------------------------------------------------------------------
# 6. 3-D scene flow
# ---------------------------------------------------------------------------

def _backproject(
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    r"""Back-project a depth map to a (H, W, 3) point cloud.

    .. math::

        \mathbf{P} = Z \; K^{-1} \tilde{\mathbf{u}}
    """
    H, W = depth.shape[:2]
    K_inv = np.linalg.inv(K)

    us, vs = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    ones = np.ones_like(us)
    uv1 = np.stack([us, vs, ones], axis=-1)  # (H, W, 3)

    rays = np.einsum("ij,...j->...i", K_inv, uv1)
    return rays * depth[..., np.newaxis]


def compute_scene_flow(
    flow_2d: np.ndarray,
    depth1: np.ndarray,
    depth2: np.ndarray,
    K: np.ndarray,
    T_12: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Compute 3-D scene flow from 2-D optical flow and paired depth maps.

    **Method** (back-project, warp, difference):

    1. Back-project every pixel in frame 1 to 3-D using *depth1*:

       .. math::

           \mathbf{P}_1 = Z_1 \; K^{-1} \tilde{\mathbf{u}}_1

    2. Warp each pixel's image coordinates by the 2-D optical flow to
       obtain its frame-2 position
       :math:`\mathbf{u}_2 = \mathbf{u}_1 + \mathbf{f}(\mathbf{u}_1)`.

    3. Sample the frame-2 depth at :math:`\mathbf{u}_2` (bilinear) and
       back-project:

       .. math::

           \mathbf{P}_2 = Z_2(\mathbf{u}_2) \; K^{-1} \tilde{\mathbf{u}}_2

    4. The **scene flow** at each pixel is

       .. math::

           \mathbf{S} = \mathbf{P}_2 - \mathbf{P}_1

       If the camera ego-motion :math:`T_{12}` is known, we can subtract
       it to isolate *object* motion:

       .. math::

           \mathbf{S}_{\text{obj}}
           = \mathbf{P}_2 - T_{12}\,\mathbf{P}_1

    Parameters
    ----------
    flow_2d : (H, W, 2) optical flow ``(dx, dy)`` from frame 1 → 2.
    depth1 : (H, W) depth map for frame 1 (metric depth).
    depth2 : (H, W) depth map for frame 2 (metric depth).
    K : (3, 3) camera intrinsic matrix.
    T_12 : (4, 4) rigid transform from frame 1 to frame 2 (optional).
        When provided, ego-motion is subtracted so the returned flow
        reflects only dynamic-object motion.

    Returns
    -------
    scene_flow : (H, W, 3) 3-D flow vectors :math:`(dX, dY, dZ)` per
        pixel.

    References
    ----------
    [4] Vedula *et al.*, "Three-Dimensional Scene Flow", TPAMI 2005.
    """
    K = np.asarray(K, dtype=np.float64)
    H, W = depth1.shape[:2]

    # Step 1 — back-project frame 1
    P1 = _backproject(depth1, K)  # (H, W, 3)

    # Step 2 — warped pixel coordinates in frame 2
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    u2 = us + flow_2d[..., 0]
    v2 = vs + flow_2d[..., 1]

    # Step 3 — bilinear-sample depth2 at (u2, v2)
    map_x = u2.astype(np.float32)
    map_y = v2.astype(np.float32)
    depth2_warped = cv2.remap(
        depth2.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float64)

    # Back-project frame 2
    K_inv = np.linalg.inv(K)
    ones = np.ones_like(u2)
    uv2_h = np.stack([u2, v2, ones], axis=-1)
    rays2 = np.einsum("ij,...j->...i", K_inv, uv2_h)
    P2 = rays2 * depth2_warped[..., np.newaxis]

    # Step 4 — scene flow
    if T_12 is not None:
        T_12 = np.asarray(T_12, dtype=np.float64)
        R = T_12[:3, :3]
        tvec = T_12[:3, 3]
        P1_in_2 = np.einsum("ij,...j->...i", R, P1) + tvec
        scene_flow = P2 - P1_in_2
    else:
        scene_flow = P2 - P1

    return scene_flow


# ---------------------------------------------------------------------------
# 7. Flow warping (bilinear backward warp)
# ---------------------------------------------------------------------------

def warp_image_flow(
    image: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    r"""Warp *image* according to a dense optical-flow field.

    Uses **backward warping**: for every output pixel
    :math:`\mathbf{u}` we look up the source position

    .. math::

        \mathbf{u}_s = \mathbf{u} + \mathbf{f}(\mathbf{u})

    and bilinearly interpolate the input image at :math:`\mathbf{u}_s`.
    This avoids holes and is equivalent to synthesising frame 2 from
    frame 1 given the flow from 1 → 2, or vice-versa given the
    backward flow.

    Parameters
    ----------
    image : (H, W) or (H, W, C) input image.
    flow : (H, W, 2) flow field — ``flow[..., 0]`` is *dx*,
        ``flow[..., 1]`` is *dy*.

    Returns
    -------
    warped : Same shape and dtype as *image*.  Pixels that map outside
        the image domain are replicated from the nearest border pixel.
    """
    H, W = flow.shape[:2]
    us, vs = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    map_x = (us + flow[..., 0]).astype(np.float32)
    map_y = (vs + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


# ======================================================================
#  RAFT wrapper (GPU optional)
# ======================================================================

def neural_flow_raft(
    prev: np.ndarray,
    curr: np.ndarray,
    model_name: str = "raft_small",
) -> np.ndarray:
    r"""Compute optical flow using RAFT (torchvision).

    RAFT (Recurrent All-Pairs Field Transforms) uses:
    1. A feature encoder to extract per-pixel features from both frames
    2. A 4-D correlation volume between all feature pairs
    3. An iterative ConvGRU that indexes into the correlation volume to
       refine the flow estimate

    Falls back to Farneback if ``torch`` / ``torchvision`` are unavailable.

    Parameters
    ----------
    prev, curr : (H, W) or (H, W, 3) images (uint8 or float)
    model_name : ``"raft_small"`` or ``"raft_large"``

    Returns
    -------
    flow : (H, W, 2) optical flow field
    """
    try:
        import torch
        import torchvision.models.optical_flow as tof
        from torchvision.transforms.functional import to_tensor
    except ImportError:
        import warnings
        warnings.warn("torch/torchvision not available — falling back to Farneback.")
        return dense_flow_farneback(prev, curr)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "raft_large":
        weights = tof.Raft_Large_Weights.DEFAULT
        model = tof.raft_large(weights=weights).to(device).eval()
    else:
        weights = tof.Raft_Small_Weights.DEFAULT
        model = tof.raft_small(weights=weights).to(device).eval()

    transforms = weights.transforms()

    def _prep(img):
        """Convert an image to a batched float tensor for RAFT inference."""
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        t = to_tensor(img).unsqueeze(0).to(device)
        return t

    t1 = _prep(prev)
    t2 = _prep(curr)
    t1, t2 = transforms(t1, t2)

    with torch.no_grad():
        flows = model(t1, t2)
        flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()

    return flow


# Aliases matching plan naming convention
sparse_flow_lk = pyramidal_lk
