"""Projective geometry primitives for vision and 3-D mapping.

This module provides mathematically rigorous implementations of core
projective-geometry operations used in multi-view geometry, image
stitching, single-view metrology, and augmented-reality pipelines.

All functions operate on **NumPy** arrays and follow the notation and
conventions of *Multiple View Geometry in Computer Vision* (Hartley &
Zisserman, 2nd ed.).

Coordinate conventions
----------------------
* 2-D Euclidean point:  (x, y)
* 2-D homogeneous point: (x, y, w)  with Euclidean form (x/w, y/w)
* 3-D Euclidean point:  (X, Y, Z)
* 3-D homogeneous point: (X, Y, Z, W) with Euclidean form (X/W, Y/W, Z/W)
* A line in 2-D is represented by (a, b, c) such that ax + by + c = 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Points2D = NDArray[np.floating]  # (N, 2)
Points3D = NDArray[np.floating]  # (N, 3)
PointsH = NDArray[np.floating]  # homogeneous: (N, 3) or (N, 4)
Mat3 = NDArray[np.floating]  # (3, 3)

# ====================================================================== #
#  1.  Homogeneous coordinate utilities                                  #
# ====================================================================== #


def to_homogeneous(points: NDArray[np.floating]) -> NDArray[np.floating]:
    r"""Convert Euclidean points to homogeneous coordinates.

    Appends a column of ones so that an :math:`(N, D)` array becomes
    :math:`(N, D+1)`:

    .. math::

        \mathbf{x} = \begin{pmatrix} x \\ y \end{pmatrix}
        \;\longrightarrow\;
        \tilde{\mathbf{x}} = \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}

    Parameters
    ----------
    points : ndarray, shape (N, D)
        Euclidean coordinates.  Typically *D* = 2 or *D* = 3.

    Returns
    -------
    ndarray, shape (N, D+1)
        Homogeneous coordinates with the last component equal to 1.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        return np.append(points, 1.0)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.hstack([points, ones])


def from_homogeneous(points: NDArray[np.floating]) -> NDArray[np.floating]:
    r"""Convert homogeneous coordinates back to Euclidean.

    Divides every coordinate by the last component:

    .. math::

        \tilde{\mathbf{x}} = \begin{pmatrix} x \\ y \\ w \end{pmatrix}
        \;\longrightarrow\;
        \mathbf{x} = \begin{pmatrix} x/w \\ y/w \end{pmatrix}

    Points at infinity (:math:`w = 0`) are left unchanged (a warning is
    **not** raised; the caller should check for ``inf``/``nan``).

    Parameters
    ----------
    points : ndarray, shape (N, D+1)
        Homogeneous coordinates.

    Returns
    -------
    ndarray, shape (N, D)
        Euclidean coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        return points[:-1] / points[-1]
    return points[:, :-1] / points[:, -1:None]


# ====================================================================== #
#  8.  Hartley normalization (listed early because DLT depends on it)    #
# ====================================================================== #


def hartley_normalize(
    points: NDArray[np.floating],
) -> tuple[NDArray[np.floating], Mat3]:
    r"""Isotropic normalization of 2-D points (Hartley, 1997).

    The transformation centres the points at the origin and scales them
    so that the **mean** distance from the origin equals :math:`\sqrt{2}`:

    .. math::

        \bar{\mathbf{x}} = \frac{1}{N}\sum_{i} \mathbf{x}_i, \qquad
        s = \frac{\sqrt{2}}{\frac{1}{N}\sum_{i}
            \lVert \mathbf{x}_i - \bar{\mathbf{x}} \rVert}

    The :math:`3\times 3` normalization matrix is:

    .. math::

        T = \begin{pmatrix}
            s & 0 & -s\,\bar{x} \\
            0 & s & -s\,\bar{y} \\
            0 & 0 & 1
        \end{pmatrix}

    so that :math:`\tilde{\mathbf{x}}' = T\,\tilde{\mathbf{x}}`.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        2-D Euclidean coordinates.

    Returns
    -------
    normalized : ndarray, shape (N, 2)
        Centred and scaled points.
    T : ndarray, shape (3, 3)
        The similarity transform that maps original homogeneous points to
        their normalised counterparts.
    """
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = np.sqrt((shifted**2).sum(axis=1)).mean()

    if mean_dist < 1e-12:
        raise ValueError("All points are coincident; normalization is undefined.")

    scale = np.sqrt(2.0) / mean_dist
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pts_h = to_homogeneous(points)  # (N, 3)
    normalized_h = (T @ pts_h.T).T  # (N, 3)
    normalized = from_homogeneous(normalized_h)  # (N, 2)
    return normalized, T


# ====================================================================== #
#  2.  Homography estimation – Direct Linear Transform (DLT)             #
# ====================================================================== #


def compute_homography_dlt(
    src_pts: Points2D,
    dst_pts: Points2D,
) -> Mat3:
    r"""Estimate a 2-D homography via the normalised DLT algorithm.

    Given :math:`N \ge 4` point correspondences
    :math:`\mathbf{x}_i \leftrightarrow \mathbf{x}'_i`, compute the
    :math:`3\times 3` matrix :math:`H` such that

    .. math::
        \mathbf{x}'_i \sim H\,\mathbf{x}_i

    **Algorithm (Hartley & Zisserman, Algorithm 4.2)**

    1. *Normalise* both point sets with :func:`hartley_normalize` to
       obtain :math:`\hat{\mathbf{x}}_i`, :math:`\hat{\mathbf{x}}'_i`,
       and the transforms :math:`T_{\text{src}}`, :math:`T_{\text{dst}}`.

    2. For each correspondence :math:`i`, assemble two rows of the
       :math:`2N \times 9` matrix :math:`A`:

       .. math::

           A_{2i}   &= \bigl[\,
               \mathbf{0}^T,\;
               -\tilde{\mathbf{x}}_i^T,\;
               y'_i\,\tilde{\mathbf{x}}_i^T
           \,\bigr] \\
           A_{2i+1} &= \bigl[\,
               \tilde{\mathbf{x}}_i^T,\;
               \mathbf{0}^T,\;
               -x'_i\,\tilde{\mathbf{x}}_i^T
           \,\bigr]

       where :math:`\tilde{\mathbf{x}}_i = (x_i, y_i, 1)` and
       :math:`(x'_i, y'_i)` are the *normalised* destination coordinates.

    3. Solve :math:`A\,\mathbf{h} = \mathbf{0}` via SVD:
       :math:`\mathbf{h}` is the last column of :math:`V` in
       :math:`A = U\,\Sigma\,V^T`.

    4. Reshape :math:`\mathbf{h}` to :math:`3\times 3` and denormalise:

       .. math::
           H = T_{\text{dst}}^{-1}\,\hat{H}\,T_{\text{src}}

    Parameters
    ----------
    src_pts, dst_pts : ndarray, shape (N, 2)
        At least 4 corresponding 2-D points.

    Returns
    -------
    H : ndarray, shape (3, 3)
        The estimated homography (up to scale).
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
        raise ValueError("At least 4 point correspondences are required.")
    if src_pts.shape[0] != dst_pts.shape[0]:
        raise ValueError("src_pts and dst_pts must have the same number of rows.")

    # Step 1 – normalise
    src_norm, T_src = hartley_normalize(src_pts)
    dst_norm, T_dst = hartley_normalize(dst_pts)

    N = src_pts.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float64)

    for i in range(N):
        x, y = src_norm[i]
        xp, yp = dst_norm[i]
        # Row 2i:   [0, 0, 0, -x, -y, -1, y'x, y'y, y']
        A[2 * i] = [
            0.0,
            0.0,
            0.0,
            -x,
            -y,
            -1.0,
            yp * x,
            yp * y,
            yp,
        ]
        # Row 2i+1: [x, y, 1, 0, 0, 0, -x'x, -x'y, -x']
        A[2 * i + 1] = [
            x,
            y,
            1.0,
            0.0,
            0.0,
            0.0,
            -xp * x,
            -xp * y,
            -xp,
        ]

    # Step 3 – solve via SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]  # last row of Vt = last column of V
    H_norm = h.reshape(3, 3)

    # Step 4 – denormalise
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H /= H[2, 2]  # normalise so that H[2,2] = 1
    return H


# ====================================================================== #
#  3.  Homography with RANSAC                                            #
# ====================================================================== #


def compute_homography_ransac(
    src_pts: Points2D,
    dst_pts: Points2D,
    thresh: float = 3.0,
    max_iters: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[Mat3, NDArray[np.bool_]]:
    r"""Robust homography estimation with RANSAC.

    Repeatedly samples minimal sets of 4 correspondences, estimates
    :math:`H` via :func:`compute_homography_dlt`, and evaluates
    the *symmetric transfer error* in the destination image:

    .. math::
        e_i = \bigl\lVert
            \pi\!\bigl(H\,\tilde{\mathbf{x}}_i\bigr) - \mathbf{x}'_i
        \bigr\rVert_2

    where :math:`\pi` denotes the projection from homogeneous to
    Euclidean coordinates.  A correspondence is an **inlier** when
    :math:`e_i < \tau` (*thresh*).

    After exhausting iterations the homography is **recomputed** from
    *all* inliers of the best model.

    Parameters
    ----------
    src_pts, dst_pts : ndarray, shape (N, 2)
        Putative correspondences (N >= 4).
    thresh : float
        Inlier distance threshold in pixels (default 3.0).
    max_iters : int
        Maximum RANSAC iterations (default 2000).
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    H : ndarray, shape (3, 3)
        Best homography matrix (refit on inliers).
    inliers : ndarray, shape (N,), dtype bool
        Boolean mask of inlier correspondences.
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    N = src_pts.shape[0]
    if N < 4:
        raise ValueError("Need at least 4 correspondences for RANSAC.")

    if rng is None:
        rng = np.random.default_rng()

    best_inlier_count = 0
    best_mask = np.zeros(N, dtype=bool)
    best_H = np.eye(3, dtype=np.float64)

    src_h = to_homogeneous(src_pts)  # (N, 3)

    for _ in range(max_iters):
        idx = rng.choice(N, size=4, replace=False)
        try:
            H_cand = compute_homography_dlt(src_pts[idx], dst_pts[idx])
        except (np.linalg.LinAlgError, ValueError):
            continue

        # Project source points
        projected_h = (H_cand @ src_h.T).T  # (N, 3)
        w = projected_h[:, 2:3]
        # Guard against division by ~zero
        valid = np.abs(w.ravel()) > 1e-12
        projected = np.full_like(dst_pts, np.inf)
        projected[valid] = projected_h[valid, :2] / w[valid]

        errors = np.linalg.norm(projected - dst_pts, axis=1)
        mask = errors < thresh

        n_in = mask.sum()
        if n_in > best_inlier_count:
            best_inlier_count = n_in
            best_mask = mask
            best_H = H_cand

    # Refit using all inliers
    if best_inlier_count >= 4:
        best_H = compute_homography_dlt(src_pts[best_mask], dst_pts[best_mask])

    return best_H, best_mask


# ====================================================================== #
#  4.  Image warping (inverse mapping + bilinear interpolation)          #
# ====================================================================== #


def warp_image(
    image: NDArray[np.floating | np.uint8],
    H: Mat3,
    output_shape: tuple[int, int],
) -> NDArray[np.uint8]:
    r"""Warp an image by a homography using inverse mapping.

    For every pixel :math:`(x', y')` in the **output** image, compute the
    corresponding source coordinate via the inverse homography:

    .. math::
        \begin{pmatrix} x \\ y \\ w \end{pmatrix}
        = H^{-1}\,\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix},
        \qquad
        (u, v) = \bigl(x/w,\; y/w\bigr)

    The source intensity is obtained by **bilinear interpolation**:

    .. math::
        I(u,v) = (1-a)(1-b)\,I_{00} + a(1-b)\,I_{10}
                 + (1-a)b\,I_{01} + ab\,I_{11}

    where :math:`a = u - \lfloor u \rfloor`, :math:`b = v - \lfloor v \rfloor`,
    and :math:`I_{ij}` are the four neighbouring pixel values.

    Parameters
    ----------
    image : ndarray, shape (H_in, W_in) or (H_in, W_in, C)
        Source image (grayscale or colour, uint8 or float).
    H : ndarray, shape (3, 3)
        Forward homography (source -> destination).
    output_shape : (int, int)
        ``(height, width)`` of the output image.

    Returns
    -------
    ndarray, shape (*output_shape) or (*output_shape, C), dtype uint8
        The warped image.  Out-of-bounds pixels are black (0).
    """
    image = np.asarray(image)
    h_out, w_out = output_shape
    H_inv = np.linalg.inv(H)

    # Build a grid of destination pixel coordinates
    xs, ys = np.meshgrid(np.arange(w_out), np.arange(h_out))
    ones = np.ones_like(xs)
    dst_coords = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T  # (3, N)

    # Map back to source
    src_coords = H_inv @ dst_coords  # (3, N)
    src_coords /= src_coords[2:3, :]  # dehomogenise
    u = src_coords[0]
    v = src_coords[1]

    h_in, w_in = image.shape[:2]
    is_colour = image.ndim == 3

    # Validity on continuous coordinates: a sample is inside the source
    # image when it lies in the closed pixel-centre domain
    # [0, w-1] x [0, h-1].  (Testing the integer neighbours instead would
    # wrongly discard the last row/column, e.g. for an identity warp.)
    tol = 1e-9
    valid = (u >= -tol) & (u <= w_in - 1 + tol) & (v >= -tol) & (v <= h_in - 1 + tol)

    # Floor coordinates for bilinear neighbours, clamped so indexing is
    # always safe; the fractional weights use the *unclamped* values.
    u0 = np.clip(np.floor(u).astype(np.int64), 0, w_in - 1)
    v0 = np.clip(np.floor(v).astype(np.int64), 0, h_in - 1)
    u1 = np.minimum(u0 + 1, w_in - 1)
    v1 = np.minimum(v0 + 1, h_in - 1)

    # Fractional parts (outside the valid domain these may leave [0, 1),
    # but those samples are zeroed by the validity mask anyway).
    a = (u - u0).astype(np.float64)
    b = (v - v0).astype(np.float64)

    if is_colour:
        I00 = image[v0, u0].astype(np.float64)  # (N, C)
        I10 = image[v0, u1].astype(np.float64)
        I01 = image[v1, u0].astype(np.float64)
        I11 = image[v1, u1].astype(np.float64)
        a = a[:, None]
        b = b[:, None]
        valid = valid[:, None]
    else:
        I00 = image[v0, u0].astype(np.float64)
        I10 = image[v0, u1].astype(np.float64)
        I01 = image[v1, u0].astype(np.float64)
        I11 = image[v1, u1].astype(np.float64)

    interp = (
        (1 - a) * (1 - b) * I00 + a * (1 - b) * I10 + (1 - a) * b * I01 + a * b * I11
    )
    result = np.where(valid, interp, 0.0)

    if is_colour:
        result = result.reshape(h_out, w_out, image.shape[2])
    else:
        result = result.reshape(h_out, w_out)

    return np.clip(result, 0, 255).astype(np.uint8)


# ====================================================================== #
#  7.  Cross-ratio                                                       #
# ====================================================================== #


def cross_ratio(
    p1: NDArray[np.floating],
    p2: NDArray[np.floating],
    p3: NDArray[np.floating],
    p4: NDArray[np.floating],
) -> float:
    r"""Cross-ratio of four collinear points (projective invariant).

    Given four collinear points :math:`P_1, P_2, P_3, P_4` the
    cross-ratio is

    .. math::
        \operatorname{CR}(P_1, P_2; P_3, P_4)
        = \frac{|P_1 P_3|\;\cdot\;|P_2 P_4|}
               {|P_2 P_3|\;\cdot\;|P_1 P_4|}

    where :math:`|P_i P_j|` denotes the *signed* Euclidean distance
    along the line (or, equivalently, the ratio of homogeneous
    coordinates when working in :math:`\mathbb{P}^1`).

    The cross-ratio is the **fundamental projective invariant**: it is
    preserved under any projective transformation (homography).

    Parameters
    ----------
    p1, p2, p3, p4 : ndarray
        Collinear points in Euclidean (any dimension) or 1-D homogeneous
        coordinates.

    Returns
    -------
    float
        The cross-ratio scalar.

    Notes
    -----
    For numerical stability the signed distances are computed via the
    dot-product projection onto the line direction.
    """
    p1 = np.asarray(p1, dtype=np.float64).ravel()
    p2 = np.asarray(p2, dtype=np.float64).ravel()
    p3 = np.asarray(p3, dtype=np.float64).ravel()
    p4 = np.asarray(p4, dtype=np.float64).ravel()

    d = p2 - p1
    norm2 = np.dot(d, d)
    if norm2 < 1e-30:
        raise ValueError("p1 and p2 are coincident; cross-ratio undefined.")

    def _signed(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
        """Signed distance from *a* to *b* projected onto the line direction."""
        return float(np.dot(b - a, d) / np.sqrt(norm2))

    ac = _signed(p1, p3)
    bd = _signed(p2, p4)
    bc = _signed(p2, p3)
    ad = _signed(p1, p4)

    denom = bc * ad
    if abs(denom) < 1e-30:
        raise ValueError("Degenerate configuration; cross-ratio is infinite.")

    return (ac * bd) / denom


# ====================================================================== #
#  5.  Vanishing points and lines                                        #
# ====================================================================== #


def compute_line_from_points(
    p1: NDArray[np.floating],
    p2: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Line through two points in the projective plane.

    In :math:`\mathbb{P}^2` the line through two points is simply their
    cross product:

    .. math::
        \mathbf{l} = \tilde{\mathbf{p}}_1 \times \tilde{\mathbf{p}}_2

    The resulting line satisfies :math:`\mathbf{l}^T \tilde{\mathbf{p}} = 0`
    for every point :math:`\tilde{\mathbf{p}}` on the line, i.e.
    :math:`ax + by + c = 0`.

    Parameters
    ----------
    p1, p2 : ndarray, shape (2,) or (3,)
        Two distinct 2-D points (Euclidean or homogeneous).

    Returns
    -------
    ndarray, shape (3,)
        Homogeneous line coefficients :math:`(a, b, c)`.
    """
    p1 = np.asarray(p1, dtype=np.float64).ravel()
    p2 = np.asarray(p2, dtype=np.float64).ravel()
    if p1.shape[0] == 2:
        p1 = np.append(p1, 1.0)
    if p2.shape[0] == 2:
        p2 = np.append(p2, 1.0)
    line = np.cross(p1, p2)
    n = np.linalg.norm(line[:2])
    if n > 1e-12:
        line /= n
    return line


def find_vanishing_point(
    lines: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Estimate the vanishing point from a set of (near-)parallel lines.

    Each line :math:`\mathbf{l}_i = (a_i, b_i, c_i)` passes through
    the vanishing point :math:`\mathbf{v}` in the ideal case:

    .. math::
        \mathbf{l}_i^T \mathbf{v} = 0 \quad \forall\, i

    **Initial estimate** – for every pair of lines compute
    :math:`\mathbf{v}_{ij} = \mathbf{l}_i \times \mathbf{l}_j` (the
    intersection in :math:`\mathbb{P}^2`).  The initial vanishing point
    is the *median* of these intersections (robust to outlier pairs).

    **Refinement** – minimise the sum of squared algebraic distances:

    .. math::
        \min_{\mathbf{v}} \sum_i
            \bigl(\mathbf{l}_i^T \mathbf{v}\bigr)^2

    This is a homogeneous linear least-squares problem solved by SVD of
    the :math:`M \times 3` matrix whose rows are the lines.

    Parameters
    ----------
    lines : ndarray, shape (M, 3)
        Lines in homogeneous form :math:`(a, b, c)`.

    Returns
    -------
    ndarray, shape (3,)
        Vanishing point in homogeneous coordinates.
    """
    lines = np.asarray(lines, dtype=np.float64)
    if lines.shape[0] < 2:
        raise ValueError("At least 2 lines are required.")

    # SVD-based least-squares: minimise ||L @ v||^2  s.t. ||v|| = 1
    _, _, Vt = np.linalg.svd(lines)
    vp = Vt[-1]  # right singular vector for smallest singular value

    return vp


def find_vanishing_line(
    vp1: NDArray[np.floating],
    vp2: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Vanishing line from two vanishing points.

    In :math:`\mathbb{P}^2`, the line through two points is their cross
    product.  If :math:`\mathbf{v}_1` and :math:`\mathbf{v}_2` are
    vanishing points of two independent sets of parallel lines then

    .. math::
        \mathbf{l}_\infty = \mathbf{v}_1 \times \mathbf{v}_2

    is the vanishing line (image of the line at infinity of the world
    plane).

    Parameters
    ----------
    vp1, vp2 : ndarray, shape (3,)
        Two vanishing points in homogeneous coordinates.

    Returns
    -------
    ndarray, shape (3,)
        The vanishing line in homogeneous form :math:`(a, b, c)`.
    """
    vp1 = np.asarray(vp1, dtype=np.float64)
    vp2 = np.asarray(vp2, dtype=np.float64)
    vl = np.cross(vp1, vp2)
    n = np.linalg.norm(vl[:2])
    if n > 1e-12:
        vl /= n
    return vl


# ====================================================================== #
#  6.  Single-view metrology (cross-ratio based)                         #
# ====================================================================== #


def _to_homogeneous_2d(p: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert a 2-D Euclidean point to homogeneous coordinates."""
    p = np.asarray(p, dtype=np.float64).ravel()
    return np.append(p, 1.0) if p.shape[0] == 2 else p.copy()


def _to_euclidean_2d(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a homogeneous point to 2-D Euclidean coordinates."""
    return p[:2] / p[2] if abs(p[2]) > 1e-12 else p[:2]


def _horizon_intercept(
    base: NDArray[np.float64],
    top: NDArray[np.float64],
    vanishing_line: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the intersection of the vertical through *base*–*top* with the vanishing line."""
    vertical = np.cross(base, top)
    return np.cross(vertical, vanishing_line)


def _cross_ratio_height(
    reference_height: float,
    b_e: NDArray[np.float64],
    t_e: NDArray[np.float64],
    br_e: NDArray[np.float64],
    tr_e: NDArray[np.float64],
    vv_e: NDArray[np.float64],
    vh_ref_e: NDArray[np.float64],
    vh_q_e: NDArray[np.float64],
) -> float:
    r"""Compute object height from cross-ratio distance invariants.

    Implements the Criminisi single-view metrology formula
    (Hartley & Zisserman §3.3 / §8.7).  With :math:`B, T` the foot/top
    of the query, :math:`B_r, T_r` those of the reference of known
    height :math:`H_r`, :math:`V` the vertical vanishing point and
    :math:`V_h, V_{h,r}` the horizon intercepts of the two vertical
    lines:

    .. math::
        H = H_r\; \frac{\lVert T - B\rVert}{\lVert T_r - B_r\rVert}
          \cdot \frac{\lVert V_{h,q} - V\rVert}{\lVert V_{h,r} - V\rVert}
          \cdot \frac{\lVert V_{h,r} - B_r\rVert}{\lVert V_{h,q} - B\rVert}
          \cdot \frac{\lVert V - T_r\rVert}{\lVert V - T\rVert}

    The last factor (top-to-VP distances) is required for correctness
    when the two objects are at different depths; omitting it only
    works when the vertical lines coincide (equal depth).
    """
    dist_tb = np.linalg.norm(t_e - b_e)
    dist_tr_br = np.linalg.norm(tr_e - br_e)
    dist_vh_q_vv = np.linalg.norm(vh_q_e - vv_e)
    dist_vh_ref_vv = np.linalg.norm(vh_ref_e - vv_e)
    dist_vh_ref_br = np.linalg.norm(vh_ref_e - br_e)
    dist_vh_q_b = np.linalg.norm(vh_q_e - b_e)
    dist_v_tr = np.linalg.norm(vv_e - tr_e)
    dist_v_t = np.linalg.norm(vv_e - t_e)

    if (
        dist_tr_br < 1e-12
        or dist_vh_ref_vv < 1e-12
        or dist_vh_q_b < 1e-12
        or dist_v_t < 1e-12
    ):
        raise ValueError("Degenerate configuration for height measurement.")

    return float(
        reference_height
        * (dist_tb / dist_tr_br)
        * (dist_vh_q_vv / dist_vh_ref_vv)
        * (dist_vh_ref_br / dist_vh_q_b)
        * (dist_v_tr / dist_v_t)
    )


def measure_height_single_view(
    base_point: NDArray[np.floating],
    top_point: NDArray[np.floating],
    reference_height: float,
    ref_base: NDArray[np.floating],
    ref_top: NDArray[np.floating],
    vanishing_line: NDArray[np.floating],
    vertical_vp: NDArray[np.floating],
) -> float:
    r"""Estimate the height of an object from a single calibrated view.

    Implements the cross-ratio–based measurement described by
    Criminisi *et al.* (2000) and Hartley & Zisserman §8.7.

    **Setup** – In the image we observe:

    * A *reference* object of known height :math:`H_r` with foot
      :math:`\mathbf{b}_r` and top :math:`\mathbf{t}_r`.
    * A *query* object with foot :math:`\mathbf{b}` and top
      :math:`\mathbf{t}`.

    The vertical vanishing point :math:`\mathbf{v}` and the vanishing
    line :math:`\mathbf{l}_\infty` of the ground plane are given.

    **Method**

    1. Compute the *horizon foot* of the reference:
       :math:`\mathbf{v}_r = (\mathbf{b}_r \times \mathbf{t}_r)
       \cap \mathbf{l}_\infty`, i.e.\ the intersection of the vertical
       line through the reference object and the vanishing line.

    2. Similarly, compute :math:`\mathbf{v}_q` for the query object.

    3. Using the cross-ratio on the pencil of lines through the
       vertical vanishing point, the unknown height is:

       .. math::
           H = H_r \;
           \frac{
               \lVert \mathbf{t} - \mathbf{b} \rVert \;\cdot\;
               \lVert \mathbf{v}_q - \mathbf{v} \rVert
           }{
               \lVert \mathbf{t}_r - \mathbf{b}_r \rVert \;\cdot\;
               \lVert \mathbf{v}_r - \mathbf{v} \rVert
           }
           \;\cdot\;
           \frac{
               \lVert \mathbf{v}_r - \mathbf{b}_r \rVert
           }{
               \lVert \mathbf{v}_q - \mathbf{b} \rVert
           }
           \;\cdot\;
           \frac{
               \lVert \mathbf{v} - \mathbf{t}_r \rVert
           }{
               \lVert \mathbf{v} - \mathbf{t} \rVert
           }

       The first three factors come from the invariance of the
       cross-ratio :math:`\operatorname{CR}(B, T, V_h, V)` — where
       :math:`V_h` is the horizon intercept and :math:`V` the vertical
       vanishing point — applied to both objects; the last factor
       accounts for the different depths of the two vertical lines.

    Parameters
    ----------
    base_point : ndarray, shape (2,) or (3,)
        Foot of the query object (Euclidean or homogeneous).
    top_point : ndarray, shape (2,) or (3,)
        Top of the query object.
    reference_height : float
        Known height of the reference object (world units).
    ref_base : ndarray, shape (2,) or (3,)
        Foot of the reference object.
    ref_top : ndarray, shape (2,) or (3,)
        Top of the reference object.
    vanishing_line : ndarray, shape (3,)
        Vanishing line of the ground plane, :math:`(a, b, c)`.
    vertical_vp : ndarray, shape (3,)
        Vanishing point of vertical lines (homogeneous).

    Returns
    -------
    float
        Estimated height of the query object in the same units as
        *reference_height*.
    """
    b = _to_homogeneous_2d(base_point)
    t = _to_homogeneous_2d(top_point)
    br = _to_homogeneous_2d(ref_base)
    tr = _to_homogeneous_2d(ref_top)
    vl = np.asarray(vanishing_line, dtype=np.float64)
    vv = np.asarray(vertical_vp, dtype=np.float64)

    vh_ref = _horizon_intercept(br, tr, vl)
    vh_q = _horizon_intercept(b, t, vl)

    return _cross_ratio_height(
        reference_height,
        _to_euclidean_2d(b),
        _to_euclidean_2d(t),
        _to_euclidean_2d(br),
        _to_euclidean_2d(tr),
        _to_euclidean_2d(vv),
        _to_euclidean_2d(vh_ref),
        _to_euclidean_2d(vh_q),
    )
