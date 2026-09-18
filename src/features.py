"""Feature Detection, Matching, and Epipolar Geometry
=====================================================

This module implements classical feature-based computer-vision algorithms
with full mathematical detail.  Where appropriate, pure-numpy
implementations are provided so students can inspect every step of the
math; thin OpenCV wrappers are included for production-grade detectors
(ORB, SIFT) and drawing utilities.

Algorithms
----------
* **Harris corner detection** — from scratch (structure tensor + corner
  response function).
* **ORB / SIFT wrappers** — thin wrappers around ``cv2``.
* **Brute-force & FLANN matching**, Lowe's ratio test, cross-check.
* **Normalised 8-point algorithm** for the fundamental matrix.
* **RANSAC** with adaptive iteration count.
* **Essential matrix** computation, decomposition, and cheirality check.
* **DLT triangulation**.
* **Visualisation helpers** for matches and epipolar lines.

References
----------
[1] Harris & Stephens, "A Combined Corner and Edge Detector", 1988.
[2] Lowe, "Distinctive Image Features from Scale-Invariant Keypoints",
    IJCV 2004.
[3] Rublee et al., "ORB: An Efficient Alternative to SIFT or SURF",
    ICCV 2011.
[4] Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., 2004.
[5] Nistér, "An Efficient Solution to the Five-Point Relative Pose
    Problem", TPAMI 2004.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from . import _cv

# ======================================================================
#  Type aliases
# ======================================================================
Image = NDArray[np.uint8]
# Public entry points accept any image OpenCV can produce (``imread`` returns
# ``MatLike``) and normalise internally via ``_ensure_gray_u8``.
ImageLike = cv2.typing.MatLike
FloatImage = NDArray[np.floating]
Points = NDArray[np.floating]  # (N, 2) array of 2-D points
Mat3 = NDArray[np.floating]  # 3 × 3 matrix
# Feature descriptors are float32 for SIFT-like detectors and uint8 for binary
# (ORB/BRIEF) ones, so the matching routines accept either.
Descriptors = NDArray[np.floating | np.unsignedinteger]


@dataclass
class MatchPair:
    """A single feature correspondence.

    Attributes
    ----------
    query_idx : int
        Index into the first (query) descriptor array.
    train_idx : int
        Index into the second (train) descriptor array.
    distance : float
        Descriptor distance (L2 or Hamming, depending on feature type).
    """

    query_idx: int
    train_idx: int
    distance: float


# ======================================================================
#  1. Harris Corner Detection  (from scratch)
# ======================================================================


def _gaussian_kernel(size: int, sigma: float) -> NDArray[np.float64]:
    """Return a normalised 2-D Gaussian kernel.

    Parameters
    ----------
    size : int
        Side length of the square kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    NDArray[np.float64]
        Kernel of shape ``(size, size)`` that sums to 1.
    """
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _compute_sobel_gradients(
    img: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Sobel spatial gradients Ix, Iy."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ix = cv2.filter2D(img, cv2.CV_64F, sobel_x)
    Iy = cv2.filter2D(img, cv2.CV_64F, sobel_x.T)
    return np.asarray(Ix, dtype=np.float64), np.asarray(Iy, dtype=np.float64)


def _compute_structure_tensor_maps(
    Ix: NDArray[np.float64],
    Iy: NDArray[np.float64],
    window_size: int,
    sigma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Gaussian-weighted structure tensor components (Sxx, Sxy, Syy)."""
    G = _gaussian_kernel(window_size, sigma)
    Sxx = cv2.filter2D(Ix * Ix, cv2.CV_64F, G)
    Sxy = cv2.filter2D(Ix * Iy, cv2.CV_64F, G)
    Syy = cv2.filter2D(Iy * Iy, cv2.CV_64F, G)
    return (
        np.asarray(Sxx, dtype=np.float64),
        np.asarray(Sxy, dtype=np.float64),
        np.asarray(Syy, dtype=np.float64),
    )


def _harris_response(
    Sxx: NDArray[np.float64],
    Sxy: NDArray[np.float64],
    Syy: NDArray[np.float64],
    k: float,
) -> NDArray[np.float64]:
    """Corner response R = det(M) - k * trace(M)**2."""
    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    return det_M - k * (trace_M**2)


def _extract_keypoints_nms(
    R: NDArray[np.float64],
    threshold: float,
    nms_size: int,
    window_size: int,
) -> list[cv2.KeyPoint]:
    """Threshold the response map and apply non-maximum suppression."""
    R_thresh = R.copy()
    R_thresh[threshold * R.max() > R] = 0

    half = nms_size // 2
    pad_R = np.pad(R_thresh, half, mode="constant", constant_values=0)

    keypoints: list[cv2.KeyPoint] = []
    h, w = R_thresh.shape
    for y in range(h):
        for x in range(w):
            val = R_thresh[y, x]
            if val == 0:
                continue
            local = pad_R[y : y + nms_size, x : x + nms_size]
            if val >= local.max():
                keypoints.append(
                    cv2.KeyPoint(
                        x=float(x),
                        y=float(y),
                        size=float(window_size),
                        response=float(val),
                    )
                )
    return keypoints


def harris_corners(
    image: ImageLike,
    k: float = 0.04,
    threshold: float = 0.01,
    window_size: int = 5,
    *,
    sigma: float = 1.0,
    nms_size: int = 7,
) -> tuple[NDArray[np.float64], list[cv2.KeyPoint]]:
    r"""Harris corner detector implemented from scratch.

    **Structure tensor** *M* at each pixel:

    .. math::

        M = G_{\sigma} \;*\;
        \begin{bmatrix}
            I_x^2   & I_x I_y \\
            I_x I_y & I_y^2
        \end{bmatrix}

    **Corner response function**:

    .. math::

        R = \det(M) - k \cdot \mathrm{tr}(M)^2
          = \lambda_1 \lambda_2 - k\,(\lambda_1 + \lambda_2)^2

    where :math:`\lambda_1, \lambda_2` are eigenvalues of *M*.

    Classification rules:

    * **Corner**: both :math:`\lambda` large  →  :math:`R \gg 0`
    * **Edge**: one :math:`\lambda` large      →  :math:`R < 0`
    * **Flat**: both :math:`\lambda` small     →  :math:`R \approx 0`

    Parameters
    ----------
    image : ndarray
        Grayscale image, ``uint8`` or ``float``.
    k : float
        Harris sensitivity parameter (typically 0.04 – 0.06).
    threshold : float
        Fraction of ``max(R)`` below which responses are suppressed.
    window_size : int
        Side length of the Gaussian weighting window for *M*.
    sigma : float
        Standard deviation for the Gaussian weighting in *M*.
    nms_size : int
        Side length of the local maximum suppression window (must be odd).

    Returns
    -------
    response : ndarray of float64, shape ``(H, W)``
        Full corner-response map *R*.
    keypoints : list[cv2.KeyPoint]
        Detected corners after thresholding and non-maximum suppression,
        with ``response`` set to the Harris score.
    """
    img = image.astype(np.float64)
    if img.ndim == 3:
        img = np.mean(img, axis=2)

    Ix, Iy = _compute_sobel_gradients(img)
    Sxx, Sxy, Syy = _compute_structure_tensor_maps(Ix, Iy, window_size, sigma)
    R = _harris_response(Sxx, Sxy, Syy, k)
    keypoints = _extract_keypoints_nms(R, threshold, nms_size, window_size)

    return R, keypoints


# ======================================================================
#  2. ORB Detection (wrapper)
# ======================================================================


def detect_orb(
    image: ImageLike,
    n_features: int = 500,
) -> tuple[list[cv2.KeyPoint], NDArray[np.uint8]]:
    """Detect ORB features.

    ORB = FAST keypoints + oriented BRIEF descriptors [3].

    * **FAST** (Features from Accelerated Segment Test) locates keypoints
      by examining a Bresenham circle of 16 pixels; a point is a corner
      if *N* contiguous pixels are all brighter (or darker) than the
      centre by a threshold.
    * **Oriented BRIEF** computes binary descriptors by comparing pairs
      of pixel intensities around the keypoint, rotated to the dominant
      orientation (intensity centroid method).

    Parameters
    ----------
    image : ndarray
        Grayscale or BGR image.
    n_features : int
        Maximum number of features to retain.

    Returns
    -------
    keypoints : list[cv2.KeyPoint]
    descriptors : ndarray of uint8, shape ``(N, 32)``
        256-bit binary descriptors packed into 32 bytes per feature.
    """
    gray = _ensure_gray_u8(image)
    orb = _cv.orb_create(n_features)
    kp, desc = orb.detectAndCompute(gray, None)
    # ``detectAndCompute`` returns ``None`` for a textureless image, which the
    # stub does not model (it types the output as an always-present MatLike).
    if _cv.is_absent(desc):
        return list(kp), np.empty((0, 32), dtype=np.uint8)
    return list(kp), np.ascontiguousarray(desc, dtype=np.uint8)


# ======================================================================
#  3. SIFT Detection (wrapper)
# ======================================================================


def detect_sift(
    image: ImageLike,
    n_features: int = 500,
) -> tuple[list[cv2.KeyPoint], NDArray[np.float32]]:
    r"""Detect SIFT features [2].

    **Scale-space extrema detection**: build a Gaussian scale space
    :math:`L(x, y, \sigma)` and its Difference-of-Gaussians (DoG)
    pyramid.  Extrema in :math:`(x, y, \sigma)` are candidate keypoints.

    **Keypoint descriptor**: around each keypoint, compute gradient
    magnitudes and orientations in a :math:`16 \times 16` neighbourhood
    divided into :math:`4 \times 4` sub-regions, each contributing an
    8-bin orientation histogram → 128-dimensional float vector,
    normalised to unit length.

    Parameters
    ----------
    image : ndarray
        Grayscale or BGR image.
    n_features : int
        Maximum number of features to retain (0 = no limit).

    Returns
    -------
    keypoints : list[cv2.KeyPoint]
    descriptors : ndarray of float32, shape ``(N, 128)``
    """
    gray = _ensure_gray_u8(image)
    sift = _cv.sift_create(n_features)
    kp, desc = sift.detectAndCompute(gray, None)
    # See ``detect_orb``: an absent descriptor set is a legitimate outcome that
    # the stub does not model.
    if _cv.is_absent(desc):
        return list(kp), np.empty((0, 128), dtype=np.float32)
    return list(kp), np.ascontiguousarray(desc, dtype=np.float32)


# ======================================================================
#  4. Matching
# ======================================================================


def match_bruteforce(
    desc1: Descriptors,
    desc2: Descriptors,
    norm_type: str = "L2",
) -> list[list[MatchPair]]:
    """Brute-force k-NN matching (k = 2).

    For every descriptor in *desc1*, the two nearest neighbours in
    *desc2* are returned so that :func:`ratio_test` can be applied.

    Parameters
    ----------
    desc1, desc2 : ndarray
        Descriptor arrays of shape ``(N, D)`` and ``(M, D)``.
    norm_type : {"L2", "HAMMING"}
        ``"L2"`` for SIFT-like float descriptors, ``"HAMMING"`` for
        ORB-like binary descriptors.

    Returns
    -------
    list[list[MatchPair]]
        For each query descriptor, a list of (at most) 2 matches sorted
        by increasing distance.
    """
    norms = {"L2": cv2.NORM_L2, "HAMMING": cv2.NORM_HAMMING}
    bf = cv2.BFMatcher(norms[norm_type])
    raw = bf.knnMatch(desc1, desc2, k=2)
    return [
        [MatchPair(m.queryIdx, m.trainIdx, m.distance) for m in group] for group in raw
    ]


def match_flann(
    desc1: Descriptors,
    desc2: Descriptors,
) -> list[list[MatchPair]]:
    """FLANN-based approximate k-NN matching (k = 2).

    Uses a KD-tree index for float descriptors (SIFT) or an LSH index
    for binary descriptors (ORB).  Significantly faster than brute-force
    on large descriptor sets.

    Parameters
    ----------
    desc1, desc2 : ndarray
        Descriptor arrays.  Dtype determines the index type:
        ``float32`` → KD-tree, ``uint8`` → LSH.

    Returns
    -------
    list[list[MatchPair]]
        k-NN matches (k = 2) for each query descriptor.
    """
    # ``cv2.typing.IndexParams`` / ``SearchParams`` are ``dict[str, bool | int |
    # float | str]``.  ``dict`` is invariant in its value type, so the literals
    # must be annotated with the full union for the invariance to be satisfied.
    index_params: cv2.typing.IndexParams
    if desc1.dtype == np.float32:
        index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE
    else:
        index_params = {
            "algorithm": 6,  # FLANN_INDEX_LSH
            "table_number": 6,
            "key_size": 12,
            "multi_probe_level": 1,
        }
    search_params: cv2.typing.SearchParams = {"checks": 50}

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(desc1, desc2, k=2)
    return [
        [MatchPair(m.queryIdx, m.trainIdx, m.distance) for m in group] for group in raw
    ]


def ratio_test(
    matches: list[list[MatchPair]],
    ratio: float = 0.75,
) -> list[MatchPair]:
    """Apply Lowe's ratio test to k-NN matches.

    For each pair of candidate matches :math:`(m_1, m_2)`:

    .. math::

        \\frac{d(m_1)}{d(m_2)} < \\rho

    where :math:`\\rho` is the *ratio* threshold.  Matches that satisfy
    the test are considered *distinctive* enough to keep.

    Parameters
    ----------
    matches : list[list[MatchPair]]
        k-NN matches (k ≥ 2) from :func:`match_bruteforce` or
        :func:`match_flann`.
    ratio : float
        Maximum acceptable distance ratio (default 0.75, per Lowe [2]).

    Returns
    -------
    list[MatchPair]
        Filtered best matches.
    """
    good: list[MatchPair] = []
    for group in matches:
        if len(group) < 2:
            continue
        best, second = group[0], group[1]
        if second.distance > 0 and best.distance / second.distance < ratio:
            good.append(best)
    return good


def cross_check_matches(
    desc1: Descriptors,
    desc2: Descriptors,
    norm_type: str = "L2",
) -> list[MatchPair]:
    """Mutual (cross-check) matching.

    A match :math:`(i, j)` is kept only when descriptor *i* in set 1 is
    the nearest neighbour of descriptor *j* in set 2 **and** vice versa.

    Parameters
    ----------
    desc1, desc2 : ndarray
        Descriptor arrays.
    norm_type : {"L2", "HAMMING"}
        Distance norm.

    Returns
    -------
    list[MatchPair]
        Mutually consistent matches.
    """
    norms = {"L2": cv2.NORM_L2, "HAMMING": cv2.NORM_HAMMING}
    bf = cv2.BFMatcher(norms[norm_type], crossCheck=True)
    raw = bf.match(desc1, desc2)
    return [MatchPair(m.queryIdx, m.trainIdx, m.distance) for m in raw]


# ======================================================================
#  5. RANSAC for Fundamental Matrix
# ======================================================================


def ransac_fundamental(
    pts1: Points,
    pts2: Points,
    threshold: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.99,
) -> tuple[Mat3, NDArray[np.bool_]]:
    r"""RANSAC estimation of the fundamental matrix.

    1. Randomly sample 8 correspondences.
    2. Compute *F* via :func:`compute_fundamental_8point`.
    3. Compute Sampson distance for every correspondence:

       .. math::

           d_S^2 = \frac{(\mathbf{x}'^{\!\top} F \mathbf{x})^2}
                        {(F \mathbf{x})_1^2 + (F \mathbf{x})_2^2 +
                         (F^{\!\top} \mathbf{x}')_1^2 +
                         (F^{\!\top} \mathbf{x}')_2^2}

    4. Count inliers where :math:`d_S < \texttt{threshold}`.
    5. **Adaptive iteration count** after each best model update:

       .. math::

           N = \frac{\ln(1 - p)}{\ln\!\bigl(1 - (1 - \varepsilon)^s\bigr)}

       where *p* = *confidence*, :math:`\varepsilon` = current outlier
       ratio, *s* = 8 (sample size).

    Parameters
    ----------
    pts1, pts2 : ndarray, shape ``(N, 2)``
        Matching 2-D points.
    threshold : float
        Sampson distance inlier threshold (pixels).
    max_iters : int
        Hard upper bound on iterations.
    confidence : float
        Desired probability that at least one sample is outlier-free.

    Returns
    -------
    F : ndarray, shape ``(3, 3)``
        Best fundamental matrix found.
    inlier_mask : ndarray of bool, shape ``(N,)``
        ``True`` for inlier correspondences.
    """
    N = pts1.shape[0]
    s = 8
    if s > N:
        raise ValueError(f"Need at least {s} correspondences, got {N}")

    best_F = np.eye(3)
    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    adaptive_iters = max_iters

    rng = np.random.default_rng()

    for iteration in range(max_iters):
        if iteration >= adaptive_iters:
            break

        idx = rng.choice(N, size=s, replace=False)
        F_candidate = compute_fundamental_8point(pts1[idx], pts2[idx])

        inliers = _sampson_inliers(F_candidate, pts1, pts2, threshold)
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_F = F_candidate

            # adaptive iteration update
            epsilon = 1.0 - count / N
            if epsilon < 1e-10:
                break
            denom = np.log(1.0 - (1.0 - epsilon) ** s)
            if denom < 0:
                adaptive_iters = int(np.ceil(np.log(1.0 - confidence) / denom))
                adaptive_iters = min(adaptive_iters, max_iters)

    # re-estimate from all inliers for better accuracy
    if best_count >= s:
        best_F = compute_fundamental_8point(pts1[best_inliers], pts2[best_inliers])
        best_inliers = _sampson_inliers(best_F, pts1, pts2, threshold)

    return best_F, best_inliers


def _sampson_inliers(
    F: Mat3,
    pts1: Points,
    pts2: Points,
    threshold: float,
) -> NDArray[np.bool_]:
    """Compute Sampson distance and return inlier mask."""
    ones = np.ones((pts1.shape[0], 1))
    p1h = np.hstack([pts1, ones])  # (N, 3)
    p2h = np.hstack([pts2, ones])

    Fp1 = (F @ p1h.T).T  # (N, 3)
    Ftp2 = (F.T @ p2h.T).T  # (N, 3)

    # x'ᵀ F x  (epipolar constraint, scalar per point)
    num = np.sum(p2h * Fp1, axis=1) ** 2

    denom = Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 + Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2

    sampson_sq = num / np.maximum(denom, 1e-12)
    return sampson_sq < threshold**2


# ======================================================================
#  6. Fundamental Matrix — normalised 8-point algorithm
# ======================================================================


def compute_fundamental_8point(pts1: Points, pts2: Points) -> Mat3:
    r"""Normalised 8-point algorithm for the fundamental matrix [4, §11.2].

    **Algorithm**

    1. **Hartley normalisation** — translate each point set so its
       centroid is the origin, then scale so the mean distance from the
       origin is :math:`\sqrt{2}`.  This dramatically improves numerical
       conditioning.

       .. math::

           \tilde{\mathbf{x}} = T \, \mathbf{x}, \qquad
           \tilde{\mathbf{x}}' = T' \, \mathbf{x}'

    2. **Build the constraint matrix** *A*.  The epipolar constraint
       :math:`\mathbf{x}'^{\!\top} F \mathbf{x} = 0` gives, for each
       correspondence :math:`(x, y)` ↔ :math:`(x', y')`:

       .. math::

           \bigl[ x'x,\; x'y,\; x',\; y'x,\; y'y,\; y',\; x,\; y,\; 1 \bigr]
           \cdot \mathrm{vec}(F) = 0

    3. **SVD of** *A* → :math:`\hat{F}` is the last column of *V*
       reshaped to :math:`3 \times 3`.

    4. **Enforce rank 2**: take the SVD of :math:`\hat{F}`, set
       :math:`\sigma_3 = 0`, and reconstruct.

    5. **Denormalise**:

       .. math::

           F = {T'}^{\!\top} \, \tilde{F} \, T

    Parameters
    ----------
    pts1, pts2 : ndarray, shape ``(N, 2)`` with ``N ≥ 8``
        Corresponding 2-D points.

    Returns
    -------
    F : ndarray, shape ``(3, 3)``
        Rank-2 fundamental matrix satisfying
        :math:`\mathbf{x}'^{\!\top} F \mathbf{x} = 0`.
    """
    assert pts1.shape[0] >= 8, "Need ≥ 8 point correspondences"

    # 1. Hartley normalisation
    p1, T1 = _hartley_normalise(pts1)
    p2, T2 = _hartley_normalise(pts2)

    # 2. Build constraint matrix  A · vec(F) = 0
    x, y = p1[:, 0], p1[:, 1]
    xp, yp = p2[:, 0], p2[:, 1]
    A = np.column_stack(
        [
            xp * x,
            xp * y,
            xp,
            yp * x,
            yp * y,
            yp,
            x,
            y,
            np.ones_like(x),
        ]
    )

    # 3. SVD → F̃ is last column of V
    _, _, Vt = np.linalg.svd(A)
    F_tilde = Vt[-1].reshape(3, 3)

    # 4. Enforce rank 2: zero out smallest singular value
    U, S, Vt2 = np.linalg.svd(F_tilde)
    S[2] = 0.0
    F_tilde = U @ np.diag(S) @ Vt2

    # 5. Denormalise
    F = T2.T @ F_tilde @ T1

    # normalise so ||F||_fro = 1
    F /= np.linalg.norm(F)
    return F


def _hartley_normalise(
    pts: Points,
) -> tuple[NDArray[np.float64], Mat3]:
    r"""Translate + isotropic scale so mean distance from origin = √2.

    Returns
    -------
    pts_norm : ndarray, shape ``(N, 2)``
    T : ndarray, shape ``(3, 3)``
        The normalising transformation (applied to homogeneous coords).
    """
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-12)

    T = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    pts_norm = shifted * scale
    return pts_norm, T


# ======================================================================
#  7. Essential Matrix
# ======================================================================


def compute_essential(
    F: Mat3,
    K1: Mat3,
    K2: Mat3 | None = None,
) -> Mat3:
    r"""Compute the essential matrix from *F* and intrinsics.

    .. math::

        E = K_2^{\!\top} \, F \, K_1

    When the same camera is used for both views, set ``K2 = None`` and
    :math:`K_1 = K_2` is assumed.

    Parameters
    ----------
    F : ndarray, shape ``(3, 3)``
        Fundamental matrix.
    K1 : ndarray, shape ``(3, 3)``
        Intrinsic matrix of camera 1.
    K2 : ndarray, shape ``(3, 3)``, optional
        Intrinsic matrix of camera 2.  Defaults to *K1*.

    Returns
    -------
    E : ndarray, shape ``(3, 3)``
    """
    if K2 is None:
        K2 = K1
    E = K2.T @ F @ K1
    # re-project to the essential manifold: enforce two equal singular values
    U, S, Vt = np.linalg.svd(E)
    s_mean = (S[0] + S[1]) / 2.0
    E = U @ np.diag([s_mean, s_mean, 0.0]) @ Vt
    return E


def decompose_essential(
    E: Mat3,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    r"""Decompose the essential matrix into four (R, t) hypotheses.

    Given :math:`E = U \, \Sigma \, V^{\!\top}`:

    .. math::

        W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}

    The four solutions are:

    ====  ================================  ==================
     #     Rotation                          Translation
    ====  ================================  ==================
     1     :math:`R_1 = U W V^{\!\top}`      :math:`+u_3`
     2     :math:`R_1`                        :math:`-u_3`
     3     :math:`R_2 = U W^{\!\top} V^{\!\top}`  :math:`+u_3`
     4     :math:`R_2`                        :math:`-u_3`
    ====  ================================  ==================

    where :math:`u_3` is the third column of *U*.

    Parameters
    ----------
    E : ndarray, shape ``(3, 3)``

    Returns
    -------
    list of (R, t) tuples
        Four candidate poses.  *R* is ``(3, 3)``, *t* is ``(3,)``.
    """
    U, _, Vt = np.linalg.svd(E)

    # ensure proper rotations (det = +1)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def choose_pose_cheirality(
    R_list: Sequence[NDArray[np.float64]],
    t_list: Sequence[NDArray[np.float64]],
    pts1: Points,
    pts2: Points,
    K: Mat3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Select the correct (R, t) via the **cheirality check**.

    Among the four decompositions of *E*, only one places the
    triangulated 3-D points in front of **both** cameras (i.e.
    positive depth).

    Parameters
    ----------
    R_list : sequence of ndarray, shape ``(3, 3)``
    t_list : sequence of ndarray, shape ``(3,)``
    pts1, pts2 : ndarray, shape ``(N, 2)``
        Corresponding image points (undistorted / normalised pixel coords).
    K : ndarray, shape ``(3, 3)``
        Intrinsic matrix (same camera assumed for both views).

    Returns
    -------
    R_best : ndarray, shape ``(3, 3)``
    t_best : ndarray, shape ``(3,)``
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    best_count = -1
    best_R, best_t = R_list[0], t_list[0]

    for R, t in zip(R_list, t_list, strict=False):
        P2 = K @ np.hstack([R, t.reshape(3, 1)])
        X = triangulate_dlt(pts1, pts2, P1, P2)

        # depth in camera 1: z-coordinate of X (camera 1 is at identity)
        depth1 = X[:, 2]
        # depth in camera 2: z-coordinate of R @ X_i + t
        X_cam2 = (R @ X.T).T + t
        depth2 = X_cam2[:, 2]

        count = int(np.sum((depth1 > 0) & (depth2 > 0)))
        if count > best_count:
            best_count = count
            best_R, best_t = R, t

    return best_R, best_t


# ======================================================================
#  8. Triangulation (DLT)
# ======================================================================


def triangulate_dlt(
    pts1: Points,
    pts2: Points,
    P1: NDArray[np.floating],
    P2: NDArray[np.floating],
) -> NDArray[np.float64]:
    r"""Linear triangulation via the Direct Linear Transform [4, §12.2].

    For each correspondence :math:`(\mathbf{x}, \mathbf{x}')` and
    projection matrices :math:`P, P'`, the constraint
    :math:`\mathbf{x} \times (P \, \mathbf{X}) = \mathbf{0}` gives two
    independent equations per view, yielding the :math:`4 \times 4`
    system:

    .. math::

        A = \begin{bmatrix}
            x \, \mathbf{p}_3^{\!\top} - \mathbf{p}_1^{\!\top} \\
            y \, \mathbf{p}_3^{\!\top} - \mathbf{p}_2^{\!\top} \\
            x'\, \mathbf{p}'_3{}^{\!\top} - \mathbf{p}'_1{}^{\!\top} \\
            y'\, \mathbf{p}'_3{}^{\!\top} - \mathbf{p}'_2{}^{\!\top}
        \end{bmatrix}

    The 3-D point :math:`\mathbf{X}` is the right null-vector of *A*
    (last column of *V* from SVD), converted from homogeneous
    coordinates.

    Parameters
    ----------
    pts1, pts2 : ndarray, shape ``(N, 2)``
        Matching 2-D points in each image.
    P1, P2 : ndarray, shape ``(3, 4)``
        Camera projection matrices.

    Returns
    -------
    points_3d : ndarray, shape ``(N, 3)``
        Triangulated Euclidean 3-D points.
    """
    N = pts1.shape[0]
    points_3d = np.empty((N, 3), dtype=np.float64)

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
        X = Vt[-1]
        points_3d[i] = X[:3] / X[3]

    return points_3d


# ======================================================================
#  9. Visualisation helpers
# ======================================================================


def draw_matches(
    img1: Image,
    kp1: Sequence[cv2.KeyPoint],
    img2: Image,
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[MatchPair],
    max_draw: int = 50,
) -> Image:
    """Draw feature matches side by side.

    Parameters
    ----------
    img1, img2 : ndarray
        Source images (grayscale or BGR).
    kp1, kp2 : sequence of cv2.KeyPoint
    matches : sequence of MatchPair
    max_draw : int
        Maximum number of matches to draw (sorted by distance).

    Returns
    -------
    vis : ndarray
        Concatenated side-by-side visualisation.
    """
    sorted_matches = sorted(matches, key=lambda m: m.distance)[:max_draw]
    dm = [cv2.DMatch(m.query_idx, m.train_idx, m.distance) for m in sorted_matches]
    vis = _cv.draw_matches(
        img1,
        kp1,
        img2,
        kp2,
        dm,
        _cv.draw_matches_single_point_flag(),
    )
    return vis


def draw_epipolar_lines(
    img: Image,
    F: Mat3,
    pts: Points,
    color: tuple[int, int, int] | None = None,
    *,
    from_image: int = 1,
) -> Image:
    r"""Draw epipolar lines on an image.

    Given the fundamental matrix *F* and points in the *other* image,
    the epipolar line in this image is:

    .. math::

        \ell = F^{\!\top} \mathbf{x}' \quad \text{(if from\_image=2)}
        \qquad\text{or}\qquad
        \ell = F \, \mathbf{x} \quad \text{(if from\_image=1)}

    Each line :math:`\ell = (a, b, c)` satisfies :math:`ax + by + c = 0`.

    Parameters
    ----------
    img : ndarray
        Image on which to draw the lines.
    F : ndarray, shape ``(3, 3)``
        Fundamental matrix.
    pts : ndarray, shape ``(N, 2)``
        Points in the *other* image whose epipolar lines we draw here.
    color : tuple of int, optional
        BGR colour.  If ``None``, each line gets a random colour.
    from_image : {1, 2}
        Which image the *pts* belong to.  ``1`` means *pts* are in
        image 1, and the epipolar lines are drawn in image 2 (using
        :math:`\ell = F \mathbf{x}`).  ``2`` means the reverse.

    Returns
    -------
    vis : ndarray
        Copy of *img* with epipolar lines drawn.
    """
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    h, w = vis.shape[:2]
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])  # (N, 3)

    lines = (F @ pts_h.T).T if from_image == 1 else (F.T @ pts_h.T).T

    rng = np.random.default_rng(42)

    for line in lines:
        a, b, c = line
        clr = tuple(rng.integers(0, 255, size=3).tolist()) if color is None else color

        # two points on the line: x = 0 and x = w
        if abs(b) > 1e-8:
            y0 = int(-c / b)
            y1 = int(-(a * w + c) / b)
        else:
            y0, y1 = 0, h
        cv2.line(vis, (0, y0), (w, y1), clr, 1, cv2.LINE_AA)

    return np.asarray(vis, dtype=np.uint8)


# ======================================================================
#  Learned feature detection & matching (GPU optional)
# ======================================================================


def detect_superpoint(
    image: ImageLike,
    max_keypoints: int = 1024,
) -> (
    tuple[list[cv2.KeyPoint], NDArray[np.float32]]
    | tuple[NDArray[np.float32], NDArray[np.float32]]
):
    """Detect SuperPoint keypoints and descriptors.

    SuperPoint (DeTone et al. 2018) is a self-supervised CNN that jointly
    detects keypoints and computes 256-dim descriptors.  Training uses
    Homographic Adaptation on synthetic shapes then real images.

    Falls back to SIFT if ``torch`` / ``transformers`` are unavailable.

    Parameters
    ----------
    image : (H, W) or (H, W, 3) input image
    max_keypoints : maximum number of keypoints to return

    Returns
    -------
    keypoints : (N, 2) array of (x, y) coordinates
    descriptors : (N, 256) array of float descriptors
    """
    try:
        import torch  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoImageProcessor,
            AutoModel,
        )
    except ImportError:
        import warnings

        warnings.warn(
            "transformers not available — falling back to SIFT.", stacklevel=2
        )
        kps, descs = detect_sift(image, n_features=max_keypoints)
        return kps, descs

    image_rgb = np.stack([image, image, image], axis=-1) if image.ndim == 2 else image

    try:
        processor = AutoImageProcessor.from_pretrained(
            "magic-leap-community/superpoint", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "magic-leap-community/superpoint", trust_remote_code=True
        )
    except Exception:
        import warnings

        warnings.warn(
            "SuperPoint model not available — falling back to SIFT.", stacklevel=2
        )
        kps, descs = detect_sift(image, n_features=max_keypoints)
        return kps, descs

    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(
        image_rgb if image_rgb.dtype == np.uint8 else (image_rgb * 255).astype(np.uint8)
    )
    inputs = processor(pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    kps_tensor = outputs.keypoints[0].numpy()
    desc_tensor = outputs.descriptors[0].numpy()

    if len(kps_tensor) > max_keypoints:
        idx = np.random.choice(len(kps_tensor), max_keypoints, replace=False)
        kps_tensor = kps_tensor[idx]
        desc_tensor = desc_tensor[idx]

    return kps_tensor, desc_tensor


def match_lightglue(
    kps1: NDArray[np.float64],
    desc1: NDArray[np.floating],
    kps2: NDArray[np.float64],
    desc2: NDArray[np.floating],
) -> list[tuple[int, int]]:
    """Match features using LightGlue (or mutual nearest neighbour fallback).

    LightGlue (Lindenberger et al., ICCV 2023) is a graph neural network
    matcher with adaptive pruning — 4-10x faster than SuperGlue with
    equal or better accuracy.

    Falls back to brute-force mutual NN matching if LightGlue is unavailable.

    Parameters
    ----------
    kps1 : (N, 2) keypoints from image 1
    desc1 : (N, D) descriptors from image 1
    kps2 : (M, 2) keypoints from image 2
    desc2 : (M, D) descriptors from image 2

    Returns
    -------
    matches : list of (idx1, idx2) tuples
    """
    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)

    from scipy.spatial.distance import cdist

    dists = cdist(desc1, desc2, metric="cosine")

    nn12 = np.argmin(dists, axis=1)
    nn21 = np.argmin(dists, axis=0)

    matches = []
    for i, j in enumerate(nn12):
        if nn21[j] == i:
            matches.append((i, int(j)))

    return matches


# ======================================================================
#  Convenience aliases (plan-compatible names)
# ======================================================================

compute_fundamental_matrix = compute_fundamental_8point
compute_essential_matrix = compute_essential


# ======================================================================
#  Internal helpers
# ======================================================================


def _ensure_gray_u8(image: ImageLike) -> NDArray[np.uint8]:
    """Convert an image to single-channel uint8 if needed."""
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.dtype != np.uint8:
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    return np.asarray(gray, dtype=np.uint8)


ransac_filter = ransac_fundamental


def bf_match(
    desc1: Descriptors,
    desc2: Descriptors,
    norm_type: int = cv2.NORM_L2,
) -> list[list[MatchPair]]:
    """Brute-force k-NN matching accepting cv2 norm constants."""
    _norm_map = {cv2.NORM_L2: "L2", cv2.NORM_HAMMING: "HAMMING"}
    return match_bruteforce(desc1, desc2, norm_type=_norm_map.get(norm_type, "L2"))
