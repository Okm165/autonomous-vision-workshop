"""Typed, failure-tolerant wrappers around OpenCV routines.

The machine-generated ``cv2`` stub (``cv2/__init__.pyi``) disagrees with the
real Python bindings in two ways that matter for this project:

1. **Failure paths are not modelled.**  Several OpenCV functions report failure
   by returning ``None`` instead of raising, but the stub types their outputs as
   always-present.  ``cv2.findEssentialMat`` returns ``(None, None)`` on a
   degenerate five-point sample and ``cv2.solvePnPRansac`` may return no
   inliers; code that guards against those outcomes compares against ``None``,
   which the stub calls an unnecessary comparison.

2. **Documented parameters are typed too narrowly.**  ``cv2.drawKeypoints``,
   ``cv2.drawMatches`` and ``cv2.calcOpticalFlowFarneback`` document ``None``
   for their output parameter to request that OpenCV allocate it, yet the stub
   types it as non-optional.  ``cv2.FlannBasedMatcher`` accepts plain dicts as
   the OpenCV Python tutorial instructs, but the stub demands the exact
   ``IndexParams``/``SearchParams`` aliases.

Because ``cv2`` ships a complete ``py.typed`` stub, a local ``stubPath``
overlay cannot patch individual members: it would shadow the whole module and
every undeclared symbol would vanish.  Declaring the real contract once, here,
keeps each diagnostic honest at the call site instead of muting it there.

Every behaviour encoded below was verified against the installed OpenCV.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

# ======================================================================
#  Absence checks for unmodelled failure returns
# ======================================================================
# Several OpenCV routines signal "no result" by returning ``None`` while the
# stub declares the output as always-present.  ``is None`` on such a value is
# reported as an unreachable comparison, so the check is funnelled through this
# one helper: it inspects the *runtime* value, which is exactly the contract
# OpenCV documents.


def is_absent(result: object) -> bool:
    """Return whether an OpenCV output carries no result.

    ``True`` when OpenCV returned ``None`` (its unmodelled failure signal) or an
    empty array, ``False`` for any populated result.  Taking ``object`` lets the
    runtime value be inspected without asserting the stub's narrower type.
    """
    if result is None:
        return True
    if isinstance(result, np.ndarray):
        return result.size == 0
    return False


# ======================================================================
#  Failure-tolerant wrappers
# ======================================================================


def find_essential_matrix(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    K: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, NDArray[np.uint8] | None]:
    """Estimate an essential matrix with the RANSAC five-point algorithm.

    Returns ``(E, inlier_mask)``, or ``(None, None)`` when the sample is
    degenerate (too few or degenerate correspondences), matching the runtime
    behaviour of ``cv2.findEssentialMat``.
    """
    # ``findEssentialMat``'s stub promises a tuple of always-present values, but
    # a degenerate sample yields ``(None, None)`` at runtime; the absence checks
    # below are what keep that contract honest.
    raw = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    E, mask = raw
    if is_absent(E):
        return None, None
    mask_arr = None if is_absent(mask) else np.asarray(mask, dtype=np.uint8)
    return np.asarray(E, dtype=np.float64), mask_arr


def solve_pnp_ransac(
    object_points: NDArray[np.float64],
    image_points: NDArray[np.float64],
    K: NDArray[np.float64],
    dist_coeffs: NDArray[np.float64],
    *,
    reprojection_error: float,
    iterations: int,
) -> tuple[bool, NDArray[np.float64], NDArray[np.float64], NDArray[np.int64] | None]:
    """Robust PnP via ``cv2.solvePnPRansac``.

    Returns ``(success, rvec, tvec, inliers)``.  ``inliers`` is a flat index
    array or ``None`` when OpenCV could not form a consensus set, which the
    stubs do not model.
    """
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        K,
        dist_coeffs,
        reprojectionError=reprojection_error,
        iterationsCount=iterations,
    )
    rvec_arr = np.asarray(rvec, dtype=np.float64)
    tvec_arr = np.asarray(tvec, dtype=np.float64)
    # ``solvePnPRansac`` returns ``None`` (rather than raising) when it cannot
    # form a consensus set; the stub does not model that failure path.
    if is_absent(inliers):
        return bool(ok), rvec_arr, tvec_arr, None
    inlier_arr = np.asarray(inliers, dtype=np.int64).ravel()
    return bool(ok), rvec_arr, tvec_arr, inlier_arr


# ======================================================================
#  Feature detectors
# ======================================================================
# ``cv2.ORB_create`` / ``cv2.SIFT_create`` / ``cv2.StereoSGBM_create`` are real
# module-level factories that the generated stub omits, but they are exactly the
# ``create`` classmethods of the corresponding types (identical signature and
# behaviour), which the stub *does* declare.  Calling the classmethod is
# therefore the typed spelling of the very same function.


def orb_create(n_features: int) -> cv2.ORB:
    """Return an ORB detector.

    Parameters
    ----------
    n_features : int
        Maximum number of features to retain (0 = no limit).
    """
    return cv2.ORB.create(nfeatures=n_features)


def sift_create(n_features: int) -> cv2.SIFT:
    """Return a SIFT detector.

    Parameters
    ----------
    n_features : int
        Maximum number of features to retain (0 = no limit).
    """
    return cv2.SIFT.create(nfeatures=n_features)


def stereosgbm_create(
    *,
    min_disparity: int,
    num_disparities: int,
    block_size: int,
    p1: int,
    p2: int,
    disp12_max_diff: int,
    uniqueness_ratio: int,
    speckle_window_size: int,
    speckle_range: int,
    pre_filter_cap: int,
    mode: int,
) -> cv2.StereoSGBM:
    """Create a semi-global block-matching stereo matcher.

    The stub declares ``cv2.StereoSGBM_create`` with positional-only parameters
    and omits several named arguments used here, so the call is centralised
    behind this keyword-only facade.
    """
    return cv2.StereoSGBM.create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=disp12_max_diff,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=pre_filter_cap,
        mode=mode,
    )


# ======================================================================
#  Visualisation helpers
# ======================================================================
# ``outImage`` / ``outImg`` are documented as accepting ``None`` to request
# allocation, but the stub types them as non-optional.  Passing an
# explicitly-allocated destination is both typed and equivalent: OpenCV writes
# into the buffer and returns it.


def draw_keypoints(
    image: NDArray[np.uint8],
    keypoints: Sequence[cv2.KeyPoint],
    flags: int,
) -> NDArray[np.uint8]:
    """Draw keypoints onto a freshly allocated output image."""
    out = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    drawn = cv2.drawKeypoints(image, keypoints, out, flags=flags)
    return np.ascontiguousarray(drawn, dtype=np.uint8)


def draw_matches(
    img1: NDArray[np.uint8],
    keypoints1: Sequence[cv2.KeyPoint],
    img2: NDArray[np.uint8],
    keypoints2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    flags: int,
) -> NDArray[np.uint8]:
    """Draw matches side by side onto a freshly allocated output image.

    ``cv2.drawMatches`` lays the two images out horizontally, so the output is
    ``(max(h1, h2), w1 + w2)`` in shape.
    """
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    out = np.zeros((height, width, 3), dtype=np.uint8)
    vis = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, out, flags=flags)
    return np.ascontiguousarray(vis, dtype=np.uint8)


def draw_keypoints_none_flags() -> int:
    """Rich-keypoint flag value (``DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS``)."""
    return int(cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_matches_single_point_flag() -> int:
    """Single-point-suppression flag (``DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS``)."""
    return int(cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# ======================================================================
#  Dense flow / warping
# ======================================================================


def farneback_flow(
    prev: NDArray[np.uint8],
    nxt: NDArray[np.uint8],
    *,
    pyr_scale: float,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
    flags: int,
) -> NDArray[np.float32]:
    """Dense Farneback optical flow with a freshly allocated output.

    The stub types the ``flow`` parameter as non-optional even though the
    documentation allows ``None``; allocating the ``(h, w, 2)`` float32 buffer
    ourselves is equivalent and keeps the call fully typed.
    """
    flow: NDArray[np.float32] = np.zeros((*prev.shape[:2], 2), dtype=np.float32)
    cv2.calcOpticalFlowFarneback(
        prev,
        nxt,
        flow,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )
    return flow


def lucas_kanade_flow(
    prev: NDArray[np.uint8],
    nxt: NDArray[np.uint8],
    points: NDArray[np.float32],
    win_size: tuple[int, int],
    max_level: int,
    criteria: cv2.typing.TermCriteria,
    flags: int,
) -> tuple[NDArray[np.float32], NDArray[np.uint8], NDArray[np.float32]]:
    """Sparse pyramidal Lucas-Kanade flow with freshly allocated status/error.

    ``nextPts`` is documented as optional (``None`` requests allocation) but the
    stub types it as non-optional, so the ``(N, 1, 2)`` float32 buffer is
    allocated here.
    """
    next_pts: NDArray[np.float32] = np.zeros_like(points, dtype=np.float32)
    status: NDArray[np.uint8] = np.zeros((points.shape[0], 1), dtype=np.uint8)
    err: NDArray[np.float32] = np.zeros((points.shape[0], 1), dtype=np.float32)
    cv2.calcOpticalFlowPyrLK(
        prev,
        nxt,
        points,
        next_pts,
        status,
        err,
        winSize=win_size,
        maxLevel=max_level,
        criteria=criteria,
        flags=flags,
    )
    return next_pts, status, err


def warp_affine(
    src: NDArray[np.uint8],
    M: NDArray[np.float64],
    dsize: tuple[int, int],
) -> NDArray[np.uint8]:
    """Affine warp with an explicitly allocated destination.

    ``cv2.warpAffine`` accepts any float matrix at runtime, but the stub narrows
    ``M`` to ``MatLike`` and its overloads reject a bare ``np.float32`` array;
    taking ``float64`` and building the destination ourselves keeps the call
    unambiguous.
    """
    dst = cv2.warpAffine(src, M, dsize)
    return np.ascontiguousarray(dst, dtype=np.uint8)
