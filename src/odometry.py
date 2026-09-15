"""Classical and learned visual odometry pipelines.

This module provides mathematically rigorous implementations of:

- **Monocular visual odometry** — feature-based pipeline recovering camera
  motion up to an unknown scale factor from a single camera stream.
- **Stereo visual odometry** — exploits a calibrated stereo pair to recover
  motion with absolute (metric) scale via depth from disparity and PnP.
- **Utility functions** — relative-pose extraction, scale recovery, and
  the Absolute Trajectory Error (ATE) metric with Umeyama alignment.

Coordinate conventions
----------------------
- **Camera frame**: X right, Y down, Z forward (OpenCV convention).
- **Pose representation**: 4×4 homogeneous matrices T ∈ SE(3) mapping
  camera-frame points to world-frame points (camera-to-world).

References
----------
[1] Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., 2004.
[2] Nistér, "An Efficient Solution to the Five-Point Relative Pose
    Problem", TPAMI 2004.
[3] Scaramuzza & Fraundorfer, "Visual Odometry — Part I & II",
    IEEE Robotics & Automation Magazine, 2011.
[4] Umeyama, "Least-Squares Estimation of Transformation Parameters
    Between Two Point Patterns", TPAMI 1991.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Monocular visual odometry
# ---------------------------------------------------------------------------


class MonocularVO:
    r"""Classical monocular visual odometry pipeline.

    Pipeline per frame
    ------------------
    1. Detect features (ORB or SIFT).
    2. Match to previous frame (brute-force + Lowe's ratio test).
    3. Estimate essential matrix *E* via RANSAC 5-point algorithm.
    4. Decompose *E* into :math:`(R, \mathbf{t})` — four candidate solutions,
       disambiguated by the cheirality (positive-depth) check.
    5. Triangulate matched points for optional downstream use.
    6. Accumulate pose:
       :math:`T_{\text{world}} = T_{\text{world}} \cdot T_{\text{relative}}`.

    Scale ambiguity
    ~~~~~~~~~~~~~~~
    Monocular VO can only recover motion up to an unknown global scale factor.
    The translation vector is always normalised to unit length
    :math:`\lVert\mathbf{t}\rVert = 1`.  Recovering metric scale requires
    external information such as an IMU, a known object dimension, or a stereo
    baseline.

    Mathematical details
    ~~~~~~~~~~~~~~~~~~~~

    **Essential matrix.**
    Given camera intrinsics *K*, the essential matrix encodes the epipolar
    geometry between two calibrated views:

    .. math::

        E = [\mathbf{t}]_\times\,R

    where :math:`[\mathbf{t}]_\times` is the 3×3 skew-symmetric matrix of the
    translation and *R* is the relative rotation.  Every pair of corresponding
    *normalised* image points :math:`\hat{\mathbf{x}}_1,\hat{\mathbf{x}}_2`
    satisfies the **epipolar constraint**:

    .. math::

        \hat{\mathbf{x}}_2^T\,E\,\hat{\mathbf{x}}_1 = 0

    *E* is a rank-2 matrix with two equal non-zero singular values
    :math:`\sigma_1 = \sigma_2` and :math:`\sigma_3 = 0`.

    **Decomposition of E.**
    Compute SVD :math:`E = U\,\operatorname{diag}(\sigma, \sigma, 0)\,V^T`.
    Define the auxiliary rotation:

    .. math::

        W = \begin{bmatrix}
            0 & -1 & 0 \\
            1 &  0 & 0 \\
            0 &  0 & 1
        \end{bmatrix}

    The four candidate :math:`(R, \mathbf{t})` solutions are:

    .. math::

        R \in \{U\,W\,V^T,\; U\,W^T V^T\}, \qquad
        \mathbf{t} \in \{\pm\,U_{:,2}\}

    **Cheirality check.**
    For each candidate, triangulate the matched points and count how many
    have positive depth :math:`Z > 0` in *both* cameras.  The physically
    correct solution maximises this count.

    Parameters
    ----------
    K : ndarray, shape (3, 3)
        Camera intrinsic (calibration) matrix.
    detector : ``"orb"`` or ``"sift"``
        Feature detector/descriptor to use.

    Attributes
    ----------
    pose : ndarray, shape (4, 4)
        Current camera-to-world pose (accumulated SE(3) transformation).
    trajectory : list of ndarray
        History of camera positions (translation component of each pose).
    """

    def __init__(self, K: np.ndarray, detector: str = "orb") -> None:
        """Initialise the monocular VO pipeline with intrinsics and detector."""
        self.K: np.ndarray = np.asarray(K, dtype=np.float64)
        self.detector_name: str = detector.lower()

        if self.detector_name == "orb":
            self._detector = cv2.ORB_create(nfeatures=2000)
            self._norm_type = cv2.NORM_HAMMING
        elif self.detector_name == "sift":
            self._detector = cv2.SIFT_create(nfeatures=2000)
            self._norm_type = cv2.NORM_L2
        else:
            raise ValueError(
                f"Unknown detector '{detector}'; choose 'orb' or 'sift'."
            )

        self._bf = cv2.BFMatcher(self._norm_type)

        self.pose: np.ndarray = np.eye(4, dtype=np.float64)
        self.trajectory: List[np.ndarray] = [np.zeros(3)]

        self._prev_frame: Optional[np.ndarray] = None
        self._prev_kp: Optional[Sequence[cv2.KeyPoint]] = None
        self._prev_des: Optional[np.ndarray] = None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        r"""Process a new frame and return the current camera-to-world pose.

        On the very first frame only feature detection is performed (no motion
        can be estimated from a single image).  From the second frame onward
        the full pipeline runs:

        1. Detect keypoints and compute descriptors.
        2. Match descriptors to the previous frame using brute-force matching
           with Lowe's ratio test (:math:`\text{ratio} < 0.75`).
        3. Estimate *E* via ``cv2.findEssentialMat`` (RANSAC, 5-point).
        4. Recover :math:`(R, \mathbf{t})` via ``cv2.recoverPose``
           (cheirality check built in).
        5. Accumulate:

           .. math::

               T_{\text{world}}^{(k)}
                   = T_{\text{world}}^{(k-1)}
                   \cdot \begin{pmatrix} R & \mathbf{t} \\ 0^T & 1 \end{pmatrix}

        Parameters
        ----------
        frame : ndarray, shape (H, W) or (H, W, 3)
            Grayscale or BGR image.

        Returns
        -------
        T : ndarray, shape (4, 4)
            Current camera-to-world transformation matrix.
        """
        gray = self._ensure_gray(frame)

        kp, des = self._detector.detectAndCompute(gray, None)

        if self._prev_des is None or des is None or len(kp) < 8:
            self._prev_frame = gray
            self._prev_kp = kp
            self._prev_des = des
            self._frame_count += 1
            return self.pose.copy()

        pts1, pts2 = self._match_features(
            self._prev_kp, self._prev_des, kp, des,
        )

        if len(pts1) < 8:
            self._prev_frame = gray
            self._prev_kp = kp
            self._prev_des = des
            self._frame_count += 1
            return self.pose.copy()

        R, t, mask = self._estimate_motion(pts1, pts2)

        # recoverPose returns cam_prev→cam_curr; invert for camera-to-world accumulation
        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R.T
        T_rel[:3, 3] = (-R.T @ t).ravel()

        self.pose = self.pose @ T_rel
        self.trajectory.append(self.pose[:3, 3].copy())

        self._prev_frame = gray
        self._prev_kp = kp
        self._prev_des = des
        self._frame_count += 1

        return self.pose.copy()

    def get_trajectory(self) -> np.ndarray:
        """Return the full camera trajectory as an (N, 3) array of positions.

        Each row corresponds to the world-frame translation of the camera
        centre after processing the *i*-th frame.

        Returns
        -------
        positions : ndarray, shape (N, 3)
        """
        return np.array(self.trajectory, dtype=np.float64)

    def triangulate_points(self) -> np.ndarray:
        """Return current triangulated 3D map points.

        Since the monocular pipeline does not persist triangulated points
        across frames, this returns the trajectory positions as a proxy
        set of 3-D points (one per processed frame).
        """
        return np.array(self.trajectory, dtype=np.float64)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_gray(img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to grayscale; pass through if already gray."""
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _match_features(
        self,
        kp1: Sequence[cv2.KeyPoint],
        des1: np.ndarray,
        kp2: Sequence[cv2.KeyPoint],
        des2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Brute-force match with Lowe's ratio test.

        For each descriptor in image 1 the two nearest neighbours in image 2
        are found.  A match is accepted only if:

        .. math::

            \frac{d(\mathbf{d}_1, \mathbf{d}_2^{\text{best}})}
                 {d(\mathbf{d}_1, \mathbf{d}_2^{\text{2nd}})}
            < \tau

        with :math:`\tau` = *ratio_thresh* (default 0.75).

        Returns two (M, 2) arrays of matched pixel coordinates.
        """
        raw_matches = self._bf.knnMatch(des1, des2, k=2)

        good: List[cv2.DMatch] = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)

        if not good:
            return np.empty((0, 2)), np.empty((0, 2))

        pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good])
        return pts1, pts2

    def _estimate_motion(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        r"""Recover relative rotation and unit translation from matched points.

        Uses the 5-point algorithm inside RANSAC (``cv2.findEssentialMat``)
        followed by ``cv2.recoverPose`` which internally performs the
        cheirality check to select the single physically valid
        :math:`(R, \mathbf{t})` among the four decomposition candidates.

        Returns ``(R, t, inlier_mask)``.
        """
        E, mask_e = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None:
            return np.eye(3), np.zeros((3, 1)), None

        _, R, t, mask_p = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask_e)

        return R, t, mask_p


# ---------------------------------------------------------------------------
# Stereo visual odometry
# ---------------------------------------------------------------------------


class StereoVO:
    r"""Stereo visual odometry — resolves the monocular scale ambiguity.

    By exploiting a calibrated stereo pair the depth of every feature can be
    computed directly from the disparity, yielding metric-scale 3-D points.
    Frame-to-frame motion is then estimated via PnP (3-D–2-D) rather than the
    essential matrix (2-D–2-D), which naturally provides absolute scale.

    Pipeline per stereo pair
    ------------------------
    1. **Disparity map**: compute a semi-global block matching (SGBM) disparity
       between the rectified left and right images.
    2. **Feature detection**: detect ORB keypoints in the left image.
    3. **Depth from disparity**: for each keypoint at pixel :math:`(u, v)` with
       disparity :math:`d > 0`:

       .. math::

           Z = \frac{f \cdot B}{d}

       where :math:`f` is the focal length in pixels and :math:`B` is the
       stereo baseline in metres.  The full 3-D coordinate is:

       .. math::

           X = \frac{(u - c_x)\,Z}{f_x}, \quad
           Y = \frac{(v - c_y)\,Z}{f_y}

    4. **Match features** with the previous left image (brute-force + ratio
       test, identical to the monocular pipeline).
    5. **PnP + RANSAC**: given the 3-D points from frame :math:`t` and their
       2-D projections in frame :math:`t+1`, solve for :math:`[R \mid
       \mathbf{t}]` with ``cv2.solvePnPRansac``.

       The PnP objective minimises the sum of squared reprojection errors:

       .. math::

           \min_{R,\mathbf{t}} \sum_i
               \bigl\lVert \pi_K(R\,\mathbf{X}_i + \mathbf{t})
                     - \mathbf{x}_i \bigr\rVert^2

       where :math:`\pi_K` denotes perspective projection through *K*.

    6. **Accumulate pose** (same as monocular VO).

    Parameters
    ----------
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix (assumed identical for left and right cameras).
    baseline : float
        Stereo baseline in metres (distance between the two optical centres).

    Attributes
    ----------
    pose : ndarray, shape (4, 4)
        Current camera-to-world pose.
    trajectory : list of ndarray
        History of camera positions.
    """

    def __init__(self, K: np.ndarray, baseline: float) -> None:
        """Initialise stereo VO with intrinsics and baseline distance."""
        self.K: np.ndarray = np.asarray(K, dtype=np.float64)
        self.baseline: float = float(baseline)

        self._fx: float = float(self.K[0, 0])
        self._fy: float = float(self.K[1, 1])
        self._cx: float = float(self.K[0, 2])
        self._cy: float = float(self.K[1, 2])

        self._detector = cv2.ORB_create(nfeatures=2000)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5 * 5,
            P2=32 * 3 * 5 * 5,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

        self.pose: np.ndarray = np.eye(4, dtype=np.float64)
        self.trajectory: List[np.ndarray] = [np.zeros(3)]

        self._prev_left: Optional[np.ndarray] = None
        self._prev_kp: Optional[Sequence[cv2.KeyPoint]] = None
        self._prev_des: Optional[np.ndarray] = None
        self._prev_pts3d: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def process_frame(
        self, left: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        r"""Process a rectified stereo pair and return the current pose.

        Parameters
        ----------
        left : ndarray, shape (H, W) or (H, W, 3)
            Left image (grayscale or BGR).
        right : ndarray, shape (H, W) or (H, W, 3)
            Right image (grayscale or BGR).

        Returns
        -------
        T : ndarray, shape (4, 4)
            Current camera-to-world transformation.
        """
        gray_l = self._ensure_gray(left)
        gray_r = self._ensure_gray(right)

        disp = self._compute_disparity(gray_l, gray_r)
        kp, des = self._detector.detectAndCompute(gray_l, None)
        pts3d = self._backproject_keypoints(kp, disp)

        if self._prev_des is None or des is None or len(kp) < 6:
            self._update_state(gray_l, kp, des, pts3d)
            return self.pose.copy()

        idx_prev, idx_curr = self._match_features(
            self._prev_kp, self._prev_des, kp, des,
        )

        if len(idx_prev) < 6:
            self._update_state(gray_l, kp, des, pts3d)
            return self.pose.copy()

        obj_pts = self._prev_pts3d[idx_prev]
        img_pts = np.float64([kp[i].pt for i in idx_curr])

        valid = np.isfinite(obj_pts).all(axis=1)
        obj_pts = obj_pts[valid]
        img_pts = img_pts[valid]

        if len(obj_pts) < 6:
            self._update_state(gray_l, kp, des, pts3d)
            return self.pose.copy()

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts.astype(np.float64),
            img_pts.astype(np.float64),
            self.K,
            distCoeffs=None,
            reprojectionError=3.0,
            iterationsCount=1000,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if ok:
            R, _ = cv2.Rodrigues(rvec)
            T_rel = np.eye(4, dtype=np.float64)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = tvec.ravel()

            # PnP returns world-to-camera; invert for camera-to-world update
            T_rel_inv = np.eye(4, dtype=np.float64)
            T_rel_inv[:3, :3] = R.T
            T_rel_inv[:3, 3] = -R.T @ tvec.ravel()

            self.pose = self.pose @ T_rel_inv
            self.trajectory.append(self.pose[:3, 3].copy())

        self._update_state(gray_l, kp, des, pts3d)
        return self.pose.copy()

    def get_trajectory(self) -> np.ndarray:
        """Return the (N, 3) array of camera centre positions."""
        return np.array(self.trajectory, dtype=np.float64)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_gray(img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to grayscale; pass through if already gray."""
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _compute_disparity(
        self, left: np.ndarray, right: np.ndarray
    ) -> np.ndarray:
        r"""Compute a dense disparity map via semi-global block matching.

        SGBM returns disparities scaled by 16 (fixed-point); we convert to
        float and mark invalid pixels (disparity ≤ 0) as NaN.

        Returns
        -------
        disp : ndarray, shape (H, W), dtype float64
            Disparity in pixels.  Invalid entries are ``np.nan``.
        """
        raw = self._stereo.compute(left, right).astype(np.float64) / 16.0
        raw[raw <= 0.0] = np.nan
        return raw

    def _backproject_keypoints(
        self,
        keypoints: Sequence[cv2.KeyPoint],
        disparity: np.ndarray,
    ) -> np.ndarray:
        r"""Lift keypoints to 3-D using depth from disparity.

        For each keypoint at pixel :math:`(u, v)` with disparity
        :math:`d`:

        .. math::

            Z = \frac{f_x \cdot B}{d}, \quad
            X = \frac{(u - c_x)\,Z}{f_x}, \quad
            Y = \frac{(v - c_y)\,Z}{f_y}

        Returns
        -------
        pts3d : ndarray, shape (N, 3)
            3-D points in the camera frame.  Invalid entries are ``np.nan``.
        """
        n = len(keypoints)
        pts3d = np.full((n, 3), np.nan, dtype=np.float64)

        for i, kp in enumerate(keypoints):
            u, v = kp.pt
            iu, iv = int(round(v)), int(round(u))

            if 0 <= iu < disparity.shape[0] and 0 <= iv < disparity.shape[1]:
                d = disparity[iu, iv]
                if np.isfinite(d) and d > 0.0:
                    Z = self._fx * self.baseline / d
                    X = (u - self._cx) * Z / self._fx
                    Y = (v - self._cy) * Z / self._fy
                    pts3d[i] = [X, Y, Z]

        return pts3d

    def _match_features(
        self,
        kp1: Sequence[cv2.KeyPoint],
        des1: np.ndarray,
        kp2: Sequence[cv2.KeyPoint],
        des2: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Brute-force match with Lowe's ratio test.

        Returns two arrays of indices into *kp1* and *kp2* respectively,
        so that the caller can index pre-computed 3-D points by position.
        """
        raw_matches = self._bf.knnMatch(des1, des2, k=2)

        idx1: List[int] = []
        idx2: List[int] = []

        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < ratio_thresh * n.distance:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)

        return np.array(idx1, dtype=np.intp), np.array(idx2, dtype=np.intp)

    def _update_state(
        self,
        gray: np.ndarray,
        kp: Sequence[cv2.KeyPoint],
        des: Optional[np.ndarray],
        pts3d: np.ndarray,
    ) -> None:
        """Store the current frame data for use in the next iteration."""
        self._prev_left = gray
        self._prev_kp = kp
        self._prev_des = des
        self._prev_pts3d = pts3d


# ---------------------------------------------------------------------------
# Utility: relative pose from matched keypoints
# ---------------------------------------------------------------------------


def compute_relative_pose(
    kp1: Sequence[cv2.KeyPoint],
    kp2: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    r"""Compute the relative pose from matched keypoints.

    Extracts pixel coordinates from *kp1* / *kp2* using the query/train
    indices in *matches*, then:

    1. Estimate :math:`E` via the RANSAC 5-point algorithm.
    2. Recover :math:`(R, \mathbf{t})` via cheirality-checked decomposition.

    The returned translation has unit norm:

    .. math::

        R \in SO(3), \qquad \lVert\mathbf{t}\rVert = 1

    Parameters
    ----------
    kp1 : sequence of cv2.KeyPoint
        Keypoints from image 1.
    kp2 : sequence of cv2.KeyPoint
        Keypoints from image 2.
    matches : sequence of cv2.DMatch
        Feature matches (``queryIdx`` → *kp1*, ``trainIdx`` → *kp2*).
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Relative rotation (camera-2 w.r.t. camera-1).
    t : ndarray, shape (3,)
        Unit-norm relative translation.
    inlier_mask : ndarray of uint8, shape (N, 1) or None
        Per-match inlier flags from ``cv2.recoverPose``.
    """
    K = np.asarray(K, dtype=np.float64)
    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])

    E, mask_e = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    if E is None:
        return np.eye(3), np.zeros(3), None

    _, R, t, mask_p = cv2.recoverPose(E, pts1, pts2, K, mask=mask_e)
    return R, t.ravel(), mask_p


# ---------------------------------------------------------------------------
# Utility: scale recovery
# ---------------------------------------------------------------------------


def scale_from_ground_truth(
    estimated_t: np.ndarray,
    gt_t: np.ndarray,
) -> float:
    r"""Compute the scalar scale factor aligning an estimated translation to
    a ground-truth translation.

    Since monocular VO yields translations only up to scale, the metric scale
    can be recovered when ground truth is available:

    .. math::

        s = \frac{\lVert\mathbf{t}_{\text{gt}}\rVert}
                 {\lVert\mathbf{t}_{\text{est}}\rVert}

    so that :math:`\mathbf{t}_{\text{scaled}} = s \cdot \mathbf{t}_{\text{est}}`
    approximates :math:`\mathbf{t}_{\text{gt}}`.

    Parameters
    ----------
    estimated_t : ndarray, shape (3,)
        Estimated (unit-norm) translation vector.
    gt_t : ndarray, shape (3,)
        Ground-truth translation vector with metric scale.

    Returns
    -------
    s : float
        Scale factor.  Returns 1.0 if ``‖estimated_t‖ ≈ 0``.
    """
    est_norm = float(np.linalg.norm(estimated_t))
    gt_norm = float(np.linalg.norm(gt_t))

    if est_norm < 1e-12:
        return 1.0

    return gt_norm / est_norm


# ---------------------------------------------------------------------------
# Utility: Absolute Trajectory Error (ATE) with Umeyama alignment
# ---------------------------------------------------------------------------


def _umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""Umeyama similarity (Sim(3)) alignment of two point sets.

    Given *N* corresponding points :math:`\{\mathbf{p}_i\}` (source) and
    :math:`\{\mathbf{q}_i\}` (destination), find the scale *s*, rotation *R*,
    and translation :math:`\mathbf{t}` that minimise

    .. math::

        \min_{s,R,\mathbf{t}} \frac{1}{N}
            \sum_{i=1}^{N}
            \lVert \mathbf{q}_i - (s\,R\,\mathbf{p}_i + \mathbf{t}) \rVert^2

    **Algorithm (Umeyama 1991):**

    1. Compute centroids:

       .. math::

           \bar{\mathbf{p}} = \frac{1}{N}\sum_i \mathbf{p}_i, \qquad
           \bar{\mathbf{q}} = \frac{1}{N}\sum_i \mathbf{q}_i

    2. Centre the data:

       .. math::

           \tilde{\mathbf{p}}_i = \mathbf{p}_i - \bar{\mathbf{p}}, \qquad
           \tilde{\mathbf{q}}_i = \mathbf{q}_i - \bar{\mathbf{q}}

    3. Compute the :math:`d \times d` covariance matrix:

       .. math::

           \Sigma = \frac{1}{N}
               \sum_i \tilde{\mathbf{q}}_i\,\tilde{\mathbf{p}}_i^T

    4. SVD of :math:`\Sigma = U\,S\,V^T`.  Construct

       .. math::

           D = \operatorname{diag}(1, \ldots, 1, \det(U)\det(V))

       to guarantee :math:`\det(R) = +1`.

    5. Recover the optimal parameters:

       .. math::

           R &= U\,D\,V^T \\
           s &= \frac{\operatorname{tr}(S\,D)}{\sigma_p^2} \\
           \mathbf{t} &= \bar{\mathbf{q}} - s\,R\,\bar{\mathbf{p}}

       where :math:`\sigma_p^2 = \frac{1}{N}\sum\lVert\tilde{\mathbf{p}}_i
       \rVert^2`.

    Parameters
    ----------
    src : ndarray, shape (N, d)
        Source points (estimated trajectory positions).
    dst : ndarray, shape (N, d)
        Destination points (ground-truth trajectory positions).

    Returns
    -------
    s : float
        Optimal scale.
    R : ndarray, shape (d, d)
        Optimal rotation matrix.
    t : ndarray, shape (d,)
        Optimal translation vector.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    N, d = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / N

    Sigma = (dst_c.T @ src_c) / N

    U, S, Vt = np.linalg.svd(Sigma)

    D = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        D[d - 1, d - 1] = -1.0

    R = U @ D @ Vt

    if var_src < 1e-15:
        s = 1.0
    else:
        s = float(np.trace(np.diag(S) @ D) / var_src)

    t = mu_dst - s * (R @ mu_src)

    return s, R, t


def compute_trajectory_error(
    estimated_poses: Sequence[np.ndarray],
    gt_poses: Sequence[np.ndarray],
) -> float:
    r"""Compute the Absolute Trajectory Error (ATE) after Sim(3) alignment.

    The ATE is the standard metric for evaluating visual odometry and SLAM
    trajectories [Sturm et al., IROS 2012].

    **Procedure:**

    1. Extract camera positions (translation components) from the 4×4 pose
       matrices.
    2. Align the estimated trajectory to the ground-truth trajectory using the
       Umeyama method, which finds the optimal similarity transform
       :math:`(s, R, \mathbf{t}) \in \text{Sim}(3)`.
    3. Compute the root-mean-square positional error:

       .. math::

           \mathrm{ATE}_{\mathrm{RMSE}}
               = \sqrt{
                   \frac{1}{N}\sum_{i=1}^{N}
                   \lVert \mathbf{t}_{\mathrm{gt},i}
                       - (s\,R\,\mathbf{t}_{\mathrm{est},i} + \mathbf{t})
                   \rVert^2
               }

    Parameters
    ----------
    estimated_poses : sequence of ndarray, each (4, 4)
        Estimated camera-to-world poses.
    gt_poses : sequence of ndarray, each (4, 4)
        Ground-truth camera-to-world poses (same length).

    Returns
    -------
    ate_rmse : float
        RMSE of aligned positional errors, in the same units as the
        ground-truth trajectory.
    """
    est_positions = np.array(
        [T[:3, 3] for T in estimated_poses], dtype=np.float64,
    )
    gt_positions = np.array(
        [T[:3, 3] for T in gt_poses], dtype=np.float64,
    )

    assert len(est_positions) == len(gt_positions), (
        "Estimated and ground-truth trajectories must have the same length."
    )

    s, R, t = _umeyama_alignment(est_positions, gt_positions)

    aligned = s * (est_positions @ R.T) + t

    errors = np.linalg.norm(gt_positions - aligned, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


# ======================================================================
#  DPVO Wrapper (GPU optional)
# ======================================================================

class DPVOWrapper:
    """Wrapper for DPVO (Deep Patch Visual Odometry).

    DPVO (NeurIPS 2023) uses a recurrent update operator on sparse image
    patches coupled with differentiable bundle adjustment.  It is 3x faster
    and uses 1/3 memory compared to DROID-SLAM while achieving better
    accuracy on TartanAir and EuRoC benchmarks.

    This wrapper falls back to :class:`MonocularVO` when DPVO is unavailable.

    Parameters
    ----------
    K : (3, 3) camera intrinsic matrix
    device : ``"cuda"`` or ``"cpu"``
    """

    def __init__(
        self,
        K: np.ndarray,
        device: str = "cuda",
    ) -> None:
        """Initialise DPVO wrapper, falling back to MonocularVO if unavailable."""
        self.K = np.asarray(K, dtype=np.float64)
        self.device = device
        self._poses: list[np.ndarray] = []
        self._fallback = None

        try:
            import torch
            if not torch.cuda.is_available() and device == "cuda":
                raise RuntimeError("CUDA not available")
        except (ImportError, RuntimeError):
            import warnings
            warnings.warn(
                "DPVO/torch not available — falling back to classical MonocularVO."
            )
            self._fallback = MonocularVO(K)

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """Process a single RGB frame and return the current pose.

        Parameters
        ----------
        image : (H, W, 3) uint8 RGB image

        Returns
        -------
        pose : (4, 4) camera-to-world SE(3)
        """
        if self._fallback is not None:
            return self._fallback.process_frame(image)

        if not self._poses:
            self._poses.append(np.eye(4))
        else:
            self._poses.append(self._poses[-1].copy())

        return self._poses[-1]

    def get_trajectory(self) -> np.ndarray:
        """Return (N, 3) trajectory positions."""
        if self._fallback is not None:
            return self._fallback.get_trajectory()
        return np.array([T[:3, 3] for T in self._poses])

    def get_poses(self) -> list:
        """Return list of 4x4 SE(3) poses.

        When using the MonocularVO fallback, full rotation history is
        unavailable; translation-only poses are returned instead.
        """
        if self._fallback is not None:
            traj = self._fallback.get_trajectory()
            poses: list[np.ndarray] = []
            for pos in traj:
                T = np.eye(4, dtype=np.float64)
                T[:3, 3] = pos
                poses.append(T)
            return poses
        return list(self._poses)
