"""Structure from Motion — incremental 3-D reconstruction from images.

This module implements an incremental SfM pipeline that recovers both the
3-D structure of a scene and the camera poses from a collection of
unordered images.

Algorithm overview
------------------
1. **Feature detection + matching** across all image pairs.
2. **Initialise** from the best pair (most inliers, sufficient baseline).
3. For each new image:
   a. Find 2-D → 3-D correspondences (observed features matching existing
      3-D points).
   b. Estimate the camera pose via **PnP-RANSAC**.
   c. **Triangulate** new 3-D points visible in this and any registered view.
   d. Run **bundle adjustment** to jointly refine cameras and points.
4. Repeat until all images are registered.

Mathematical background
-----------------------

**Perspective-n-Point (PnP)**

Given *n* 3-D ↔ 2-D correspondences :math:`(\\mathbf{X}_j, \\mathbf{x}_j)`,
PnP minimises the reprojection error:

.. math::

    \\min_{R,\\mathbf{t}} \\sum_j
    \\bigl\\lVert \\pi\\!\\bigl(K [R \\mid \\mathbf{t}]\\,\\mathbf{X}_j\\bigr)
    - \\mathbf{x}_j \\bigr\\rVert^2

where :math:`\\pi` projects a 3-D homogeneous point to its 2-D image
coordinates.

**Bundle Adjustment (BA)**

BA jointly refines all cameras and 3-D points by minimising:

.. math::

    \\min_{\\{R_i,\\mathbf{t}_i\\}, \\{\\mathbf{X}_j\\}}
    \\sum_i \\sum_j
    \\bigl\\lVert \\pi\\!\\bigl(K_i [R_i \\mid \\mathbf{t}_i]\\,\\mathbf{X}_j\\bigr)
    - \\mathbf{x}_{ij} \\bigr\\rVert^2

This is a large sparse non-linear least-squares problem, solved here with
:func:`scipy.optimize.least_squares` and its Levenberg–Marquardt back-end.

References
----------
[1] Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., 2004.
[2] Triggs et al., "Bundle Adjustment — A Modern Synthesis", 2000.
[3] Schönberger & Frahm, "Structure-from-Motion Revisited", CVPR 2016.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from ._cv import solve_pnp_ransac
from .features import (
    choose_pose_cheirality,
    compute_essential,
    decompose_essential,
    ransac_fundamental,
    triangulate_dlt,
)
from .transforms import se3_from_Rt, se3_inverse

# ======================================================================
#  Type aliases
# ======================================================================
Mat3 = NDArray[np.floating]
Mat4 = NDArray[np.floating]
Points2D = NDArray[np.floating]
Points3D = NDArray[np.floating]


# ======================================================================
#  Data containers
# ======================================================================


@dataclass
class ImageData:
    """Feature and match data associated with a single image."""

    keypoints: NDArray[np.float64]  # (N, 2) keypoint pixel coordinates
    descriptors: NDArray[np.floating] | None = None  # (N, D) descriptor array
    # Mapping from local keypoint idx to global 3-D point idx (-1 = unassigned)
    point_indices: NDArray[np.intp] = field(
        default_factory=lambda: np.array([], dtype=np.intp)
    )


@dataclass
class MatchData:
    """Pairwise match data between two images."""

    img1_id: int
    img2_id: int
    idx1: NDArray[np.intp]  # indices into img1 keypoints
    idx2: NDArray[np.intp]  # indices into img2 keypoints
    inlier_mask: NDArray[np.bool_] | None = None


# ======================================================================
#  Bundle-adjustment helpers
# ======================================================================


def _ba_pack_parameters(
    cameras: dict[int, NDArray[np.float64]],
    cam_ids: list[int],
    cam_id_to_idx: dict[int, int],
    points_3d: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Pack camera and point parameters into a single flat vector."""
    n_cameras = len(cam_ids)
    cam_params = np.zeros(6 * n_cameras, dtype=np.float64)
    for cid in cam_ids:
        idx = cam_id_to_idx[cid]
        T_cw = cameras[cid]
        T_wc = se3_inverse(T_cw)
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        rvec, _ = cv2.Rodrigues(R_wc)
        cam_params[6 * idx : 6 * idx + 3] = rvec.ravel()
        cam_params[6 * idx + 3 : 6 * idx + 6] = t_wc
    point_params = np.array(points_3d, dtype=np.float64).ravel()
    return np.concatenate([cam_params, point_params])


def _ba_compute_residuals(
    x: NDArray[np.float64],
    n_cameras: int,
    n_points: int,
    obs_cam_idx: NDArray[np.intp],
    obs_pt_idx: NDArray[np.intp],
    obs_2d_arr: NDArray[np.float64],
    K: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the 2-D reprojection residual vector for all observations."""
    cams = x[: 6 * n_cameras].reshape(n_cameras, 6)
    pts = x[6 * n_cameras :].reshape(n_points, 3)

    residual_vec = np.zeros(2 * len(obs_cam_idx), dtype=np.float64)
    for k in range(len(obs_cam_idx)):
        ci = obs_cam_idx[k]
        pi = obs_pt_idx[k]
        rvec_k = cams[ci, :3]
        tvec_k = cams[ci, 3:6]
        R_k, _ = cv2.Rodrigues(rvec_k)
        X_cam = R_k @ pts[pi] + tvec_k
        if X_cam[2] < 1e-8:
            residual_vec[2 * k : 2 * k + 2] = 0.0
            continue
        proj = K @ X_cam
        proj_2d = proj[:2] / proj[2]
        residual_vec[2 * k : 2 * k + 2] = proj_2d - obs_2d_arr[k]
    return residual_vec


# ======================================================================
#  IncrementalSfM
# ======================================================================


class IncrementalSfM:
    r"""Incremental Structure from Motion pipeline.

    SfM reconstructs 3-D structure and camera poses from unordered images.

    Algorithm
    ---------
    1. Feature detection + matching across all image pairs.
    2. Initialise from the best pair (most inliers, sufficient baseline).
    3. For each new image:

       a. Find 2-D → 3-D correspondences (observed features that match
          existing 3-D points).
       b. Estimate camera pose via PnP-RANSAC.
       c. Triangulate new 3-D points.
       d. Run bundle adjustment to refine everything.

    4. Repeat until all images are registered.

    Math
    ----
    PnP minimises:

    .. math::

        \sum_j \bigl\lVert \pi(K[R \mid \mathbf{t}]\,\mathbf{X}_j)
        - \mathbf{x}_j \bigr\rVert^2

    BA minimises:

    .. math::

        \sum_i \sum_j \bigl\lVert
        \pi(K_i[R_i \mid \mathbf{t}_i]\,\mathbf{X}_j)
        - \mathbf{x}_{ij} \bigr\rVert^2

    Parameters
    ----------
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix (assumed constant across all views).
    """

    def __init__(self, K: NDArray[np.float64]) -> None:
        """Initialise the SfM pipeline with a shared camera intrinsic matrix."""
        self.K: NDArray[np.float64] = np.asarray(K, dtype=np.float64)

        # Registered cameras: img_id → 4×4 camera-to-world SE(3)
        self.cameras: dict[int, NDArray[np.float64]] = {}

        # Global 3-D point cloud
        self.points_3d: list[NDArray[np.float64]] = []

        # observations[img_id] → list of (kp_idx, point3d_idx)
        self.observations: dict[int, list[tuple[int, int]]] = {}

        # Per-image feature data
        self._images: dict[int, ImageData] = {}

        # Pairwise matches: (img1, img2) → MatchData
        self._matches: dict[tuple[int, int], MatchData] = {}

        self._registered: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_image(
        self,
        img_id: int,
        keypoints: NDArray[np.float64],
        descriptors: NDArray[np.floating] | None = None,
    ) -> None:
        """Register image features.

        Parameters
        ----------
        img_id : int
            Unique identifier for this image.
        keypoints : ndarray, shape (N, 2)
            Detected keypoint pixel coordinates.
        descriptors : ndarray, shape (N, D), optional
            Per-keypoint descriptor vectors.
        """
        point_indices = np.full(len(keypoints), -1, dtype=np.intp)
        self._images[img_id] = ImageData(
            keypoints=np.asarray(keypoints, dtype=np.float64),
            descriptors=descriptors,
            point_indices=point_indices,
        )

    def add_image_pair(
        self,
        img1_id: int,
        img2_id: int,
        matches: Sequence[tuple[int, int]],
        kp1: NDArray[np.float64],
        kp2: NDArray[np.float64],
    ) -> None:
        """Register a matched image pair.

        Parameters
        ----------
        img1_id, img2_id : int
            Image identifiers.
        matches : sequence of (idx1, idx2)
            Feature index correspondences.
        kp1, kp2 : ndarray, shape (N, 2)
            Keypoint coordinates for each image.
        """
        if img1_id not in self._images:
            self.add_image(img1_id, kp1)
        if img2_id not in self._images:
            self.add_image(img2_id, kp2)

        idx1 = np.array([m[0] for m in matches], dtype=np.intp)
        idx2 = np.array([m[1] for m in matches], dtype=np.intp)
        self._matches[(img1_id, img2_id)] = MatchData(
            img1_id=img1_id,
            img2_id=img2_id,
            idx1=idx1,
            idx2=idx2,
        )

    def initialize(self, img1_id: int, img2_id: int) -> bool:
        r"""Initialise the reconstruction from a two-view pair.

        Steps
        -----
        1. Estimate the fundamental matrix *F* via RANSAC.
        2. Compute the essential matrix :math:`E = K^T F K`.
        3. Decompose *E* into four :math:`(R, \mathbf{t})` candidates.
        4. Select the correct solution via the cheirality check.
        5. Triangulate matched inlier points.

        Parameters
        ----------
        img1_id, img2_id : int
            Identifiers of the initialisation pair.

        Returns
        -------
        bool
            ``True`` if initialisation succeeded.
        """
        key = (img1_id, img2_id)
        if key not in self._matches:
            key = (img2_id, img1_id)
            if key not in self._matches:
                return False

        md = self._matches[key]
        id1, id2 = md.img1_id, md.img2_id

        pts1 = self._images[id1].keypoints[md.idx1]
        pts2 = self._images[id2].keypoints[md.idx2]

        if len(pts1) < 8:
            return False

        F, inlier_mask = ransac_fundamental(pts1, pts2, threshold=1.0)
        md.inlier_mask = inlier_mask

        inlier_pts1 = pts1[inlier_mask]
        inlier_pts2 = pts2[inlier_mask]
        inlier_idx1 = md.idx1[inlier_mask]
        inlier_idx2 = md.idx2[inlier_mask]

        if len(inlier_pts1) < 8:
            return False

        E = compute_essential(F, self.K)
        decompositions = decompose_essential(E)
        R_list = [d[0] for d in decompositions]
        t_list = [d[1] for d in decompositions]

        R_best, t_best = choose_pose_cheirality(
            R_list,
            t_list,
            inlier_pts1,
            inlier_pts2,
            self.K,
        )

        # Camera 1 at origin.  (R_best, t_best) maps camera-1 (world) coords
        # to camera-2 coords, so store the *inverse* to keep the documented
        # camera-to-world convention used by register_image/bundle_adjust.
        T1 = np.eye(4, dtype=np.float64)
        T2_w2c = se3_from_Rt(R_best, t_best)

        self.cameras[id1] = T1
        self.cameras[id2] = se3_inverse(T2_w2c)
        self._registered.update({id1, id2})
        self.observations[id1] = []
        self.observations[id2] = []

        # Triangulate initial points
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R_best, t_best.reshape(3, 1)])

        X = triangulate_dlt(inlier_pts1, inlier_pts2, P1, P2)

        for i in range(len(X)):
            pt_idx = len(self.points_3d)
            self.points_3d.append(X[i])

            kp_i1, kp_i2 = int(inlier_idx1[i]), int(inlier_idx2[i])
            self._images[id1].point_indices[kp_i1] = pt_idx
            self._images[id2].point_indices[kp_i2] = pt_idx
            self.observations[id1].append((kp_i1, pt_idx))
            self.observations[id2].append((kp_i2, pt_idx))

        return True

    def register_image(self, img_id: int) -> bool:
        r"""Register a new image via PnP and triangulate new points.

        Steps
        -----
        1. Collect 2-D → 3-D correspondences from already-triangulated
           points that share matches with this image.
        2. Solve PnP-RANSAC:

           .. math::

               \min_{R,\mathbf{t}} \sum_j
               \bigl\lVert \pi(K[R|\mathbf{t}]\mathbf{X}_j)
               - \mathbf{x}_j \bigr\rVert^2

        3. Triangulate new points against all registered views.

        Parameters
        ----------
        img_id : int
            Image to register.

        Returns
        -------
        bool
            ``True`` if registration succeeded.
        """
        if img_id in self._registered:
            return True

        pts_3d_list: list[NDArray[np.float64]] = []
        pts_2d_list: list[NDArray[np.float64]] = []
        kp_indices: list[int] = []

        img_data = self._images[img_id]

        for (id1, id2), md in self._matches.items():
            if id1 == img_id and id2 in self._registered:
                partner, my_idx, their_idx = id2, md.idx1, md.idx2
            elif id2 == img_id and id1 in self._registered:
                partner, my_idx, their_idx = id1, md.idx2, md.idx1
            else:
                continue

            partner_data = self._images[partner]
            for mi, ti in zip(my_idx, their_idx, strict=False):
                pt3d_idx = partner_data.point_indices[ti]
                if pt3d_idx >= 0 and img_data.point_indices[mi] < 0:
                    pts_3d_list.append(self.points_3d[pt3d_idx])
                    pts_2d_list.append(img_data.keypoints[mi])
                    kp_indices.append(int(mi))

        if len(pts_3d_list) < 6:
            return False

        obj_pts = np.array(pts_3d_list, dtype=np.float64)
        img_pts = np.array(pts_2d_list, dtype=np.float64)

        ok, rvec, tvec, inliers = solve_pnp_ransac(
            obj_pts,
            img_pts,
            np.asarray(self.K, dtype=np.float64),
            np.zeros(5, dtype=np.float64),
            reprojection_error=4.0,
            iterations=1000,
        )

        if not ok or inliers is None:
            return False

        R, _ = cv2.Rodrigues(rvec)
        T_world_to_cam = se3_from_Rt(
            np.asarray(R, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64).ravel(),
        )
        T_cam_to_world = se3_inverse(T_world_to_cam)

        self.cameras[img_id] = T_cam_to_world
        self._registered.add(img_id)
        self.observations[img_id] = []

        # Record observations for PnP inliers
        inlier_set = {int(x) for x in np.asarray(inliers).ravel().tolist()}
        for i in inlier_set:
            kp_i = kp_indices[i]
            pt3d_idx = (
                int(
                    np.where(np.all(np.array(self.points_3d) == obj_pts[i], axis=1))[0][
                        0
                    ]
                )
                if len(self.points_3d) > 0
                else -1
            )

            if pt3d_idx >= 0:
                img_data.point_indices[kp_i] = pt3d_idx
                self.observations[img_id].append((kp_i, pt3d_idx))

        # Triangulate new points against all registered partners
        self._triangulate_new_points(img_id)

        return True

    def bundle_adjust(self, max_iterations: int = 50) -> float:
        r"""Run bundle adjustment over all cameras and 3-D points.

        Parameterisation
        ----------------
        - Each camera is parameterised by a 6-vector
          :math:`[\rho_1, \rho_2, \rho_3, \omega_1, \omega_2, \omega_3]`
          (axis-angle rotation + translation) in the *world-to-camera* frame.
        - Each 3-D point is parameterised by :math:`(X, Y, Z)`.

        The residual for observation :math:`(i, j)` is the 2-D reprojection
        error:

        .. math::

            \mathbf{r}_{ij} = \pi(K [R_i | \mathbf{t}_i] \mathbf{X}_j)
            - \mathbf{x}_{ij}

        Solved with :func:`scipy.optimize.least_squares` using the ``lm``
        (Levenberg–Marquardt) method.

        Parameters
        ----------
        max_iterations : int
            Maximum number of LM iterations.

        Returns
        -------
        float
            Final mean reprojection error in pixels.
        """
        if len(self.cameras) < 2 or len(self.points_3d) < 1:
            return 0.0

        cam_ids = sorted(self.cameras.keys())
        cam_id_to_idx = {cid: i for i, cid in enumerate(cam_ids)}
        n_cameras = len(cam_ids)
        n_points = len(self.points_3d)

        # Gather all observations
        obs_cam_idx: list[int] | NDArray[np.intp] = []
        obs_pt_idx: list[int] | NDArray[np.intp] = []
        obs_2d: list[NDArray[np.float64]] = []

        for cid in cam_ids:
            for kp_idx, pt_idx in self.observations.get(cid, []):
                if pt_idx < n_points:
                    obs_cam_idx.append(cam_id_to_idx[cid])
                    obs_pt_idx.append(pt_idx)
                    obs_2d.append(self._images[cid].keypoints[kp_idx])

        if len(obs_2d) == 0:
            return 0.0

        obs_cam_idx = np.array(obs_cam_idx, dtype=np.intp)
        obs_pt_idx = np.array(obs_pt_idx, dtype=np.intp)
        obs_2d_arr = np.array(obs_2d, dtype=np.float64)

        x0 = _ba_pack_parameters(
            self.cameras,
            cam_ids,
            cam_id_to_idx,
            self.points_3d,
        )
        K = self.K

        result = least_squares(
            lambda x: _ba_compute_residuals(
                x,
                n_cameras,
                n_points,
                obs_cam_idx,
                obs_pt_idx,
                obs_2d_arr,
                K,
            ),
            x0,
            method="lm",
            max_nfev=max_iterations * len(x0),
        )

        # Unpack results
        cams_opt = result.x[: 6 * n_cameras].reshape(n_cameras, 6)
        pts_opt = result.x[6 * n_cameras :].reshape(n_points, 3)

        for cid in cam_ids:
            idx = cam_id_to_idx[cid]
            rvec = cams_opt[idx, :3]
            tvec = cams_opt[idx, 3:6]
            R_wc, _ = cv2.Rodrigues(rvec)
            T_wc = se3_from_Rt(R_wc, tvec)
            self.cameras[cid] = se3_inverse(T_wc)

        self.points_3d = [pts_opt[i] for i in range(n_points)]

        mean_reproj = float(np.sqrt(np.mean(result.fun**2)))
        return mean_reproj

    def get_reconstruction(
        self,
    ) -> tuple[dict[int, NDArray[np.float64]], NDArray[np.float64]]:
        """Return the current reconstruction.

        Returns
        -------
        cameras : dict
            Mapping from image id to 4×4 camera-to-world SE(3) pose.
        points_3d : ndarray, shape (M, 3)
            Reconstructed 3-D point cloud.
        """
        pts = (
            np.array(self.points_3d, dtype=np.float64)
            if self.points_3d
            else np.zeros((0, 3))
        )
        return dict(self.cameras), pts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _triangulate_new_points(self, img_id: int) -> None:
        """Triangulate new 3-D points between *img_id* and registered views."""
        img_data = self._images[img_id]
        T_this = self.cameras[img_id]
        T_this_inv = se3_inverse(T_this)
        P_this = self.K @ T_this_inv[:3]

        for (id1, id2), md in self._matches.items():
            if id1 == img_id and id2 in self._registered:
                partner, my_idx, their_idx = id2, md.idx1, md.idx2
            elif id2 == img_id and id1 in self._registered:
                partner, my_idx, their_idx = id1, md.idx2, md.idx1
            else:
                continue

            partner_data = self._images[partner]
            T_partner = self.cameras[partner]
            T_partner_inv = se3_inverse(T_partner)
            P_partner = self.K @ T_partner_inv[:3]

            for mi, ti in zip(my_idx, their_idx, strict=False):
                if img_data.point_indices[mi] >= 0:
                    continue
                if partner_data.point_indices[ti] >= 0:
                    continue

                pt1 = img_data.keypoints[mi : mi + 1]
                pt2 = partner_data.keypoints[ti : ti + 1]

                X = triangulate_dlt(pt1, pt2, P_this, P_partner)

                # Cheirality: must be in front of both cameras
                X_cam_this = T_this_inv[:3, :3] @ X[0] + T_this_inv[:3, 3]
                X_cam_partner = T_partner_inv[:3, :3] @ X[0] + T_partner_inv[:3, 3]
                if X_cam_this[2] <= 0 or X_cam_partner[2] <= 0:
                    continue

                pt_idx = len(self.points_3d)
                self.points_3d.append(X[0])

                img_data.point_indices[mi] = pt_idx
                partner_data.point_indices[ti] = pt_idx
                self.observations[img_id].append((int(mi), pt_idx))
                self.observations.setdefault(partner, []).append((int(ti), pt_idx))

    # -- Plan-compatible aliases --

    def add_images(
        self,
        images: Sequence[tuple[int, NDArray[np.float64], NDArray[np.floating] | None]],
    ) -> None:
        """Add multiple ``(img_id, keypoints[, descriptors])`` entries at once."""
        for img in images:
            self.add_image(*img)

    def register_next_image(self) -> bool:
        """Register the next unregistered image with the most 3D-2D matches."""
        unregistered = [i for i in self._images if i not in self._registered]
        if not unregistered:
            return False
        best_id = max(
            unregistered,
            key=lambda i: int(np.sum(self._images[i].point_indices >= 0)),
        )
        return self.register_image(best_id)

    def triangulate_new_points(self) -> None:
        """Triangulate new points from the most recently registered image."""
        if self._registered:
            self._triangulate_new_points(max(self._registered))

    def reconstruct_all(
        self,
    ) -> tuple[dict[int, NDArray[np.float64]], NDArray[np.float64]]:
        """Run full incremental SfM: initialize + register all + BA."""
        ids = sorted(self._images.keys())
        if len(ids) < 2:
            return self.get_reconstruction()
        self.initialize(ids[0], ids[1])
        while self.register_next_image():
            self.bundle_adjust(max_iterations=10)
        self.bundle_adjust(max_iterations=50)
        return self.get_reconstruction()
