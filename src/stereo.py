"""
Stereo vision: rectification, disparity estimation, and depth reconstruction.

The stereo depth pipeline:
    1. Rectify image pair → epipolar lines become horizontal
    2. Compute disparity map d(x,y) via block matching
    3. Convert disparity to depth: Z = f·B / d
    4. Back-project to 3D point cloud using camera intrinsics
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def rectify_stereo_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    dist1: np.ndarray,
    dist2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rectify a stereo pair so epipolar lines are horizontal.

    After rectification, corresponding points lie on the same row.
    This reduces 2D search to 1D horizontal scan, enabling fast disparity
    computation.

    **Math**:

    Given extrinsics (R, T) between cameras, stereo rectification finds
    rotation matrices R₁, R₂ that rotate each camera so:

    - Both image planes become coplanar
    - Epipoles are sent to infinity (pure horizontal epipolar lines)
    - The new x-axis aligns with the baseline direction: e₁ = T / ‖T‖

    The rectification homographies are:

        H₁ = K₁ · R₁ · K₁⁻¹
        H₂ = K₂ · R₂ · K₂⁻¹

    After applying H₁, H₂ the fundamental matrix becomes:

        F_rect = [e']ₓ = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]

    meaning every epipolar line is a horizontal scanline.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Left and right images (BGR or grayscale).
    K1, K2 : np.ndarray
        3×3 intrinsic matrices for each camera.
    dist1, dist2 : np.ndarray
        Distortion coefficients for each camera.
    R : np.ndarray
        3×3 rotation matrix from camera 1 to camera 2.
    T : np.ndarray
        3×1 translation vector from camera 1 to camera 2.
    image_size : tuple of int
        (width, height) of the images.

    Returns
    -------
    rect1, rect2 : np.ndarray
        Rectified left and right images.
    Q : np.ndarray
        4×4 disparity-to-depth reprojection matrix used by
        ``cv2.reprojectImageTo3D``:
            [X, Y, Z, W]ᵀ = Q · [x, y, d, 1]ᵀ
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist1, K2, dist2, image_size, R, T, alpha=0,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, dist1, R1, P1, image_size, cv2.CV_32FC1,
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, dist2, R2, P2, image_size, cv2.CV_32FC1,
    )

    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    return rect1, rect2, Q


def compute_disparity_sgbm(
    left: np.ndarray,
    right: np.ndarray,
    num_disparities: int = 128,
    block_size: int = 5,
) -> np.ndarray:
    """
    Semi-Global Block Matching (SGBM) for dense disparity estimation.

    Disparity ``d(x,y) = x_left − x_right`` for corresponding pixels.

    **Energy function** minimised by SGBM:

        E(D) = Σ_p [ C(p, D(p))
                      + Σ_{q ∈ N(p)} P₁ · 𝟙[|D(p)−D(q)| = 1]
                      + Σ_{q ∈ N(p)} P₂ · 𝟙[|D(p)−D(q)| > 1] ]

    where:

    - C(p, d): pixel-wise matching cost (Census transform / absolute diff)
    - P₁: penalty for 1-pixel disparity change (smooth surfaces)
    - P₂: penalty for larger jumps (preserves depth discontinuities, P₂ > P₁)
    - N(p): 4-connected neighbourhood of pixel p
    - Aggregation runs along 8 directions (semi-global), then the minimum
      cost disparity is selected per pixel.

    The full-global optimum of this MRF energy is NP-hard; SGBM approximates
    it by aggregating 1D dynamic-programming solutions along multiple paths.

    Parameters
    ----------
    left, right : np.ndarray
        Rectified left and right images (grayscale uint8).
    num_disparities : int
        Maximum disparity range; must be divisible by 16.
    block_size : int
        Matched block size. Must be odd, ≥ 1.

    Returns
    -------
    disparity : np.ndarray
        Float32 disparity map in pixels. Invalid regions are set to 0.
    """
    if left.ndim == 3:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    if right.ndim == 3:
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    num_disparities = max(16, (num_disparities // 16) * 16)

    p1 = 8 * block_size * block_size
    p2 = 32 * block_size * block_size

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    raw = sgbm.compute(left, right)
    disparity = raw.astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.0
    return disparity


def disparity_to_depth(
    disparity: np.ndarray,
    focal_length: float,
    baseline: float,
) -> np.ndarray:
    """
    Convert a disparity map to metric depth via the stereo equation.

    **Stereo depth equation**::

        Z = f · B / d

    where:

    - Z  = depth in metres
    - f  = focal length in pixels
    - B  = stereo baseline in metres (distance between optical centres)
    - d  = disparity in pixels

    **Depth precision analysis**:

    Differentiating Z = fB/d with respect to d:

        ΔZ = (Z² / (f·B)) · Δd

    Depth error grows *quadratically* with distance. Example:

        f = 500 px, B = 0.12 m, Z = 10 m, Δd = 0.5 px
        ΔZ = 10² / (500 × 0.12) × 0.5 = 0.83 m

    At 50 m the same sub-pixel error gives ΔZ ≈ 20.8 m — stereo is most
    useful at close range.

    Parameters
    ----------
    disparity : np.ndarray
        Float disparity map (pixels). Zeros / negatives → infinite depth.
    focal_length : float
        Focal length in pixels.
    baseline : float
        Stereo baseline in metres.

    Returns
    -------
    depth : np.ndarray
        Depth map in metres. Invalid pixels (d ≤ 0) are set to 0.
    """
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (focal_length * baseline) / disparity[valid]
    return depth


class StereoDepthEstimator:
    """
    End-to-end stereo depth estimation and 3D reconstruction.

    Pipeline::

        rectified pair → SGBM disparity → depth map → coloured point cloud

    The back-projection from pixel (u, v) with depth Z to 3D uses:

        X = (u − cₓ) · Z / fₓ
        Y = (v − c_y) · Z / f_y

    where (fₓ, f_y) are focal lengths and (cₓ, c_y) the principal point,
    all in the calibration matrix K.
    """

    def __init__(
        self,
        K: np.ndarray,
        baseline: float,
        image_size: Tuple[int, int],
        num_disparities: int = 128,
        block_size: int = 5,
    ) -> None:
        """
        Initialise the stereo depth estimator.

        Parameters
        ----------
        K : np.ndarray
            3×3 camera intrinsic matrix (assumed identical for both cameras).
        baseline : float
            Distance between the two optical centres in metres.
        image_size : tuple of int
            (width, height) of the input images.
        num_disparities : int
            Maximum disparity search range.
        block_size : int
            SGBM block size.
        """
        self.K = K.astype(np.float64)
        self.baseline = baseline
        self.image_size = image_size
        self.num_disparities = num_disparities
        self.block_size = block_size

        self.fx: float = K[0, 0]
        self.fy: float = K[1, 1]
        self.cx: float = K[0, 2]
        self.cy: float = K[1, 2]

    def compute_depth(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a metric depth map from a rectified stereo pair.

        Parameters
        ----------
        left, right : np.ndarray
            Rectified left and right images (BGR or grayscale uint8).

        Returns
        -------
        depth : np.ndarray
            (H, W) float32 depth map in metres.
        """
        disparity = compute_disparity_sgbm(
            left, right,
            num_disparities=self.num_disparities,
            block_size=self.block_size,
        )
        return disparity_to_depth(disparity, self.fx, self.baseline)

    def compute_pointcloud(
        self,
        left: np.ndarray,
        depth: np.ndarray,
        max_depth: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Back-project a depth map into a coloured 3D point cloud.

        For each pixel (u, v) with depth Z > 0:

            X = (u − cₓ) · Z / fₓ
            Y = (v − c_y) · Z / f_y

        Parameters
        ----------
        left : np.ndarray
            Left image for colour (BGR uint8, H×W×3).
        depth : np.ndarray
            (H, W) depth map in metres.
        max_depth : float
            Discard points beyond this distance.

        Returns
        -------
        points : np.ndarray
            (N, 3) float32 array of XYZ coordinates.
        colours : np.ndarray
            (N, 3) uint8 array of RGB colours.
        """
        h, w = depth.shape[:2]
        u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))

        valid = (depth > 0) & (depth < max_depth)

        z = depth[valid]
        u = u_coords[valid].astype(np.float32)
        v = v_coords[valid].astype(np.float32)

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        points = np.stack([x, y, z], axis=-1).astype(np.float32)

        if left.ndim == 3:
            colours = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)[valid]
        else:
            g = left[valid]
            colours = np.stack([g, g, g], axis=-1)

        return points, colours

    def save_ply(
        self,
        path: str,
        points: np.ndarray,
        colours: np.ndarray,
    ) -> None:
        """
        Write a coloured point cloud to a PLY file.

        Parameters
        ----------
        path : str
            Output ``.ply`` file path.
        points : np.ndarray
            (N, 3) XYZ coordinates.
        colours : np.ndarray
            (N, 3) RGB uint8 colours.
        """
        n = len(points)
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        with open(path, "w") as f:
            f.write(header)
            for pt, c in zip(points, colours):
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                        f"{int(c[0])} {int(c[1])} {int(c[2])}\n")
