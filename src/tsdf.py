"""
Truncated Signed Distance Function (TSDF) volumetric fusion.

Implements the integration scheme of Curless & Levoy (1996) and
surface extraction via Marching Cubes.
"""

from __future__ import annotations

import numpy as np


class TSDFVolume:
    """Truncated Signed Distance Function (TSDF) volume.

    Each voxel stores two quantities:

    * **tsdf** — signed distance to the nearest observed surface, clamped to
      [−1, +1] (in units of *trunc_dist*).

      - Positive → free space (in front of the surface)
      - Zero     → on the surface
      - Negative → behind the surface (occupied)

    * **weight** — accumulated observation count / confidence.

    Integration rule (Curless & Levoy 1996)
    ----------------------------------------
    For every voxel **v** with current values (tsdf_old, w_old):

        1. Project v into the camera frame:
               p_cam = T⁻¹ @ v_hom
           then into pixel coordinates:
               pixel = K @ p_cam[:3]  →  (u, v) = pixel[:2] / pixel[2]

        2. Look up the observed depth at that pixel:
               d_obs = depth[round(v), round(u)]

        3. Compute the raw signed distance:
               sdf = d_obs − p_cam_z

        4. Truncate:
               tsdf_new = clamp(sdf / trunc_dist, −1, +1)

        5. Weighted running average:
               tsdf = (w_old · tsdf_old + w_new · tsdf_new) / (w_old + w_new)
               w    = min(w_old + w_new,  w_max)

    Surface extraction proceeds by finding zero-crossings of the TSDF via
    Marching Cubes (Lorensen & Cline 1987).

    Parameters
    ----------
    vol_bounds : (3, 2) array — [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    voxel_size : side length of each cubic voxel in metres.
    trunc_dist : truncation distance (typically 3 × voxel_size).
    """

    def __init__(
        self,
        vol_bounds: np.ndarray,
        voxel_size: float = 0.02,
        trunc_dist: float = 0.06,
    ) -> None:
        """Initialise the TSDF volume, pre-computing voxel world coordinates."""
        vol_bounds = np.asarray(vol_bounds, dtype=np.float64)

        self.voxel_size = voxel_size
        self.trunc_dist = trunc_dist

        self._origin = vol_bounds[:, 0].copy()          # (3,)
        dims = np.ceil(
            (vol_bounds[:, 1] - vol_bounds[:, 0]) / voxel_size
        ).astype(np.int32)
        self._dims = dims                               # (Dx, Dy, Dz)

        self._tsdf = np.ones(dims, dtype=np.float32)    # +1 = free
        self._weight = np.zeros(dims, dtype=np.float32)
        self._color = np.zeros((*dims, 3), dtype=np.float32)

        self._voxel_coords = self._build_voxel_coords() # (Dx*Dy*Dz, 3)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_voxel_coords(self) -> np.ndarray:
        """Pre-compute the world coordinates of every voxel centre."""
        xv = np.arange(self._dims[0])
        yv = np.arange(self._dims[1])
        zv = np.arange(self._dims[2])
        grid = np.stack(np.meshgrid(xv, yv, zv, indexing="ij"), axis=-1)
        coords = grid.reshape(-1, 3).astype(np.float64) * self.voxel_size
        coords += self._origin + self.voxel_size / 2    # voxel centres
        return coords

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        T_camera_to_world: np.ndarray,
        color: np.ndarray | None = None,
        weight: float = 1.0,
    ) -> None:
        """Integrate a single depth frame into the TSDF volume.

        Steps (vectorised over all voxels):

            1. Transform voxel centres from world to camera frame:
                   p_cam = T_world_to_camera @ [x, y, z, 1]ᵀ
               where T_world_to_camera = inv(T_camera_to_world).

            2. Project into image coordinates via K:
                   u = fₓ · (Xc / Zc) + cₓ
                   v = fᵧ · (Yc / Zc) + cᵧ

            3. Discard voxels that fall outside the image or behind the camera.

            4. Compute raw signed distance:
                   sdf = d_obs − Zc

            5. Truncate and update:
                   tsdf_new  = clamp(sdf / trunc_dist, −1, +1)
                   tsdf, w   ← weighted running average (see class docstring).

        Parameters
        ----------
        depth              : (H, W) float array — observed depth in metres.
        K                  : (3, 3) camera intrinsic matrix.
        T_camera_to_world  : (4, 4) SE(3) — maps camera → world.
        color              : (H, W, 3) uint8, optional.
        weight             : observation weight for this frame.
        """
        h, w = depth.shape[:2]
        T_world_to_cam = np.linalg.inv(T_camera_to_world)

        # --- 1. World → camera -------------------------------------------
        n = self._voxel_coords.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        hom = np.concatenate([self._voxel_coords, ones], axis=1)   # (N, 4)
        cam = (T_world_to_cam @ hom.T).T                           # (N, 4)

        cam_x = cam[:, 0]
        cam_y = cam[:, 1]
        cam_z = cam[:, 2]

        # --- 2. Camera → pixel -------------------------------------------
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        pix_x = fx * (cam_x / cam_z) + cx
        pix_y = fy * (cam_y / cam_z) + cy

        # --- 3. Validity mask ---------------------------------------------
        valid = (
            (cam_z > 0)
            & (pix_x >= 0) & (pix_x < w - 1)
            & (pix_y >= 0) & (pix_y < h - 1)
        )

        pix_x = np.round(pix_x).astype(np.int32)
        pix_y = np.round(pix_y).astype(np.int32)

        pix_x = np.clip(pix_x, 0, w - 1)
        pix_y = np.clip(pix_y, 0, h - 1)

        d_obs = depth[pix_y, pix_x]
        valid &= (d_obs > 0) & np.isfinite(d_obs)

        # --- 4. SDF -------------------------------------------------------
        sdf = d_obs - cam_z
        valid &= sdf >= -self.trunc_dist

        tsdf_new = np.clip(sdf / self.trunc_dist, -1.0, 1.0)

        # --- 5. Weighted running average ----------------------------------
        valid_idx = np.where(valid)[0]
        vi = np.unravel_index(valid_idx, self._dims)

        w_old = self._weight[vi]
        tsdf_old = self._tsdf[vi]
        w_new = weight

        w_sum = w_old + w_new
        self._tsdf[vi] = (w_old * tsdf_old + w_new * tsdf_new[valid_idx]) / w_sum
        self._weight[vi] = np.minimum(w_sum, 255.0)

        if color is not None:
            c_obs = color[pix_y[valid_idx], pix_x[valid_idx]].astype(np.float32)
            c_old = self._color[vi]
            self._color[vi] = (w_old[..., None] * c_old + w_new * c_obs) / w_sum[..., None]

    # ------------------------------------------------------------------
    # Surface extraction
    # ------------------------------------------------------------------

    def extract_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract a triangle mesh via Marching Cubes on the TSDF zero-crossing.

        The Marching Cubes algorithm (Lorensen & Cline 1987):
            1. For every 2×2×2 block of voxels, classify each corner as
               inside (tsdf < 0) or outside (tsdf ≥ 0) → 256 cases.
            2. Look up the corresponding triangulation in a precomputed table.
            3. Interpolate vertex positions along edges using the TSDF values:
                   p = p₁ + (0 − tsdf₁) / (tsdf₂ − tsdf₁) · (p₂ − p₁)

        Returns
        -------
        verts : (V, 3) float — vertex positions in world frame.
        faces : (F, 3) int   — triangle indices.
        norms : (V, 3) float — per-vertex normals.
        colors: (V, 3) float — per-vertex colours (interpolated).
        """
        from skimage.measure import marching_cubes

        tsdf_vol = self._tsdf.copy()

        observed = self._weight > 0
        tsdf_vol[~observed] = 1.0

        try:
            verts, faces, norms, _ = marching_cubes(tsdf_vol, level=0.0)
        except (ValueError, RuntimeError):
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), np.zeros((0, 3)), np.zeros((0, 3))

        verts = verts * self.voxel_size + self._origin

        # Interpolate colours at extracted vertices.
        vert_idx = np.clip(
            np.round(
                (verts - self._origin) / self.voxel_size
            ).astype(int),
            0,
            np.array(self._dims) - 1,
        )
        colors = self._color[vert_idx[:, 0], vert_idx[:, 1], vert_idx[:, 2]]

        return verts, faces, norms, colors

    # ------------------------------------------------------------------
    # Point cloud extraction
    # ------------------------------------------------------------------

    def get_point_cloud(self) -> np.ndarray:
        """Extract points near the TSDF zero-crossing.

        Selects voxels where |tsdf| < 0.2 and weight > 0, returning
        their world-frame centres.

        Returns
        -------
        (N, 3) float array — point positions, or
        (N, 6) float array — positions + RGB if colour was integrated.
        """
        mask = (np.abs(self._tsdf) < 0.2) & (self._weight > 0)
        flat = mask.ravel()
        pts = self._voxel_coords[flat]

        colors = self._color[mask]
        if colors.any():
            return np.concatenate([pts, colors], axis=1)
        return pts

    extract_pointcloud = get_point_cloud

    def get_volume(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw TSDF volume and weight arrays."""
        return self._tsdf, self._weight
