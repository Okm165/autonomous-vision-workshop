"""Probabilistic 3D occupancy grid using log-odds representation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


class OccupancyGrid:
    """
    Probabilistic 3D occupancy grid using log-odds representation.

    Each voxel stores a log-odds value:
        l(x) = log(p(x) / (1 - p(x)))

    where p(x) is the probability that voxel x is occupied.

    Log-odds advantages:
    - Simple update: l_new = l_old + l_measurement - l_prior
    - No need for normalization
    - Numerically stable (avoids probabilities near 0 or 1)

    Inverse sensor model:
    For a depth measurement d at pixel (u,v):
    - Voxels along the ray before d: free (negative log-odds)
    - Voxels near d: occupied (positive log-odds)
    - Voxels beyond d: unknown (zero log-odds)

    Ray casting:
    Use Bresenham's line algorithm or DDA to traverse voxels along each ray.

    Conversion:
        p(x) = 1 - 1 / (1 + exp(l(x)))

    Clamping:
        l(x) ∈ [l_min, l_max] to prevent overconfidence

    Parameters
    ----------
    bounds : (3, 2) array
        Min/max for x, y, z in metres.  ``bounds[i] = [min_i, max_i]``.
    resolution : float
        Voxel edge length in metres.
    log_odds_free : float
        Log-odds update for free-space observations (negative).
    log_odds_occ : float
        Log-odds update for occupied-space observations (positive).
    log_odds_max : float
        Symmetric clamp ``[-log_odds_max, log_odds_max]`` to prevent
        overconfidence.
    """

    def __init__(
        self,
        bounds: NDArray,
        resolution: float = 0.1,
        log_odds_free: float = -0.4,
        log_odds_occ: float = 0.85,
        log_odds_max: float = 3.5,
    ) -> None:
        """Initialise the occupancy grid with bounds and log-odds parameters."""
        self.bounds = np.asarray(bounds, dtype=np.float64)  # (3, 2)
        self.resolution = resolution
        self.log_odds_free = log_odds_free
        self.log_odds_occ = log_odds_occ
        self.log_odds_max = log_odds_max

        self.origin = self.bounds[:, 0].copy()
        dims = self.bounds[:, 1] - self.bounds[:, 0]
        self.grid_size = np.ceil(dims / resolution).astype(int)
        self.grid = np.zeros(tuple(self.grid_size), dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        depth: np.ndarray,
        K: np.ndarray,
        T_cam_to_world: np.ndarray,
    ) -> None:
        """
        Update grid with a depth observation.

        For each pixel with valid depth:
        1. Back-project to 3D point in camera frame using
           ``p_cam = d · K⁻¹ [u, v, 1]ᵀ``
        2. Transform to world frame: ``p_world = T · p_cam``
        3. Ray-cast from camera origin to surface point using Bresenham 3D
        4. Mark traversed voxels as free, surface voxel as occupied

        Parameters
        ----------
        depth : (H, W) depth image in metres.  Non-positive values are
            treated as invalid / no-return.
        K : (3, 3) camera intrinsic matrix.
        T_cam_to_world : (4, 4) rigid transform from camera to world frame.
        """
        H, W = depth.shape
        R = T_cam_to_world[:3, :3]
        t = T_cam_to_world[:3, 3]

        cam_origin_voxel = self._world_to_voxel(t)
        K_inv = np.linalg.inv(K)

        step = max(1, min(H, W) // 80)
        for v in range(0, H, step):
            for u in range(0, W, step):
                d = depth[v, u]
                if d <= 0.0:
                    continue

                pixel_h = np.array([u, v, 1.0])
                p_cam = d * (K_inv @ pixel_h)
                p_world = R @ p_cam + t

                end_voxel = self._world_to_voxel(p_world)

                if not self._in_bounds(end_voxel):
                    continue

                ray_voxels = self._bresenham_3d(cam_origin_voxel, end_voxel)

                for vx, vy, vz in ray_voxels[:-1]:
                    if self._in_bounds_ijk(vx, vy, vz):
                        self.grid[vx, vy, vz] += self.log_odds_free

                ex, ey, ez = ray_voxels[-1]
                if self._in_bounds_ijk(ex, ey, ez):
                    self.grid[ex, ey, ez] += self.log_odds_occ

        np.clip(self.grid, -self.log_odds_max, self.log_odds_max, out=self.grid)

    def get_occupied_points(self, threshold: float = 0.5) -> NDArray:
        """Return (N, 3) array of voxel centres with p(occ) > *threshold*.

        Conversion from log-odds:
            p = 1 - 1 / (1 + exp(l))
        """
        log_thresh = np.log(threshold / (1.0 - threshold))
        ix, iy, iz = np.where(self.grid > log_thresh)
        return self._voxel_indices_to_world(ix, iy, iz)

    def get_free_points(self, threshold: float = 0.3) -> NDArray:
        """Return (N, 3) array of voxel centres with p(occ) < *threshold*."""
        log_thresh = np.log(threshold / (1.0 - threshold))
        ix, iy, iz = np.where(self.grid < log_thresh)
        return self._voxel_indices_to_world(ix, iy, iz)

    def is_occupied(self, point: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if a 3D point is in an occupied voxel.

        Parameters
        ----------
        point : (3,) array
            World-frame 3-D point.
        threshold : float
            Occupancy probability threshold (default 0.5, i.e. log-odds > 0).
        """
        idx = self._world_to_voxel(np.asarray(point, dtype=np.float64))
        if not self._in_bounds(idx):
            return False
        log_thresh = np.log(threshold / (1.0 - threshold))
        return bool(self.grid[idx] > log_thresh)

    def get_free_mask(self) -> NDArray:
        """Return a boolean mask of free voxels (log-odds below free threshold)."""
        log_free = np.log(0.3 / 0.7)
        return self.grid < log_free

    def raycast_3d(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_range: float,
    ) -> Tuple[np.ndarray | None, float]:
        """Cast a ray and return (hit_point, distance) or (None, max_range) if no hit."""
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        origin = np.asarray(origin, dtype=np.float64)
        step = self.resolution * 0.5  # half-voxel steps to avoid missing diagonal traversals
        for t in np.arange(0, max_range, step):
            point = origin + direction * t
            if self.is_occupied(point):
                return point, float(t)
        return None, max_range

    def to_probability(self) -> NDArray:
        """Convert the full log-odds grid to a probability grid.

        p(x) = 1 - 1 / (1 + exp(l(x)))
             = exp(l) / (1 + exp(l))
             = sigmoid(l)
        """
        return 1.0 / (1.0 + np.exp(-self.grid))

    # ------------------------------------------------------------------
    # Ray casting
    # ------------------------------------------------------------------

    def _bresenham_3d(
        self,
        start: Tuple[int, int, int],
        end: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int]]:
        """3D Bresenham line algorithm for ray casting.

        Returns a list of (ix, iy, iz) voxel indices visited along
        the line segment from *start* to *end*, inclusive.

        The algorithm generalises 2-D Bresenham to three axes by
        tracking two independent error accumulators (one per
        secondary axis) relative to the dominant axis.
        """
        x0, y0, z0 = int(start[0]), int(start[1]), int(start[2])
        x1, y1, z1 = int(end[0]), int(end[1]), int(end[2])

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)

        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        sz = 1 if z1 > z0 else -1

        points: List[Tuple[int, int, int]] = []

        if dx >= dy and dx >= dz:
            ey = 2 * dy - dx
            ez = 2 * dz - dx
            for _ in range(dx + 1):
                points.append((x0, y0, z0))
                if ey >= 0:
                    y0 += sy
                    ey -= 2 * dx
                if ez >= 0:
                    z0 += sz
                    ez -= 2 * dx
                ey += 2 * dy
                ez += 2 * dz
                x0 += sx

        elif dy >= dx and dy >= dz:
            ex = 2 * dx - dy
            ez = 2 * dz - dy
            for _ in range(dy + 1):
                points.append((x0, y0, z0))
                if ex >= 0:
                    x0 += sx
                    ex -= 2 * dy
                if ez >= 0:
                    z0 += sz
                    ez -= 2 * dy
                ex += 2 * dx
                ez += 2 * dz
                y0 += sy

        else:
            ex = 2 * dx - dz
            ey = 2 * dy - dz
            for _ in range(dz + 1):
                points.append((x0, y0, z0))
                if ex >= 0:
                    x0 += sx
                    ex -= 2 * dz
                if ey >= 0:
                    y0 += sy
                    ey -= 2 * dz
                ex += 2 * dx
                ey += 2 * dy
                z0 += sz

        if not points:
            points.append((int(start[0]), int(start[1]), int(start[2])))

        return points

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _world_to_voxel(self, point: NDArray) -> Tuple[int, int, int]:
        """Map a world-frame point to integer voxel indices."""
        idx = ((point - self.origin) / self.resolution).astype(int)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def _voxel_indices_to_world(
        self, ix: NDArray, iy: NDArray, iz: NDArray
    ) -> NDArray:
        """Convert arrays of voxel indices to (N, 3) world-frame centres."""
        coords = np.stack([ix, iy, iz], axis=-1).astype(np.float64)
        return self.origin + (coords + 0.5) * self.resolution

    def _in_bounds(self, voxel: Tuple[int, int, int]) -> bool:
        """Check whether *voxel* lies inside the grid."""
        return all(0 <= voxel[i] < self.grid_size[i] for i in range(3))

    def _in_bounds_ijk(self, i: int, j: int, k: int) -> bool:
        """Check whether voxel (i, j, k) lies inside the grid."""
        return (
            0 <= i < self.grid_size[0]
            and 0 <= j < self.grid_size[1]
            and 0 <= k < self.grid_size[2]
        )
