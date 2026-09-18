"""Comparison of 3-D scene representations.

+-----------------+---------+--------+---------+------------+
| Representation  | Storage | Query  | Dynamic | Resolution |
+=================+=========+========+=========+============+
| Point cloud     | O(N)    | O(N)   | Easy    | Variable   |
| Dense voxel grid| O(n³)   | O(1)   | Medium  | Fixed      |
| TSDF volume     | O(n³)   | O(1)   | Medium  | Fixed      |
| Octree          | O(N)    | O(logN)| Hard    | Adaptive   |
| Triangle mesh   | O(N)    | O(N)   | Hard    | Variable   |
| NeRF (MLP)      | O(W)    | O(K)   | Hard    | Continuous |
| 3D Gaussians    | O(N)    | O(N)   | Medium  | Adaptive   |
+-----------------+---------+--------+---------+------------+

Where N = number of elements, n = grid resolution, W = network weights,
K = number of ray samples.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ======================================================================
#  Dense Voxel Grid
# ======================================================================


class VoxelGrid:
    """Dense 3-D voxel grid for occupancy / feature storage.

    Each voxel stores a boolean (occupied / free) or a float value.
    Random access is O(1) via index arithmetic, but memory is O(n³)
    regardless of actual occupancy.

    Parameters
    ----------
    bounds : (3, 2) array — [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    resolution : voxel side length in metres
    """

    resolution: float
    _origin: NDArray[np.float64]
    _dims: NDArray[np.int64]
    _grid: NDArray[np.float32]

    def __init__(self, bounds: NDArray[np.float64], resolution: float = 0.05) -> None:
        """Initialise a dense voxel grid over the given world-space bounds."""
        bounds = np.asarray(bounds, dtype=np.float64)
        self.resolution = resolution
        self._origin = bounds[:, 0].copy()
        self._dims = np.ceil((bounds[:, 1] - bounds[:, 0]) / resolution).astype(int)
        self._grid = np.zeros(self._dims, dtype=np.float32)

    def _world_to_voxel(self, points: NDArray[np.float64]) -> NDArray[np.int64]:
        """Convert (N, 3) world coords to integer voxel indices."""
        idx = ((points - self._origin) / self.resolution).astype(int)
        return np.clip(idx, 0, self._dims - 1)

    def insert(
        self, points: NDArray[np.float64], values: float | NDArray[np.float64] = 1.0
    ) -> None:
        """Mark voxels containing *points* as occupied."""
        idx = self._world_to_voxel(np.atleast_2d(points))
        self._grid[idx[:, 0], idx[:, 1], idx[:, 2]] = values

    def query(self, point: NDArray[np.float64]) -> float:
        """O(1) lookup of a single world-frame point."""
        idx = self._world_to_voxel(np.atleast_2d(np.asarray(point, dtype=np.float64)))[
            0
        ]
        return float(self._grid[idx[0], idx[1], idx[2]])

    def to_pointcloud(self) -> NDArray[np.float64]:
        """Return (N, 3) world coordinates of occupied voxels."""
        occ = np.argwhere(self._grid > 0.5)
        return (
            occ.astype(np.float64) * self.resolution
            + self._origin
            + self.resolution / 2
        )

    @property
    def memory_bytes(self) -> int:
        """Total memory consumed by the underlying grid array in bytes."""
        return self._grid.nbytes


# ======================================================================
#  Octree
# ======================================================================


class _OctreeNode:
    __slots__: tuple[str, ...] = ("center", "children", "data", "half_size", "is_leaf")

    def __init__(self, center: NDArray[np.float64], half_size: float) -> None:
        """Initialise an octree node with a centre and bounding half-size."""
        self.center: NDArray[np.float64] = center
        self.half_size: float = half_size
        self.children: list[_OctreeNode | None] = [None] * 8
        self.is_leaf: bool = True
        self.data: list[NDArray[np.float64]] = []


class Octree:
    """Octree for adaptive-resolution 3-D spatial indexing.

    Subdivides space only where data exists, achieving O(N) storage
    instead of O(n³) for a dense grid.

    Query complexity is O(log n) where n is the effective grid resolution.

    Parameters
    ----------
    center : (3,) centre of the root bounding cube
    half_size : half the side length of the root cube
    max_depth : maximum subdivision depth (resolution = half_size / 2^max_depth)
    max_leaf_points : max points before subdivision
    """

    _root: _OctreeNode
    max_depth: int
    max_leaf_points: int
    n_points: int
    _n_nodes: int

    def __init__(
        self,
        center: NDArray[np.float64] = np.zeros(3),
        half_size: float = 10.0,
        max_depth: int = 8,
        max_leaf_points: int = 4,
    ) -> None:
        """Initialise an octree with root cube centre, half-size, and depth limit."""
        self._root = _OctreeNode(np.asarray(center, dtype=np.float64), half_size)
        self.max_depth = max_depth
        self.max_leaf_points = max_leaf_points
        self.n_points = 0
        self._n_nodes = 1

    @staticmethod
    def _child_index(point: NDArray[np.float64], center: NDArray[np.float64]) -> int:
        """Map a point to one of 8 octants (0–7)."""
        idx = 0
        if point[0] >= center[0]:
            idx |= 1
        if point[1] >= center[1]:
            idx |= 2
        if point[2] >= center[2]:
            idx |= 4
        return idx

    @staticmethod
    def _child_center(
        parent_center: NDArray[np.float64], half: float, octant: int
    ) -> NDArray[np.float64]:
        """Compute the centre of a child node given its octant index."""
        offset = np.array(
            [
                half if (octant & 1) else -half,
                half if (octant & 2) else -half,
                half if (octant & 4) else -half,
            ]
        )
        return parent_center + offset

    def insert(self, point: NDArray[np.float64]) -> None:
        """Insert a single 3-D point."""
        self._insert(self._root, np.asarray(point, dtype=np.float64), 0)
        self.n_points += 1

    def _insert(
        self, node: _OctreeNode, point: NDArray[np.float64], depth: int
    ) -> None:
        """Recursively insert a point, subdividing when a leaf overflows."""
        if node.is_leaf:
            node.data.append(point)
            if len(node.data) > self.max_leaf_points and depth < self.max_depth:
                self._subdivide(node, depth)
        else:
            oi = self._child_index(point, node.center)
            if node.children[oi] is None:
                child_half = node.half_size / 2.0
                child_center = self._child_center(node.center, child_half, oi)
                node.children[oi] = _OctreeNode(child_center, child_half)
                self._n_nodes += 1
            child = node.children[oi]
            assert child is not None
            self._insert(child, point, depth + 1)

    def _subdivide(self, node: _OctreeNode, depth: int) -> None:
        """Split a leaf node into eight children and redistribute its points."""
        node.is_leaf = False
        points = node.data
        node.data = []
        for p in points:
            self._insert(node, p, depth)

    def query(self, point: NDArray[np.float64]) -> bool:
        """Check whether a point is contained in the octree."""
        return self._query(self._root, np.asarray(point, dtype=np.float64))

    def _query(self, node: _OctreeNode | None, point: NDArray[np.float64]) -> bool:
        """Recursively search for a point, returning True if found."""
        if node is None:
            return False
        if node.is_leaf:
            return any(np.allclose(p, point, atol=1e-08) for p in node.data)
        oi = self._child_index(point, node.center)
        return self._query(node.children[oi], point)

    def get_all_points(self) -> NDArray[np.float64]:
        """Collect all stored points."""
        pts: list[NDArray[np.float64]] = []
        self._collect(self._root, pts)
        if not pts:
            return np.zeros((0, 3))
        return np.stack(pts)

    def _collect(
        self, node: _OctreeNode | None, acc: list[NDArray[np.float64]]
    ) -> None:
        """Recursively gather all points stored in the subtree into *acc*."""
        if node is None:
            return
        if node.is_leaf:
            acc.extend(node.data)
        else:
            for child in node.children:
                self._collect(child, acc)

    @property
    def n_nodes(self) -> int:
        """Total number of nodes (internal + leaf) in the octree."""
        return self._n_nodes


# ======================================================================
#  Comparison utility
# ======================================================================


def compare_representations(
    points: NDArray[np.float64],
    bounds: NDArray[np.float64],
    voxel_size: float = 0.05,
) -> dict[str, dict[str, float | int]]:
    """Create each representation from the same point cloud and report stats.

    Returns a dict with keys ``voxel_grid`` and ``octree``, each containing
    storage size, insertion time, and query time.
    """
    import time

    bounds = np.asarray(bounds, dtype=np.float64)

    # --- Voxel grid ---
    t0 = time.perf_counter()
    vg = VoxelGrid(bounds, voxel_size)
    vg.insert(points)
    t_vg_insert = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_probe = min(100, len(points))
    for p in points[:n_probe]:
        vg.query(p)
    t_vg_query = (time.perf_counter() - t0) / max(1, n_probe)

    # --- Octree ---
    center = bounds.mean(axis=1)
    half = float(np.max(bounds[:, 1] - bounds[:, 0]) / 2)
    t0 = time.perf_counter()
    ot = Octree(center, half, max_depth=10)
    for p in points:
        ot.insert(p)
    t_ot_insert = time.perf_counter() - t0

    t0 = time.perf_counter()
    for p in points[:n_probe]:
        ot.query(p)
    t_ot_query = (time.perf_counter() - t0) / max(1, n_probe)

    return {
        "voxel_grid": {
            "memory_bytes": vg.memory_bytes,
            "insert_time_s": t_vg_insert,
            "query_time_s": t_vg_query,
            "n_occupied": int((vg._grid > 0.5).sum()),
        },
        "octree": {
            "n_nodes": ot.n_nodes,
            "n_points": ot.n_points,
            "insert_time_s": t_ot_insert,
            "query_time_s": t_ot_query,
        },
    }
