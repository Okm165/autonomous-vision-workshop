"""
Point cloud processing utilities for 3D reconstruction.

Covers back-projection, rigid transforms, registration (ICP),
voxel downsampling, normal estimation, and outlier removal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# 1. Depth → point cloud
# ---------------------------------------------------------------------------


def depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    color: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project a depth map to a 3D point cloud.

    For each pixel (u, v) with depth *d*:

        X = (u - cₓ) · d / fₓ
        Y = (v - cᵧ) · d / fᵧ
        Z = d

    Vectorised: build a meshgrid of (u, v), then apply K⁻¹ element-wise.

    Parameters
    ----------
    depth : (H, W) float array — depth in metres (0 = invalid).
    K     : (3, 3) camera intrinsic matrix
            [[fₓ, 0, cₓ],
             [0, fᵧ, cᵧ],
             [0,  0,  1 ]]
    color : (H, W, 3) uint8, optional — per-pixel RGB.

    Returns
    -------
    (N, 3) or (N, 6) float array  [X, Y, Z (, R, G, B)].
    """
    h, w = depth.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    mask = depth > 0
    u = u[mask].astype(np.float64)
    v = v[mask].astype(np.float64)
    d = depth[mask].astype(np.float64)

    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    z = d

    points = np.stack([x, y, z], axis=-1)

    if color is not None:
        rgb = color[mask].astype(np.float64)
        if rgb.ndim == 1:
            rgb = rgb[:, np.newaxis]
        points = np.concatenate([points, rgb], axis=-1)

    return points


# ---------------------------------------------------------------------------
# 2. SE(3) transform
# ---------------------------------------------------------------------------


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply an SE(3) transformation to a point cloud.

    Maths (per point):
        P' = R @ P + t

    Vectorised form:
        P_hom = [P | 1]          (N, 4)
        P'    = (T @ P_homᵀ)ᵀ   i.e.  P_hom @ Tᵀ  →  take first 3 cols.

    Parameters
    ----------
    points : (N, 3) or (N, 6)  — XYZ or XYZ+RGB.
    T      : (4, 4) SE(3) matrix.

    Returns
    -------
    Transformed points, same shape as input (colours are preserved).
    """
    xyz = points[:, :3]
    ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    hom = np.concatenate([xyz, ones], axis=1)  # (N, 4)
    xyz_t = (hom @ T.T)[:, :3]

    if points.shape[1] > 3:
        return np.concatenate([xyz_t, points[:, 3:]], axis=1)
    return xyz_t


# ---------------------------------------------------------------------------
# 3. Merge multiple clouds
# ---------------------------------------------------------------------------


def merge_pointclouds(
    clouds: list[np.ndarray],
    poses: list[np.ndarray],
) -> np.ndarray:
    """Transform each cloud to the world frame and concatenate.

    Parameters
    ----------
    clouds : list of (Nᵢ, C) arrays (C = 3 or 6).
    poses  : list of (4, 4) camera-to-world SE(3) matrices.

    Returns
    -------
    (ΣNᵢ, C) merged point cloud.
    """
    transformed = [transform_points(c, T) for c, T in zip(clouds, poses, strict=False)]
    return np.concatenate(transformed, axis=0)


# ---------------------------------------------------------------------------
# 4. Voxel down-sampling
# ---------------------------------------------------------------------------


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Downsample a point cloud with a voxel grid filter.

    Algorithm:
        1. Compute voxel index for each point:
               idx = floor(point / voxel_size)
        2. Group points by voxel index.
        3. Replace each group with its centroid.

    Reduces point count while preserving overall geometry.

    Parameters
    ----------
    points     : (N, C) float array (C ≥ 3).
    voxel_size : side length of each cubic voxel (metres).

    Returns
    -------
    (M, C) downsampled point cloud (M ≤ N).
    """
    voxel_idx = np.floor(points[:, :3] / voxel_size).astype(np.int64)

    # Cantor-style key from three integers (shift to non-negative first).
    mins = voxel_idx.min(axis=0)
    shifted = voxel_idx - mins
    dims = shifted.max(axis=0) + 1
    keys = shifted[:, 0] * (dims[1] * dims[2]) + shifted[:, 1] * dims[2] + shifted[:, 2]

    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_pts = points[order]

    # Split at key boundaries → groups.
    splits = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(sorted_pts, splits)

    centroids = np.array([g.mean(axis=0) for g in groups])
    return centroids


# ---------------------------------------------------------------------------
# 5. Normal estimation (PCA)
# ---------------------------------------------------------------------------


def estimate_normals(
    points: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Estimate surface normals via PCA on local neighbourhoods.

    For each point **p**:
        1. Find *k* nearest neighbours {p₁, …, pₖ}.
        2. Compute centroid μ = (1/k) Σ pᵢ.
        3. Covariance matrix  C = (1/k) Σ (pᵢ − μ)(pᵢ − μ)ᵀ.
        4. SVD of C → eigenvalues λ₁ ≥ λ₂ ≥ λ₃.
        5. Normal = eigenvector corresponding to smallest eigenvalue λ₃.

    Planarity measure: (λ₂ − λ₃) / λ₁ — higher means more planar.

    Parameters
    ----------
    points : (N, 3) float array.
    k      : number of neighbours.

    Returns
    -------
    (N, 3) unit normals.
    """
    tree = KDTree(points)
    _, idx = tree.query(points, k=k)  # (N, k)

    neighbours = points[idx]  # (N, k, 3)
    centroids = neighbours.mean(axis=1, keepdims=True)  # (N, 1, 3)
    diff = neighbours - centroids  # (N, k, 3)

    # Covariance per point: (3, k) @ (k, 3) → (3, 3)  batched.
    cov = np.einsum("nki,nkj->nij", diff, diff) / k  # (N, 3, 3)

    # SVD — columns of V are eigenvectors sorted descending by σ.
    _, _, Vt = np.linalg.svd(cov)  # Vt: (N, 3, 3)
    normals = Vt[:, -1, :]  # smallest eigenvalue

    # Orient normals towards the camera (assume +Z viewpoint).
    flip = normals[:, 2] > 0
    normals[flip] *= -1

    # Normalise (some degenerate patches may yield ~0 normals).
    norms = np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    return normals / norms


# ---------------------------------------------------------------------------
# 6. ICP (Iterative Closest Point)
# ---------------------------------------------------------------------------


def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Point-to-point ICP alignment.

    Algorithm (Besl & McKay 1992, SVD solution from Arun et al. 1987):

        Repeat:
            1. For each source point, find closest target point (NN search).
            2. Compute cross-covariance matrix of matched pairs:
                   H = Σ (sᵢ − s̄)(tᵢ − t̄)ᵀ
               SVD:  H = U Σ Vᵀ
                   R = V Uᵀ   (correct sign via det check for reflections)
                   t = t̄ − R @ s̄
            3. Apply transform to source, repeat until convergence.

        Convergence: |MSEₖ − MSEₖ₋₁| < tolerance.

    The SVD closed-form solution minimises Σ ‖R·sᵢ + t − tᵢ‖² and is
    provably optimal for known correspondences (Arun et al. 1987).

    Parameters
    ----------
    source         : (N, 3) points to align.
    target         : (M, 3) reference points.
    max_iterations : upper iteration bound.
    tolerance      : convergence threshold on ΔMSE.

    Returns
    -------
    T          : (4, 4) cumulative SE(3) from source to target.
    distances  : (N,) final per-point distances.
    iterations : number of iterations executed.
    """
    src = source[:, :3].copy()
    T_accum = np.eye(4, dtype=np.float64)

    tree = KDTree(target[:, :3])
    prev_mse = np.inf

    for i in range(max_iterations):
        dists, idx = tree.query(src)
        matched = target[idx, :3]

        src_mean = src.mean(axis=0)
        tgt_mean = matched.mean(axis=0)

        src_c = src - src_mean
        tgt_c = matched - tgt_mean

        H = src_c.T @ tgt_c  # (3, 3)
        U, _, Vt = np.linalg.svd(H)
        V = Vt.T

        # Ensure a proper rotation (det = +1).
        d = np.linalg.det(V @ U.T)
        S = np.diag([1.0, 1.0, np.sign(d)])
        R = V @ S @ U.T
        t = tgt_mean - R @ src_mean

        src = (R @ src.T).T + t

        step = np.eye(4, dtype=np.float64)
        step[:3, :3] = R
        step[:3, 3] = t
        T_accum = step @ T_accum

        mse = float(np.mean(dists**2))
        if abs(prev_mse - mse) < tolerance:
            return T_accum, dists, i + 1
        prev_mse = mse

    dists, _ = tree.query(src)
    return T_accum, dists, max_iterations


# ---------------------------------------------------------------------------
# 7. Statistical outlier removal
# ---------------------------------------------------------------------------


def remove_outliers(
    points: np.ndarray,
    k: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove statistical outliers from a point cloud.

    For each point, compute the mean distance to its *k* nearest neighbours.
    A point is an outlier if that mean distance exceeds:

        μ_global + std_ratio · σ_global

    where μ_global and σ_global are the mean and standard deviation of all
    per-point mean-neighbour-distances.

    Parameters
    ----------
    points    : (N, C) float array (C ≥ 3).
    k         : number of neighbours.
    std_ratio : multiplier on σ for the threshold.

    Returns
    -------
    filtered : (M, C) inlier points.
    mask     : (N,) bool — True for inliers.
    """
    tree = KDTree(points[:, :3])
    # scipy returns inf distances when k exceeds the point count — clamp so
    # small clouds still yield finite per-point statistics.
    k = min(k, len(points) - 1)
    dists, _ = tree.query(points[:, :3], k=k + 1)  # +1 because self is first
    mean_dists = dists[:, 1:].mean(axis=1)  # skip self

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()

    threshold = global_mean + std_ratio * global_std
    mask = mean_dists <= threshold
    return points[mask], mask


# ---------------------------------------------------------------------------
# 8. Poisson surface reconstruction (Open3D wrapper)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriangleMeshData:
    """A triangle mesh as plain arrays.

    Open3D exposes ``TriangleMesh`` through pybind11, which carries no type
    information, so its ``vertices``/``triangles`` attributes are invisible to
    type checkers.  Returning the two arrays in a typed container keeps the
    mesh usable from typed code without an opaque handle.
    """

    vertices: np.ndarray
    triangles: np.ndarray


def poisson_reconstruct(
    points: np.ndarray,
    normals: np.ndarray,
    depth: int = 8,
) -> TriangleMeshData:
    """Surface reconstruction via the Poisson equation.

    The Poisson approach (Kazhdan et al. 2006) solves:

        ΔΦ = ∇·V

    where V is the vector field defined by the oriented normals, Δ is the
    Laplacian, and Φ is an indicator function whose iso-surface at a chosen
    value approximates the original surface.

    Steps (conceptual):
        1. Build an adaptive octree over the oriented point set.
        2. Define a smooth basis (B-spline) on each octree node.
        3. Compute the divergence of the normal field projected onto the basis.
        4. Solve the resulting sparse linear system  L Φ = d  (Laplacian).
        5. Extract the iso-surface of Φ via Marching Cubes.

    This function is a thin wrapper around Open3D's implementation.

    Parameters
    ----------
    points  : (N, 3) float array.
    normals : (N, 3) unit normals.
    depth   : octree depth (higher = finer detail, more memory).

    Returns
    -------
    mesh : TriangleMeshData
        Vertices ``(V, 3)`` and triangle indices ``(T, 3)``.  Open3D's
        ``TriangleMesh`` is a pybind11 type with no type information, so the
        two arrays it carries are returned in a typed container instead.

    Raises
    ------
    ImportError if Open3D is not installed.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Poisson reconstruction requires Open3D.  Install with:  pip install open3d"
        ) from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    mesh, _densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
    )
    return TriangleMeshData(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        triangles=np.asarray(mesh.triangles, dtype=np.int64),
    )
