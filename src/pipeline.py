"""Full local mapping pipeline orchestrator.

This module ties together all reconstruction stages into a single
coherent pipeline:

1. **Feature detection + matching** (:mod:`.features`)
2. **Visual odometry** for pose estimation (:mod:`.odometry`)
3. **Depth estimation** — stereo or neural (:mod:`.stereo`, :mod:`.depth`)
4. **Point cloud generation** (:mod:`.pointcloud`)
5. **TSDF volumetric fusion** (:mod:`.tsdf`)
6. **Occupancy grid update** (:mod:`.occupancy`)
7. **Optional: semantic labelling** (:mod:`.semantic`)

The :class:`LocalMapper` orchestrates the full pipeline, maintaining a
running 3-D map that is incrementally refined with each new frame.

Usage example::

    mapper = LocalMapper(K, map_bounds=np.array([[-5,5],[-5,5],[-5,5]]))
    for rgb in frames:
        mapper.process_frame(rgb)
    verts, faces, norms, colors = mapper.get_mesh()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .features import detect_orb, match_bruteforce, ratio_test
from .odometry import MonocularVO
from .pointcloud import depth_to_pointcloud, transform_points, merge_pointclouds
from .tsdf import TSDFVolume
from .occupancy import OccupancyGrid


# ======================================================================
#  LocalMapper
# ======================================================================

class LocalMapper:
    """Full local mapping pipeline combining all modules.

    Pipeline per frame
    ------------------
    1. Feature detection + matching (features.py)
    2. Visual odometry for pose estimation (odometry.py)
    3. Depth estimation — stereo or neural (stereo.py / depth.py)
    4. Point cloud generation (pointcloud.py)
    5. TSDF integration (tsdf.py)
    6. Occupancy grid update (occupancy.py)
    7. Optional: semantic labelling (semantic.py)

    Parameters
    ----------
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix.
    map_bounds : ndarray, shape (3, 2)
        World-frame bounding box for the TSDF / occupancy volumes:
        ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]``.
    voxel_size : float
        Side length of each cubic voxel in metres.
    use_semantic : bool
        If ``True``, also perform per-frame semantic segmentation and
        integrate labels into a :class:`SemanticTSDF`.
    detector : str
        Feature detector for VO (``"orb"`` or ``"sift"``).
    """

    def __init__(
        self,
        K: NDArray[np.float64],
        map_bounds: NDArray[np.float64],
        voxel_size: float = 0.05,
        use_semantic: bool = False,
        detector: str = "orb",
    ) -> None:
        """Initialise the local mapping pipeline and all sub-modules."""
        self.K = np.asarray(K, dtype=np.float64)
        self.map_bounds = np.asarray(map_bounds, dtype=np.float64)
        self.voxel_size = voxel_size
        self.use_semantic = use_semantic

        # Visual odometry
        self._vo = MonocularVO(self.K, detector=detector)

        # TSDF volume
        self._tsdf = TSDFVolume(
            vol_bounds=self.map_bounds,
            voxel_size=voxel_size,
            trunc_dist=3.0 * voxel_size,
        )

        # Occupancy grid (coarser resolution)
        self._occupancy = OccupancyGrid(
            bounds=self.map_bounds,
            resolution=voxel_size * 2,
        )

        # Semantic TSDF (lazy init)
        self._semantic_tsdf: Optional[object] = None
        self._segmenter: Optional[object] = None
        if use_semantic:
            from .semantic import SemanticTSDF, SemanticSegmenter
            self._semantic_tsdf = SemanticTSDF(
                vol_bounds=self.map_bounds,
                voxel_size=voxel_size,
            )
            self._segmenter = SemanticSegmenter(model_name="simple")

        # State
        self._frame_count: int = 0
        self._pointclouds: List[NDArray] = []
        self._poses: List[NDArray] = []
        self._depth_estimator: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        rgb: NDArray[np.uint8],
        depth: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """Process a single frame: estimate pose, integrate depth.

        If *depth* is ``None``, a neural depth estimator is used to
        predict a relative depth map (scale-ambiguous).

        Parameters
        ----------
        rgb : ndarray, shape (H, W, 3)
            Input BGR image.
        depth : ndarray, shape (H, W), optional
            Metric depth map in metres.  ``None`` → estimate via neural
            network.

        Returns
        -------
        T : ndarray, shape (4, 4)
            Current camera-to-world pose.
        """
        # 1. Visual odometry → pose
        T_cam_to_world = self._vo.process_frame(rgb)
        self._poses.append(T_cam_to_world.copy())

        # 2. Depth estimation (if not provided)
        if depth is None:
            depth = self._estimate_depth(rgb)

        if depth is None:
            self._frame_count += 1
            return T_cam_to_world

        # 3. Point cloud
        pc = depth_to_pointcloud(depth, self.K, color=rgb)
        pc_world = transform_points(pc, T_cam_to_world)
        self._pointclouds.append(pc_world)

        # 4. TSDF integration
        self._tsdf.integrate(depth, self.K, T_cam_to_world, color=rgb)

        # 5. Occupancy grid update
        self._occupancy.update(depth, self.K, T_cam_to_world)

        # 6. Semantic (optional)
        if self.use_semantic and self._segmenter is not None:
            labels, _ = self._segmenter.segment(rgb)
            self._semantic_tsdf.integrate(
                depth, self.K, T_cam_to_world, labels,
            )

        self._frame_count += 1
        return T_cam_to_world

    def get_mesh(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Extract a triangle mesh from the TSDF volume.

        Applies Marching Cubes on the accumulated TSDF zero-crossing.

        Returns
        -------
        verts : ndarray, shape (V, 3)
        faces : ndarray, shape (F, 3)
        normals : ndarray, shape (V, 3)
        colors : ndarray, shape (V, 3)
        """
        return self._tsdf.extract_mesh()

    def get_semantic_mesh(self) -> Optional[Tuple[NDArray, NDArray, NDArray, NDArray]]:
        """Extract a semantically labelled mesh (if semantic mode is active).

        Returns
        -------
        tuple or None
            ``(verts, faces, normals, vertex_labels)`` if semantic mode
            is active; ``None`` otherwise.
        """
        if self._semantic_tsdf is not None:
            return self._semantic_tsdf.get_semantic_mesh()
        return None

    def get_trajectory(self) -> NDArray[np.float64]:
        """Return the camera trajectory as an (N, 3) array of positions.

        Returns
        -------
        ndarray, shape (N, 3)
        """
        return self._vo.get_trajectory()

    def get_poses(self) -> List[NDArray[np.float64]]:
        """Return all accumulated 4×4 camera-to-world poses.

        Returns
        -------
        list of ndarray, each (4, 4)
        """
        return list(self._poses)

    def get_pointcloud(self) -> NDArray[np.float64]:
        """Return the merged global point cloud.

        Returns
        -------
        ndarray, shape (M, C)
            Concatenation of all per-frame point clouds (C = 3 or 6).
        """
        if not self._pointclouds:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(self._pointclouds, axis=0)

    def get_map(self) -> TSDFVolume:
        """Return the primary map representation (TSDF volume).

        Returns
        -------
        TSDFVolume
        """
        return self._tsdf

    def export_mesh(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Extract a triangle mesh from the TSDF volume.

        Returns
        -------
        tuple
            ``(vertices, faces, normals, colors)`` from marching cubes.
        """
        return self._tsdf.extract_mesh()

    def get_occupancy_grid(self) -> OccupancyGrid:
        """Return the occupancy grid object."""
        return self._occupancy

    def visualize(self) -> None:
        """Visualise the current map state.

        Opens a Matplotlib figure with:

        * Camera trajectory (3-D)
        * Sparse point cloud (from TSDF zero-crossing)
        * Occupied voxels from the occupancy grid
        """
        import matplotlib.pyplot as plt
        from .viz import plot_cameras_3d, plot_pointcloud_3d, create_3d_axes

        fig = plt.figure(figsize=(16, 6))

        # Panel 1: camera trajectory
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        if self._poses:
            plot_cameras_3d(self._poses, K=self.K, ax=ax1, scale=0.1)
        ax1.set_title("Camera Trajectory")

        # Panel 2: TSDF point cloud
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        tsdf_pc = self._tsdf.get_point_cloud()
        if len(tsdf_pc) > 0:
            plot_pointcloud_3d(
                tsdf_pc[:, :3],
                colors=tsdf_pc[:, 3:6] if tsdf_pc.shape[1] >= 6 else None,
                ax=ax2, subsample=10000, point_size=0.5,
                title="TSDF Point Cloud",
            )
        else:
            ax2.set_title("TSDF Point Cloud (empty)")

        # Panel 3: occupancy grid
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        occ_pts = self._occupancy.get_occupied_points(threshold=0.6)
        if len(occ_pts) > 0:
            plot_pointcloud_3d(
                occ_pts, ax=ax3, subsample=10000, point_size=0.5,
                title="Occupied Voxels",
            )
        else:
            ax3.set_title("Occupancy Grid (empty)")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _estimate_depth(self, rgb: NDArray[np.uint8]) -> Optional[NDArray]:
        """Attempt neural monocular depth estimation."""
        try:
            if self._depth_estimator is None:
                from .depth import DepthEstimator
                self._depth_estimator = DepthEstimator(
                    backend="depth_anything_v2", device="cpu",
                )
            return self._depth_estimator.predict(rgb)
        except (ImportError, RuntimeError):
            return None
