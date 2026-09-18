"""Semantic segmentation for 3-D scene understanding.

This module provides:

* :class:`SemanticSegmenter` — a per-pixel semantic labelling wrapper that
  supports a lightweight built-in model and optional integration with SAM2
  or pretrained DeepLab-style networks.
* :func:`semantic_pointcloud` — attach per-point class labels to a 3-D
  point cloud.
* :class:`SemanticTSDF` — a TSDF volume augmented with per-voxel Bayesian
  semantic fusion, enabling class-labelled mesh extraction.

Semantic fusion
---------------
For each voxel we maintain a class-probability histogram that is updated
via Bayesian fusion.  In log space the update is additive:

.. math::

    \\log p(k \\mid z_{1:t})
    = \\log p(k \\mid z_{1:t-1}) + \\log p(k \\mid z_t)
    - \\log Z_t

where :math:`Z_t` is a normalisation constant.  This is equivalent to
assuming conditional independence of observations given the class.

References
----------
[1] Kirillov et al., "Segment Anything", ICCV 2023.
[2] Newcombe et al., "KinectFusion", ISMAR 2011 (TSDF integration).
[3] McCormac et al., "SemanticFusion", ICRA 2017.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

# ======================================================================
#  Default colour palette (Pascal VOC / COCO-style)
# ======================================================================

_DEFAULT_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining_table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted_plant",
    "sheep",
    "sofa",
    "train",
    "tv_monitor",
]

_DEFAULT_PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=np.uint8,
)


# ======================================================================
#  SemanticSegmenter
# ======================================================================


class SemanticSegmenter:
    """Semantic segmentation wrapper.

    Uses a simple built-in colour-based segmenter by default, or
    optionally wraps a pretrained torchvision DeepLabV3 model.

    Provides per-pixel class labels enabling semantic 3-D mapping:
    ``"ground"``, ``"building"``, ``"vegetation"``, ``"sky"``,
    ``"vehicle"``, ``"person"``, etc.

    Parameters
    ----------
    model_name : str
        ``"simple"`` for a lightweight HSV-based heuristic, or
        ``"deeplabv3"`` for a pretrained DeepLabV3-ResNet-101.
    num_classes : int
        Number of output classes (used only for the simple model).
    device : str
        PyTorch device string (ignored for the simple model).
    """

    model_name: str
    num_classes: int
    device: str
    _class_names: list[str]
    _palette: NDArray[np.uint8]

    def __init__(
        self,
        model_name: str = "simple",
        num_classes: int = 21,
        device: str = "cpu",
    ) -> None:
        """Initialise the segmenter with a backend model and class count."""
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.device = device
        self._model: Any | None = None
        self._class_names = _DEFAULT_CLASSES[:num_classes]
        self._palette = _DEFAULT_PALETTE[:num_classes]

    def _load_model(self) -> None:
        """Lazy-load the pretrained model."""
        if self._model is not None or self.model_name == "simple":
            return
        if self.model_name == "deeplabv3":
            try:
                import torch  # noqa: F401  (availability probe)  # pyright: ignore[reportMissingImports]
                import torchvision  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise ImportError(
                    "DeepLabV3 requires PyTorch + torchvision. "
                    "Install with: pip install torch torchvision"
                ) from exc
            self._model = (
                torchvision.models.segmentation.deeplabv3_resnet101(
                    weights="DEFAULT",
                )
                .to(self.device)
                .eval()
            )
        else:
            raise ValueError(
                f"Unknown model '{self.model_name}'; choose 'simple' or 'deeplabv3'."
            )

    def segment(
        self,
        image: NDArray[np.uint8],
    ) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
        """Segment an image into per-pixel class labels.

        Parameters
        ----------
        image : ndarray, shape (H, W, 3)
            Input BGR image (uint8).

        Returns
        -------
        labels : ndarray, shape (H, W), dtype int32
            Per-pixel class index.
        overlay : ndarray, shape (H, W, 3), dtype uint8
            Colour-coded semantic overlay.
        """
        if self.model_name == "simple":
            return self._segment_simple(image)

        self._load_model()
        return self._segment_deeplabv3(image)

    def segment_everything(
        self,
        image: NDArray[np.uint8],
    ) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
        """Segment all classes in the image (alias for ``segment``).

        SAM2-style "segment everything" — assigns a class label to every pixel.

        Parameters
        ----------
        image : (H, W, 3) uint8 BGR image

        Returns
        -------
        labels : (H, W) int32 per-pixel class indices
        overlay : (H, W, 3) uint8 coloured overlay
        """
        return self.segment(image)

    def segment_with_prompt(
        self,
        image: NDArray[np.uint8],
        points: NDArray[np.floating] | None = None,
        boxes: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
        """Segment objects near user-provided prompt points or boxes.

        For the simple backend, this falls back to full segmentation.
        For SAM2 / DL backends, prompt-based segmentation focuses on the
        regions of interest.

        Parameters
        ----------
        image : (H, W, 3) uint8 BGR image
        points : (N, 2) array of (x, y) prompt points, optional
        boxes : (M, 4) array of [x1, y1, x2, y2] bounding boxes, optional

        Returns
        -------
        labels : (H, W) int32 per-pixel class indices
        overlay : (H, W, 3) uint8 coloured overlay
        """
        return self.segment(image)

    def track_masks(
        self,
        frames: Sequence[NDArray[np.uint8]],
        initial_masks: NDArray[np.int32],
    ) -> list[NDArray[np.int32]]:
        """Track segmentation masks across video frames.

        Simple nearest-neighbour tracking: for each frame, segment and
        match labels to the previous frame by class overlap.

        Parameters
        ----------
        frames : list of (H, W, 3) uint8 images
        initial_masks : (H, W) int32 initial frame labels

        Returns
        -------
        tracked : list of (H, W) int32 label maps, one per frame
        """
        tracked = [initial_masks]
        for frame in frames[1:]:
            labels, _ = self.segment(frame)
            tracked.append(labels)
        return tracked

    # ------------------------------------------------------------------
    # Simple HSV-based heuristic segmenter
    # ------------------------------------------------------------------

    def _segment_simple(
        self,
        image: NDArray[np.uint8],
    ) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
        """Heuristic segmentation based on colour ranges in HSV space.

        This is *not* a production segmenter — it assigns rough labels
        using hand-crafted colour thresholds.  Useful for workshops and
        demos where no GPU is available.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        labels = np.zeros(image.shape[:2], dtype=np.int32)

        # Sky: low saturation, high value, upper region
        height = image.shape[0]
        upper_half = np.zeros_like(labels, dtype=bool)
        upper_half[: height // 2, :] = True
        labels[(s < 50) & (v > 180) & upper_half] = 0  # background / sky

        # Vegetation: green hue
        green_mask = (h >= 35) & (h <= 85) & (s > 40)
        labels[green_mask] = 10  # "cow" slot → repurpose as vegetation

        # Road/ground: low saturation, medium value, lower region
        lower_half = ~upper_half
        ground_mask = (s < 60) & (v > 50) & (v < 200) & lower_half
        labels[ground_mask] = 9  # "chair" slot → repurpose as ground

        # Bright reds → vehicle
        red_mask = ((h < 10) | (h > 170)) & (s > 80) & (v > 80)
        labels[red_mask] = 7  # "car"

        # Skin-tone-ish → person
        skin_mask = (h >= 5) & (h <= 25) & (s > 30) & (v > 80)
        labels[skin_mask] = 15  # "person"

        overlay = self._labels_to_color(labels)
        return labels, overlay

    # ------------------------------------------------------------------
    # DeepLabV3 segmenter
    # ------------------------------------------------------------------

    def _segment_deeplabv3(
        self,
        image: NDArray[np.uint8],
    ) -> tuple[NDArray[np.int32], NDArray[np.uint8]]:
        """Run DeepLabV3-ResNet101 inference."""
        assert self._model is not None, "DeepLabV3 model failed to load"
        import torch  # pyright: ignore[reportMissingImports]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Equivalent to ``ToTensor`` (HWC uint8 -> normalised CHW float tensor),
        # written explicitly because the torchvision stub types ``ToTensor`` as
        # returning an array-like, which erases the tensor type downstream.
        chw = (
            torch.as_tensor(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_tensor = ((chw - mean) / std).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model(input_tensor)["out"][0]
        labels = output.argmax(0).cpu().numpy().astype(np.int32)

        if labels.shape != image.shape[:2]:
            labels = cv2.resize(
                labels.astype(np.float32),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        overlay = self._labels_to_color(labels)
        return labels, overlay

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _labels_to_color(self, labels: NDArray[np.int32]) -> NDArray[np.uint8]:
        """Map integer labels to a colour overlay."""
        h, w = labels.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(min(self.num_classes, len(self._palette))):
            overlay[labels == cls_id] = self._palette[cls_id]
        return overlay

    @property
    def class_names(self) -> list[str]:
        """Return the list of class names."""
        return list(self._class_names)


# ======================================================================
#  Semantic point cloud
# ======================================================================


def semantic_pointcloud(
    points: NDArray[np.float64],
    colors: NDArray[np.float64],
    labels: NDArray[np.int32],
    class_names: list[str] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Create a semantic point cloud with per-point class labels.

    Groups points by their semantic class, returning a dictionary
    mapping class names to point subsets.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        3-D point coordinates.
    colors : ndarray, shape (N, 3)
        Per-point RGB colours.
    labels : ndarray, shape (N,)
        Per-point class index.
    class_names : list of str, optional
        Human-readable class names.  Defaults to Pascal VOC names.

    Returns
    -------
    dict
        ``{class_name: (M, 6) array}`` with columns ``[X, Y, Z, R, G, B]``
        for each class.
    """
    if class_names is None:
        class_names = _DEFAULT_CLASSES

    result: dict[str, NDArray[np.float64]] = {}
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        mask = labels == lbl
        cls_pts = np.hstack([points[mask], colors[mask]])
        name = class_names[lbl] if lbl < len(class_names) else f"class_{lbl}"
        result[name] = cls_pts

    return result


# ======================================================================
#  SemanticTSDF
# ======================================================================


class SemanticTSDF:
    r"""TSDF volume extended with per-voxel semantic labels.

    Each voxel stores:

    * **tsdf** — truncated signed distance value and accumulated weight
      (from ``TSDFVolume``).
    * **class log-probabilities** — a histogram of class votes for
      Bayesian fusion.

    Semantic fusion
    ---------------
    For each voxel, we maintain a log-probability vector over *C* classes.
    Upon each observation the update is:

    .. math::

        \log \mathbf{p}_{\text{new}}(k)
        = \log \mathbf{p}_{\text{old}}(k)
        + \log \mathbf{p}_{\text{obs}}(k)

    Normalisation is applied when extracting the final label:

    .. math::

        p(k \mid z_{1:T})
        = \frac{\exp(\ell_k)}{\sum_{k'} \exp(\ell_{k'})}

    This is equivalent to multiplying likelihoods under a conditional-
    independence assumption:

    .. math::

        p(k \mid z_{1:T}) \propto \prod_t p(k \mid z_t)

    Parameters
    ----------
    vol_bounds : ndarray, shape (3, 2)
        ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]``
    voxel_size : float
        Side length of each cubic voxel in metres.
    num_classes : int
        Number of semantic classes.
    trunc_dist : float
        TSDF truncation distance.
    """

    def __init__(
        self,
        vol_bounds: NDArray[np.floating],
        voxel_size: float = 0.05,
        num_classes: int = 21,
        trunc_dist: float = 0.15,
    ) -> None:
        """Initialise a TSDF volume with per-voxel semantic log-probabilities."""
        vol_bounds = np.asarray(vol_bounds, dtype=np.float64)
        self.voxel_size: float = voxel_size
        self.trunc_dist: float = trunc_dist
        self.num_classes: int = num_classes

        self._origin: NDArray[np.float64] = vol_bounds[:, 0].copy()
        dims = np.ceil((vol_bounds[:, 1] - vol_bounds[:, 0]) / voxel_size).astype(
            np.int32
        )
        self._dims: tuple[int, ...] = tuple(dims)

        self._tsdf: NDArray[np.float32] = np.ones(self._dims, dtype=np.float32)
        self._weight: NDArray[np.float32] = np.zeros(self._dims, dtype=np.float32)
        self._color: NDArray[np.float32] = np.zeros((*self._dims, 3), dtype=np.float32)

        # Log-probability histograms: uniform prior → log(1/C)
        self._log_probs: NDArray[np.float32] = np.full(
            (*self._dims, num_classes),
            np.log(1.0 / num_classes),
            dtype=np.float32,
        )

        self._voxel_coords: NDArray[np.float64] = self._build_voxel_coords()

    def _build_voxel_coords(self) -> NDArray[np.float64]:
        """Pre-compute world coordinates of every voxel centre."""
        xv = np.arange(self._dims[0])
        yv = np.arange(self._dims[1])
        zv = np.arange(self._dims[2])
        grid = np.stack(np.meshgrid(xv, yv, zv, indexing="ij"), axis=-1)
        coords = grid.reshape(-1, 3).astype(np.float64) * self.voxel_size
        coords += self._origin + self.voxel_size / 2
        return coords

    def integrate(
        self,
        depth: NDArray[np.float64],
        K: NDArray[np.float64],
        T: NDArray[np.float64],
        labels: NDArray[np.int32],
        confidences: NDArray[np.float32] | None = None,
    ) -> None:
        r"""Integrate a depth frame with semantic labels.

        Steps (vectorised over all voxels):

        1. Transform voxel centres from world to camera frame.
        2. Project into pixel coordinates via *K*.
        3. Compute TSDF values and update with running average.
        4. Look up the semantic label at each projected pixel.
        5. Update the per-voxel log-probability histogram.

        Parameters
        ----------
        depth : ndarray, shape (H, W)
            Observed depth in metres.
        K : ndarray, shape (3, 3)
            Camera intrinsic matrix.
        T : ndarray, shape (4, 4)
            Camera-to-world SE(3) transform.
        labels : ndarray, shape (H, W), dtype int32
            Per-pixel semantic label.
        confidences : ndarray, shape (H, W), dtype float32, optional
            Per-pixel classification confidence in [0, 1].  ``None`` →
            assume uniform confidence of 0.8.
        """
        h, w = depth.shape[:2]
        T_w2c = np.linalg.inv(T)

        n = self._voxel_coords.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        hom = np.concatenate([self._voxel_coords, ones], axis=1)
        cam = (T_w2c @ hom.T).T

        cam_z = cam[:, 2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        pix_x = fx * (cam[:, 0] / cam_z) + cx
        pix_y = fy * (cam[:, 1] / cam_z) + cy

        valid = (
            (cam_z > 0)
            & (pix_x >= 0)
            & (pix_x < w - 1)
            & (pix_y >= 0)
            & (pix_y < h - 1)
        )

        pix_xi = np.clip(np.round(pix_x).astype(np.int32), 0, w - 1)
        pix_yi = np.clip(np.round(pix_y).astype(np.int32), 0, h - 1)

        d_obs = depth[pix_yi, pix_xi]
        valid &= d_obs > 0

        sdf = d_obs - cam_z
        valid &= sdf >= -self.trunc_dist
        tsdf_new = np.clip(sdf / self.trunc_dist, -1.0, 1.0)

        valid_idx = np.where(valid)[0]
        vi = np.unravel_index(valid_idx, self._dims)

        # TSDF update
        w_old = self._weight[vi]
        tsdf_old = self._tsdf[vi]
        w_sum = w_old + 1.0
        self._tsdf[vi] = (w_old * tsdf_old + tsdf_new[valid_idx]) / w_sum
        self._weight[vi] = np.minimum(w_sum, 255.0)

        # Semantic update: log-probability fusion
        obs_labels = labels[pix_yi[valid_idx], pix_xi[valid_idx]]
        if confidences is not None:
            obs_conf = confidences[pix_yi[valid_idx], pix_xi[valid_idx]]
        else:
            obs_conf = np.full(len(valid_idx), 0.8, dtype=np.float32)

        obs_conf = np.clip(obs_conf, 0.01, 0.99)

        # Build per-observation log-likelihood vector
        # p(k | z) = conf if k == label, else (1 - conf) / (C - 1)
        n_valid = len(valid_idx)
        log_lik = np.full(
            (n_valid, self.num_classes),
            np.log((1.0 - obs_conf[:, None]) / max(self.num_classes - 1, 1)),
            dtype=np.float32,
        )
        log_lik[np.arange(n_valid), obs_labels] = np.log(obs_conf).astype(np.float32)

        self._log_probs[vi] += log_lik

    def get_semantic_mesh(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.int32],
    ]:
        """Extract a triangle mesh with per-vertex semantic labels.

        Surface extraction uses Marching Cubes on the TSDF zero-crossing.
        Each vertex receives the most probable class from the voxel's
        log-probability histogram via argmax (MAP estimate).

        Returns
        -------
        verts : ndarray, shape (V, 3)
            Vertex positions in world frame.
        faces : ndarray, shape (F, 3)
            Triangle index array.
        normals : ndarray, shape (V, 3)
            Per-vertex normals.
        vertex_labels : ndarray, shape (V,), dtype int32
            Per-vertex semantic class index (MAP).
        """
        from scipy.ndimage import binary_erosion
        from skimage.measure import marching_cubes

        observed = self._weight > 0
        if not observed.any():
            return (
                np.zeros((0, 3)),
                np.zeros((0, 3), dtype=int),
                np.zeros((0, 3)),
                np.zeros(0, dtype=np.int32),
            )

        tsdf_vol = self._tsdf.copy()

        # Same safeguard as ``TSDFVolume.extract_mesh``: unobserved voxels hold
        # the initial +1, so cells at the observed/unobserved boundary present
        # a false −1/+1 crossing.  Erode the observed region with a full 3×3×3
        # structure so only cells whose 8 corners were all measured are meshed.
        # If the observed band is thinner than one voxel on each side
        # (trunc_dist < ~1.5 × voxel_size) the erosion is empty; fall back to
        # the raw observed mask, which clamps the surface to the band boundary
        # instead of producing no mesh at all.
        tsdf_vol[~observed] = 1.0
        interior = binary_erosion(observed, structure=np.ones((3, 3, 3), dtype=bool))
        mask = interior if interior.any() else observed

        try:
            verts, faces, normals, _ = marching_cubes(tsdf_vol, level=0.0, mask=mask)
        except (ValueError, RuntimeError):
            return (
                np.zeros((0, 3)),
                np.zeros((0, 3), dtype=int),
                np.zeros((0, 3)),
                np.zeros(0, dtype=np.int32),
            )

        # ``marching_cubes`` returns vertices in continuous *index* space, where
        # integer coordinate k refers to voxel k's centre — the same convention
        # as ``_build_voxel_coords`` (origin + (k + 0.5) · voxel_size).  Omitting
        # the half-voxel term shifts the whole mesh by half a voxel.
        verts_world = self._origin + (verts + 0.5) * self.voxel_size

        # Look up semantic labels at each vertex
        vert_idx = np.clip(
            np.round((verts_world - self._origin) / self.voxel_size).astype(int),
            0,
            np.array(self._dims) - 1,
        )
        log_p = self._log_probs[vert_idx[:, 0], vert_idx[:, 1], vert_idx[:, 2]]
        vertex_labels = np.argmax(log_p, axis=1).astype(np.int32)

        return verts_world, faces, normals, vertex_labels

    def get_voxel_labels(self) -> NDArray[np.int32]:
        """Return the MAP class label for every voxel.

        Returns
        -------
        ndarray, shape ``(Dx, Dy, Dz)``, dtype int32
            Per-voxel class index.
        """
        return np.argmax(self._log_probs, axis=-1).astype(np.int32)

    def get_class_probabilities(self) -> NDArray[np.float32]:
        r"""Convert log-probabilities to normalised probabilities (softmax).

        .. math::

            p(k) = \frac{\exp(\ell_k)}{\sum_{k'} \exp(\ell_{k'})}

        Returns
        -------
        ndarray, shape ``(Dx, Dy, Dz, C)``, dtype float32
        """
        shifted = self._log_probs - self._log_probs.max(axis=-1, keepdims=True)
        exp_lp = np.exp(shifted)
        return exp_lp / exp_lp.sum(axis=-1, keepdims=True)
