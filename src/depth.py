"""
Neural monocular depth estimation and depth-map utilities.

Monocular depth is inherently ill-posed: a single 2D image admits infinitely
many 3D interpretations (the depth-scale ambiguity).  Neural networks resolve
this by learning geometric priors from large-scale training data.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports for heavy optional dependencies
# ---------------------------------------------------------------------------

def _import_torch():
    """Import and return the ``torch`` module, raising if unavailable."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for neural depth estimation. "
            "Install it with: pip install torch torchvision"
        ) from exc


def _import_transformers():
    """Import and return the ``transformers`` module, raising if unavailable."""
    try:
        import transformers
        return transformers
    except ImportError as exc:
        raise ImportError(
            "HuggingFace Transformers is required for neural depth estimation. "
            "Install it with: pip install transformers"
        ) from exc


# ---------------------------------------------------------------------------
# DepthEstimator
# ---------------------------------------------------------------------------

class DepthEstimator:
    """
    Neural monocular depth estimation.

    Monocular depth is inherently ill-posed: a single image has infinitely
    many possible 3D interpretations (the depth-scale ambiguity).  Neural
    networks resolve this using learned priors from training data.

    **Supported backends**:

    - ``"depth_anything_v2"``: State-of-the-art encoder-decoder with a
      DINOv2 backbone and a DPT (Dense Prediction Transformer) head.
    - ``"midas"``: The original MiDaS model — older but robust across
      diverse scenes.

    **Output semantics**:

    The network produces *relative* (affine-invariant) depth::

        depth_metric = α · depth_network + β

    where α (scale) and β (shift) must be estimated from external
    information (known object size, stereo pair, IMU, etc.).

    **DPT architecture overview**:

    1. **ViT backbone** processes all patches at a single resolution;
       intermediate layers (e.g. 4, 11, 17, 23 of ViT-L) are tapped
       and reassembled into multi-scale feature maps.
    2. **Reassemble** projects tokens back to spatial feature maps via
       learned linear projections + spatial reshaping.
    3. **Fusion** progressively up-samples and fuses features with
       residual convolution blocks and skip connections.
    4. **Head** predicts per-pixel inverse depth with a final conv layer.

    **Scale recovery methods** (to obtain metric depth):

    - Known object size → single-point scale
    - Stereo pair → median depth ratio  α = median(d_stereo / d_mono)
    - IMU + visual-inertial scale factor
    """

    _SUPPORTED_BACKENDS = {"depth_anything_v2", "midas"}

    def __init__(
        self,
        backend: str = "depth_anything_v2",
        device: str = "cpu",
    ) -> None:
        """
        Initialise a neural depth model.

        Parameters
        ----------
        backend : str
            One of ``"depth_anything_v2"`` or ``"midas"``.
        device : str
            PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        """
        if backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose from {sorted(self._SUPPORTED_BACKENDS)}."
            )

        self.backend = backend
        self.device = device
        self._model = None
        self._transform = None

    def _load_model(self) -> None:
        """Lazy-load the model on first prediction."""
        if self._model is not None:
            return

        torch = _import_torch()

        if self.backend == "depth_anything_v2":
            self._load_depth_anything_v2(torch)
        else:
            self._load_midas(torch)

    def _load_depth_anything_v2(self, torch) -> None:
        """Load the Depth Anything V2 model via a HuggingFace pipeline."""
        transformers = _import_transformers()

        pipe = transformers.pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=self.device,
        )
        self._model = pipe
        self._transform = None

    def _load_midas(self, torch) -> None:
        """Load the MiDaS small model from torch hub."""
        self._model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True,
        )
        self._model.to(self.device).eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True,
        )
        self._transform = midas_transforms.small_transform

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict a relative depth map from an RGB image.

        The output is *inverse* relative depth: higher values indicate
        pixels closer to the camera.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR (OpenCV convention) or RGB, shape (H, W, 3).

        Returns
        -------
        depth : np.ndarray
            (H, W) float32 array of relative (inverse) depth values.
        """
        self._load_model()

        torch = _import_torch()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.backend == "depth_anything_v2":
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(rgb)
            result = self._model(pil_img)
            depth_pil = result["depth"]
            depth = np.array(depth_pil, dtype=np.float32)
            if depth.shape[:2] != image.shape[:2]:
                depth = cv2.resize(
                    depth, (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            return depth

        input_tensor = self._transform(rgb).to(self.device)
        with torch.no_grad():
            prediction = self._model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy().astype(np.float32)

    def predict_metric(
        self,
        image: np.ndarray,
        scale: float = 1.0,
        shift: float = 0.0,
    ) -> np.ndarray:
        """
        Predict depth and apply an affine correction to obtain metric depth.

        ::

            depth_metric = scale · depth_network + shift

        Parameters
        ----------
        image : np.ndarray
            BGR input image.
        scale : float
            Multiplicative scale factor  α.
        shift : float
            Additive shift  β.

        Returns
        -------
        depth : np.ndarray
            (H, W) metric depth map.
        """
        relative = self.predict(image)
        return scale * relative + shift

    @staticmethod
    def align_to_reference(
        relative_depth: np.ndarray,
        reference_points: np.ndarray,
    ) -> np.ndarray:
        """Align relative depth to metric scale using known reference distances.

        Finds scale *s* and shift *b* minimising:

        .. math::

            \\min_{s, b} \\sum_i (s \\cdot d_{\\text{pred},i} + b - d_{\\text{ref},i})^2

        Parameters
        ----------
        relative_depth : (H, W) relative depth from ``predict()``
        reference_points : (N, 3) array where each row is ``(u, v, depth_meters)``

        Returns
        -------
        metric_depth : (H, W) scaled depth map in metres
        """
        us = reference_points[:, 0].astype(int)
        vs = reference_points[:, 1].astype(int)
        d_ref = reference_points[:, 2]
        d_pred = relative_depth[vs, us].astype(np.float64)

        A = np.stack([d_pred, np.ones_like(d_pred)], axis=-1)
        result, _, _, _ = np.linalg.lstsq(A, d_ref, rcond=None)
        s, b = float(result[0]), float(result[1])

        return (s * relative_depth + b).astype(np.float32)


# ---------------------------------------------------------------------------
# Depth map utilities
# ---------------------------------------------------------------------------

def align_depth_to_ground_truth(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute optimal scale and shift to align predicted depth to ground truth.

    Solves the least-squares problem::

        min_{s, b}  Σᵢ (s · dᵢ_pred + b − dᵢ_gt)²

    over valid pixels.  Closed-form via the normal equations:

        | Σ dᵢ²   Σ dᵢ | | s |   | Σ dᵢ · gᵢ |
        | Σ dᵢ     N    | | b | = | Σ gᵢ       |

    Parameters
    ----------
    predicted : np.ndarray
        (H, W) predicted depth map.
    ground_truth : np.ndarray
        (H, W) ground-truth depth map.
    mask : np.ndarray or None
        Boolean mask of valid pixels. ``None`` → use all pixels where
        ground_truth > 0.

    Returns
    -------
    scale : float
        Optimal scale  s.
    shift : float
        Optimal shift  b.
    aligned : np.ndarray
        Aligned depth map  s · predicted + b.
    """
    if mask is None:
        mask = ground_truth > 0

    d = predicted[mask].astype(np.float64)
    g = ground_truth[mask].astype(np.float64)

    if len(d) < 2:
        return 1.0, 0.0, predicted.copy()

    A = np.stack([d, np.ones_like(d)], axis=-1)
    result, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
    scale, shift = float(result[0]), float(result[1])

    aligned = (scale * predicted + shift).astype(np.float32)
    return scale, shift, aligned


def compute_depth_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute standard monocular depth evaluation metrics.

    All metrics are computed over valid pixels only.

    **Metrics**:

    - ``abs_rel``:  mean( |d_pred − d_gt| / d_gt )
    - ``sq_rel``:   mean( (d_pred − d_gt)² / d_gt )
    - ``rmse``:     sqrt( mean( (d_pred − d_gt)² ) )
    - ``rmse_log``: sqrt( mean( (log d_pred − log d_gt)² ) )
    - ``δ₁``:       % pixels where max(d_pred/d_gt, d_gt/d_pred) < 1.25
    - ``δ₂``:       % pixels where max(d_pred/d_gt, d_gt/d_pred) < 1.25²
    - ``δ₃``:       % pixels where max(d_pred/d_gt, d_gt/d_pred) < 1.25³

    Parameters
    ----------
    predicted : np.ndarray
        (H, W) predicted metric depth.
    ground_truth : np.ndarray
        (H, W) ground-truth metric depth.
    mask : np.ndarray or None
        Boolean mask.  ``None`` → use pixels where gt > 0.

    Returns
    -------
    metrics : dict
        Dictionary with keys ``abs_rel``, ``sq_rel``, ``rmse``,
        ``rmse_log``, ``delta_1``, ``delta_2``, ``delta_3``.
    """
    if mask is None:
        mask = ground_truth > 0

    pred = predicted[mask].astype(np.float64)
    gt = ground_truth[mask].astype(np.float64)

    if len(pred) == 0:
        return {k: 0.0 for k in (
            "abs_rel", "sq_rel", "rmse", "rmse_log",
            "delta_1", "delta_2", "delta_3",
        )}

    pred = np.clip(pred, 1e-6, None)

    diff = np.abs(pred - gt)

    abs_rel = float(np.mean(diff / gt))
    sq_rel = float(np.mean(diff ** 2 / gt))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2)))

    ratio = np.maximum(pred / gt, gt / pred)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < 1.25 ** 2))
    delta_3 = float(np.mean(ratio < 1.25 ** 3))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
    }


def colorize_depth(
    depth: np.ndarray,
    cmap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Colorize a depth map for visualization.

    Maps scalar depth values through a colour map and returns an 8-bit
    BGR image suitable for ``cv2.imshow`` or saving with ``cv2.imwrite``.

    Parameters
    ----------
    depth : np.ndarray
        (H, W) depth map (float).
    cmap : str
        Matplotlib colour-map name (e.g. ``"turbo"``, ``"magma"``,
        ``"inferno"``).
    vmin, vmax : float or None
        Clamp range.  ``None`` → derived from the depth map.

    Returns
    -------
    coloured : np.ndarray
        (H, W, 3) uint8 BGR image.
    """
    import matplotlib.cm as cm

    valid = depth[depth > 0] if np.any(depth > 0) else depth.ravel()
    lo = vmin if vmin is not None else float(np.percentile(valid, 2))
    hi = vmax if vmax is not None else float(np.percentile(valid, 98))

    normalised = np.clip((depth - lo) / max(hi - lo, 1e-8), 0.0, 1.0)

    colormap = cm.get_cmap(cmap)
    coloured = (colormap(normalised)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)
