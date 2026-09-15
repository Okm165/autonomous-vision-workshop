"""Visualisation helpers for 3-D mapping and multi-view geometry.

This module provides publication-quality plotting utilities for:

* Camera poses as 3-D coordinate frames with optional frustums
* 3-D point clouds
* Depth maps with colour bars
* Feature matches between image pairs
* 2-D covariance ellipses (e.g. for EKF state uncertainty)
* Animation frame generation

All plotting uses Matplotlib for portability.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# ======================================================================
#  3-D camera visualisation
# ======================================================================

def plot_cameras_3d(
    poses: Sequence[NDArray[np.float64]],
    K: Optional[NDArray[np.float64]] = None,
    ax: Optional[object] = None,
    scale: float = 0.3,
    colors: Optional[Sequence] = None,
) -> object:
    r"""Visualise camera poses as coordinate frames in 3-D.

    Each camera is drawn as three arrows representing the local X (red),
    Y (green), and Z (blue) axes.  If intrinsics *K* are provided, a
    viewing frustum is also drawn.

    Parameters
    ----------
    poses : sequence of ndarray, each (4, 4)
        Camera-to-world SE(3) transformations.
    K : ndarray, shape (3, 3), optional
        Camera intrinsics for frustum drawing.
    ax : matplotlib Axes3D, optional
        Existing 3-D axes.  If ``None``, a new figure is created.
    scale : float
        Length of the axis arrows in world units.
    colors : sequence, optional
        Per-camera colours.  If ``None``, all cameras share a common
        colour scheme.

    Returns
    -------
    ax : matplotlib Axes3D
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    for i, T in enumerate(poses):
        R = T[:3, :3]
        t = T[:3, 3]

        # Axis directions in world frame
        x_dir = R @ np.array([scale, 0, 0])
        y_dir = R @ np.array([0, scale, 0])
        z_dir = R @ np.array([0, 0, scale])

        ax.quiver(*t, *x_dir, color="r", arrow_length_ratio=0.1, linewidth=1.5)
        ax.quiver(*t, *y_dir, color="g", arrow_length_ratio=0.1, linewidth=1.5)
        ax.quiver(*t, *z_dir, color="b", arrow_length_ratio=0.1, linewidth=1.5)

        # Frustum
        if K is not None:
            _draw_frustum(ax, T, K, scale * 0.8, colors[i] if colors else "cyan")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_equal_aspect_3d(ax)

    return ax


def _draw_frustum(
    ax: object,
    T: NDArray[np.float64],
    K: NDArray[np.float64],
    depth: float,
    color: str = "cyan",
) -> None:
    """Draw a camera frustum pyramid."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    hw = cx * depth / fx
    hh = cy * depth / fy

    corners_cam = np.array([
        [-hw, -hh, depth],
        [hw, -hh, depth],
        [hw, hh, depth],
        [-hw, hh, depth],
    ])

    R, t = T[:3, :3], T[:3, 3]
    corners_world = (R @ corners_cam.T).T + t

    for c in corners_world:
        ax.plot3D([t[0], c[0]], [t[1], c[1]], [t[2], c[2]],
                  color=color, alpha=0.4, linewidth=0.8)

    rect = np.vstack([corners_world, corners_world[0:1]])
    ax.plot3D(rect[:, 0], rect[:, 1], rect[:, 2],
              color=color, alpha=0.4, linewidth=0.8)


# ======================================================================
#  Point cloud plot
# ======================================================================

def plot_pointcloud_3d(
    points: NDArray[np.float64],
    colors: Optional[NDArray] = None,
    ax: Optional[object] = None,
    subsample: int = 5000,
    point_size: float = 1.0,
    title: str = "Point Cloud",
) -> object:
    """Plot a 3-D point cloud in Matplotlib.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        3-D point coordinates.
    colors : ndarray, shape (N, 3), optional
        Per-point RGB colours in [0, 1] or [0, 255].
    ax : matplotlib Axes3D, optional
        Existing axes.  ``None`` → create new.
    subsample : int
        Maximum number of points to display (random subset).
    point_size : float
        Marker size.
    title : str
        Plot title.

    Returns
    -------
    ax : matplotlib Axes3D
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    n = len(points)
    if n > subsample:
        idx = np.random.default_rng(42).choice(n, subsample, replace=False)
        pts = points[idx]
        clr = colors[idx] if colors is not None else None
    else:
        pts = points
        clr = colors

    if clr is not None:
        clr = np.asarray(clr, dtype=np.float64)
        if clr.max() > 1.0:
            clr = clr / 255.0
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=clr, s=point_size, marker=".")
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=point_size, marker=".", alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    _set_equal_aspect_3d(ax)

    return ax


# ======================================================================
#  Depth map plot
# ======================================================================

def plot_depth_map(
    depth: NDArray[np.float64],
    title: str = "Depth Map",
    cmap: str = "turbo",
    ax: Optional[object] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> object:
    """Visualise a depth map with a colour bar.

    Parameters
    ----------
    depth : ndarray, shape (H, W)
        Depth map (float, metres).
    title : str
        Plot title.
    cmap : str
        Matplotlib colourmap name.
    ax : matplotlib Axes, optional
    vmin, vmax : float, optional
        Colour-scale limits.

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    valid = depth[depth > 0]
    if vmin is None:
        vmin = float(np.percentile(valid, 2)) if len(valid) > 0 else 0.0
    if vmax is None:
        vmax = float(np.percentile(valid, 98)) if len(valid) > 0 else 1.0

    im = ax.imshow(depth, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Depth (m)")

    return ax


# ======================================================================
#  Feature match visualisation
# ======================================================================

def plot_matches(
    img1: NDArray[np.uint8],
    kp1: NDArray[np.float64],
    img2: NDArray[np.uint8],
    kp2: NDArray[np.float64],
    matches: Sequence[Tuple[int, int]],
    inlier_mask: Optional[NDArray[np.bool_]] = None,
    max_draw: int = 50,
) -> object:
    """Draw feature matches between two images side by side.

    Inliers are drawn in green, outliers in red.

    Parameters
    ----------
    img1, img2 : ndarray, shape (H, W, 3)
        Two images.
    kp1, kp2 : ndarray, shape (N, 2)
        Keypoint coordinates ``(x, y)`` for each image.
    matches : sequence of (idx1, idx2)
        Index pairs into *kp1* and *kp2*.
    inlier_mask : ndarray of bool, shape (M,), optional
        Per-match inlier flag.
    max_draw : int
        Maximum number of matches to draw.

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h_out = max(h1, h2)
    w_out = w1 + w2

    canvas = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1 if img1.ndim == 3 else np.stack([img1] * 3, axis=-1)
    canvas[:h2, w1:w1 + w2] = img2 if img2.ndim == 3 else np.stack([img2] * 3, axis=-1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas[:, :, ::-1] if canvas.shape[2] == 3 else canvas)

    n_draw = min(len(matches), max_draw)
    rng = np.random.default_rng(42)
    draw_idx = (rng.choice(len(matches), n_draw, replace=False)
                 if len(matches) > n_draw
                 else np.arange(len(matches)))

    for k in draw_idx:
        i1, i2 = matches[k]
        x1, y1 = kp1[i1]
        x2, y2 = kp2[i2]

        is_inlier = inlier_mask[k] if inlier_mask is not None else True
        color = "lime" if is_inlier else "red"
        alpha = 0.8 if is_inlier else 0.3

        ax.plot([x1, x2 + w1], [y1, y2], "-", color=color, alpha=alpha, linewidth=0.5)
        ax.plot(x1, y1, "o", color=color, markersize=3)
        ax.plot(x2 + w1, y2, "o", color=color, markersize=3)

    ax.set_title(f"Feature Matches ({n_draw} shown)")
    ax.axis("off")

    return ax


# ======================================================================
#  Animation helper
# ======================================================================

def make_video_frames(
    images: Sequence[NDArray[np.uint8]],
    fps: int = 10,
) -> List[NDArray[np.uint8]]:
    """Create a list of frames suitable for animation / video encoding.

    Pads images to consistent dimensions and converts grayscale to BGR.

    Parameters
    ----------
    images : sequence of ndarray
        Input images (may differ in size).
    fps : int
        Target frame rate (metadata only; returned list is not time-stamped).

    Returns
    -------
    list of ndarray
        Uniformly sized BGR frames.
    """
    if not images:
        return []

    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    frames: List[NDArray[np.uint8]] = []
    for img in images:
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        h, w = img.shape[:2]
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        canvas[:h, :w] = img
        frames.append(canvas)

    return frames


# ======================================================================
#  Covariance ellipse
# ======================================================================

def plot_covariance_ellipse(
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
    ax: Optional[object] = None,
    n_std: float = 2.0,
    color: str = "blue",
    alpha: float = 0.3,
    label: Optional[str] = None,
) -> object:
    r"""Plot a 2-D covariance ellipse.

    The ellipse represents the iso-contour of the Gaussian density at
    :math:`n_\sigma` standard deviations:

    .. math::

        (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1}
        (\mathbf{x} - \boldsymbol{\mu}) = n_\sigma^2

    The semi-axes are :math:`n_\sigma \sqrt{\lambda_i}` along the
    eigenvectors of :math:`\Sigma`.

    Parameters
    ----------
    mean : ndarray, shape (2,)
        Centre of the ellipse.
    cov : ndarray, shape (2, 2)
        Covariance matrix (must be positive semi-definite).
    ax : matplotlib Axes, optional
    n_std : float
        Number of standard deviations for the ellipse radius.
    color : str
    alpha : float
        Fill transparency.
    label : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    if ax is None:
        _, ax = plt.subplots()

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2.0 * n_std * np.sqrt(eigenvalues[0])
    height = 2.0 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=mean, width=width, height=height, angle=angle,
        edgecolor=color, facecolor=color, alpha=alpha, label=label,
    )
    ax.add_patch(ellipse)
    ax.plot(*mean, "x", color=color, markersize=5)

    return ax


# ======================================================================
#  Helper: create 3-D axes
# ======================================================================

def create_3d_axes(
    figsize: Tuple[int, int] = (10, 8),
    title: str = "",
) -> object:
    """Create Matplotlib 3-D axes with equal aspect ratio.

    Parameters
    ----------
    figsize : (int, int)
        Figure size in inches.
    title : str
        Axes title.

    Returns
    -------
    ax : matplotlib Axes3D
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title:
        ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return ax


# ======================================================================
#  Trajectory plot
# ======================================================================

def plot_trajectory(
    poses: Sequence[NDArray[np.float64]],
    gt_poses: Optional[Sequence[NDArray[np.float64]]] = None,
    title: str = "Trajectory",
    ax: Optional[object] = None,
) -> object:
    """Plot camera trajectory from SE(3) poses (bird's-eye view).

    Parameters
    ----------
    poses : sequence of (4, 4) SE(3) matrices
    gt_poses : optional ground-truth poses for comparison
    title : str
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    pos = np.array([T[:3, 3] for T in poses])
    ax.plot(pos[:, 0], pos[:, 2], "b-o", markersize=2, label="Estimated")

    if gt_poses is not None:
        gt_pos = np.array([T[:3, 3] for T in gt_poses])
        ax.plot(gt_pos[:, 0], gt_pos[:, 2], "g--", linewidth=2, label="Ground Truth")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return ax


# ======================================================================
#  Depth colourmap
# ======================================================================

def depth_colormap(
    depth: NDArray[np.float64],
    cmap: str = "turbo",
    invalid_value: float = 0.0,
) -> NDArray[np.uint8]:
    """Convert a depth map to a coloured image.

    Parameters
    ----------
    depth : (H, W) float array
    cmap : Matplotlib colourmap name
    invalid_value : value marking invalid depth (set to black)

    Returns
    -------
    coloured : (H, W, 3) uint8 array
    """
    import matplotlib.cm as cm

    valid_mask = depth > invalid_value
    d = depth.copy().astype(np.float64)
    if valid_mask.any():
        vmin = float(np.percentile(d[valid_mask], 2))
        vmax = float(np.percentile(d[valid_mask], 98))
        d = np.clip((d - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    else:
        d[:] = 0

    mapper = cm.get_cmap(cmap)
    coloured = (mapper(d)[:, :, :3] * 255).astype(np.uint8)
    coloured[~valid_mask] = 0

    return coloured


# ======================================================================
#  Optical flow colour visualisation
# ======================================================================

def flow_to_color(
    flow: NDArray[np.float64],
    max_flow: Optional[float] = None,
) -> NDArray[np.uint8]:
    """HSV colour-wheel visualisation for 2-D optical flow.

    Hue encodes direction, saturation encodes magnitude.

    Parameters
    ----------
    flow : (H, W, 2) optical flow field (dx, dy)
    max_flow : maximum magnitude for normalisation (auto if None)

    Returns
    -------
    rgb : (H, W, 3) uint8 image
    """
    import colorsys

    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)

    if max_flow is None:
        max_flow = float(np.percentile(mag, 99)) + 1e-5

    hue = (angle + np.pi) / (2 * np.pi)
    sat = np.clip(mag / max_flow, 0, 1)
    val = np.ones_like(hue)

    hsv = np.stack([hue, sat, val], axis=-1)

    H, W = flow.shape[:2]
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            r, g, b = colorsys.hsv_to_rgb(hsv[i, j, 0], hsv[i, j, 1], hsv[i, j, 2])
            rgb[i, j] = [int(r * 255), int(g * 255), int(b * 255)]

    return rgb


# ======================================================================
#  Side-by-side comparison
# ======================================================================

def side_by_side_comparison(
    images: Sequence[NDArray],
    titles: Sequence[str],
    figsize: Optional[Tuple[int, int]] = None,
    cmaps: Optional[Sequence[Optional[str]]] = None,
) -> object:
    """Display multiple images side by side.

    Parameters
    ----------
    images : sequence of ndarray
    titles : sequence of str
    figsize : optional figure size
    cmaps : optional per-image colormaps

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n = len(images)
    if figsize is None:
        figsize = (5 * n, 5)
    if cmaps is None:
        cmaps = [None] * n

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        if cmap is not None:
            ax.imshow(img, cmap=cmap)
        elif img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    return fig


# ======================================================================
#  3-D scene rendering
# ======================================================================

def render_3d_scene(
    points: Optional[NDArray] = None,
    poses: Optional[Sequence[NDArray]] = None,
    K: Optional[NDArray] = None,
    point_colors: Optional[NDArray] = None,
    title: str = "3D Scene",
    figsize: Tuple[int, int] = (12, 8),
) -> object:
    """Render a combined 3-D scene with point cloud and camera poses.

    Parameters
    ----------
    points : (N, 3) array, optional
    poses : sequence of (4, 4) SE(3) matrices, optional
    K : (3, 3) intrinsics, optional (for frustum)
    point_colors : (N, 3) array, optional
    title : str
    figsize : tuple

    Returns
    -------
    ax : matplotlib Axes3D
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if points is not None:
        ax = plot_pointcloud_3d(points, colors=point_colors, ax=ax,
                                title="", point_size=0.5)

    if poses is not None:
        ax = plot_cameras_3d(poses, K=K, ax=ax, scale=0.2)

    ax.set_title(title)
    return ax


# ======================================================================
#  Internal helpers
# ======================================================================

def _set_equal_aspect_3d(ax: object) -> None:
    """Set equal aspect ratio on a 3-D Matplotlib axes."""
    try:
        ax.set_box_aspect([1, 1, 1])
    except (AttributeError, TypeError):
        pass
