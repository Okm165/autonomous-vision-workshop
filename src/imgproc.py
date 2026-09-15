"""
Image Processing Fundamentals
==============================

Pure-numpy implementations of core image-processing algorithms for a
vision & 3D-mapping workshop.  OpenCV is intentionally avoided for every
computational kernel so that students see the underlying mathematics.

Conventions
-----------
* Images are ``numpy.ndarray`` with dtype ``float64`` in [0, 1] or
  ``uint8`` in [0, 255].  Functions that need floating-point values
  convert internally and document what they return.
* Grayscale images have shape ``(H, W)``; colour images ``(H, W, C)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 1. 2-D Convolution
# ---------------------------------------------------------------------------

def convolve2d(
    image: NDArray[np.floating],
    kernel: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Perform 2-D convolution from scratch (no OpenCV).

    The discrete 2-D convolution is defined as:

    .. math::

        (f * g)(x, y) = \sum_{i} \sum_{j} f(x - i,\; y - j)\;\cdot\; g(i, j)

    Because the kernel is flipped in a true convolution (as opposed to
    *correlation*), we flip ``kernel`` before sliding it across the image.

    Parameters
    ----------
    image : ndarray, shape (H, W) or (H, W, C)
        Input image (grayscale or colour).
    kernel : ndarray, shape (kH, kW)
        2-D convolution kernel.

    Returns
    -------
    ndarray
        Convolved image with the same spatial dimensions as *image*.
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    flipped = kernel[::-1, ::-1]

    if image.ndim == 2:
        return _convolve2d_single(image, flipped, pad_h, pad_w)

    channels = [
        _convolve2d_single(image[:, :, c], flipped, pad_h, pad_w)
        for c in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)


def _convolve2d_single(
    img: NDArray[np.float64],
    flipped_kernel: NDArray[np.float64],
    pad_h: int,
    pad_w: int,
) -> NDArray[np.float64]:
    """Convolve a single-channel image with an already-flipped kernel."""
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    kh, kw = flipped_kernel.shape
    h, w = img.shape
    out = np.empty((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            region = padded[y : y + kh, x : x + kw]
            out[y, x] = np.sum(region * flipped_kernel)
    return out


# ---------------------------------------------------------------------------
# 2. Gaussian Kernel
# ---------------------------------------------------------------------------

def gaussian_kernel(size: int, sigma: float) -> NDArray[np.float64]:
    r"""Generate a normalised 2-D Gaussian kernel.

    .. math::

        G(x, y) = \frac{1}{2\pi\sigma^{2}}
                   \exp\!\Bigl(-\frac{x^{2} + y^{2}}{2\sigma^{2}}\Bigr)

    The kernel is centred so that the peak sits at the middle element.
    After evaluation the values are divided by their sum so that the
    kernel integrates (sums) to exactly 1 — preserving image brightness
    when used for blurring.

    Parameters
    ----------
    size : int
        Side length of the square kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    ndarray, shape (size, size)
        Normalised Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= 2.0 * np.pi * sigma ** 2
    kernel /= kernel.sum()
    return kernel


# ---------------------------------------------------------------------------
# 3. Sobel Gradients
# ---------------------------------------------------------------------------

def sobel_gradients(
    image: NDArray[np.floating],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""Compute image gradients with the Sobel operator.

    The 3×3 Sobel kernels approximate the first-order partial derivatives
    of image intensity:

    .. math::

        S_x = \begin{bmatrix}-1 & 0 & 1\\-2 & 0 & 2\\-1 & 0 & 1\end{bmatrix},\quad
        S_y = \begin{bmatrix}-1 & -2 & -1\\0 & 0 & 0\\1 & 2 & 1\end{bmatrix}

    Gradient magnitude and direction:

    .. math::

        M = \sqrt{G_x^{2} + G_y^{2}},\qquad
        \theta = \operatorname{atan2}(G_y,\; G_x)

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Grayscale input image.

    Returns
    -------
    gradient_x : ndarray  – horizontal gradient
    gradient_y : ndarray  – vertical gradient
    magnitude  : ndarray  – gradient magnitude
    direction  : ndarray  – gradient direction in radians (−π, π]
    """
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    image = image.astype(np.float64)

    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float64)

    gx = convolve2d(image, sobel_x)
    gy = convolve2d(image, sobel_y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    return gx, gy, magnitude, direction


# ---------------------------------------------------------------------------
# 4. Canny Edge Detector
# ---------------------------------------------------------------------------

def canny_edges(
    image: NDArray[np.floating],
    low_thresh: float,
    high_thresh: float,
    sigma: float = 1.4,
) -> NDArray[np.uint8]:
    r"""Full Canny edge detector built from scratch.

    **Step 1 — Gaussian smoothing**
        Blur with :func:`gaussian_kernel` of size
        ``2 * ceil(3σ) + 1`` to suppress noise.

    **Step 2 — Gradient computation**
        :func:`sobel_gradients` yields magnitude *M* and direction *θ*.

    **Step 3 — Non-maximum suppression (NMS)**
        For each pixel, check the two neighbours along the gradient
        direction (quantised to 0°, 45°, 90°, 135°).  Keep the pixel
        only if its magnitude is the local maximum in that direction —
        this thins edges to one-pixel width.

    **Step 4 — Double thresholding + hysteresis**
        * *Strong* edges: ``M ≥ high_thresh``
        * *Weak* edges:  ``low_thresh ≤ M < high_thresh``
        * *Suppressed*:  ``M < low_thresh``

        A weak edge survives only if it is 8-connected to at least one
        strong edge (flood-fill / iterative dilation of the strong set).

    Parameters
    ----------
    image : ndarray
        Input image (converted to grayscale internally if colour).
    low_thresh, high_thresh : float
        Hysteresis thresholds (applied to the gradient magnitude which
        is *not* normalised — typical values depend on image dtype).
    sigma : float, default 1.4
        Gaussian smoothing σ.

    Returns
    -------
    ndarray, dtype uint8
        Binary edge map (255 = edge, 0 = non-edge).
    """
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    image = image.astype(np.float64)

    # Step 1: Gaussian smoothing
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    gk = gaussian_kernel(ksize, sigma)
    smoothed = convolve2d(image, gk)

    # Step 2: Sobel gradients
    _, _, magnitude, direction = sobel_gradients(smoothed)

    # Step 3: Non-maximum suppression
    nms = _non_maximum_suppression(magnitude, direction)

    # Step 4: Double threshold + hysteresis
    edges = _hysteresis(nms, low_thresh, high_thresh)
    return edges


def _non_maximum_suppression(
    magnitude: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Thin edges by suppressing non-maximum pixels along the gradient."""
    h, w = magnitude.shape
    nms = np.zeros_like(magnitude)

    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180.0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = angle[y, x]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[y, x - 1], magnitude[y, x + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                n1, n2 = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]

            if magnitude[y, x] >= n1 and magnitude[y, x] >= n2:
                nms[y, x] = magnitude[y, x]
    return nms


def _hysteresis(
    nms: NDArray[np.float64],
    low: float,
    high: float,
) -> NDArray[np.uint8]:
    """Double thresholding followed by hysteresis edge tracking."""
    strong = nms >= high
    weak = (nms >= low) & (~strong)

    edges = np.zeros_like(nms, dtype=np.uint8)
    edges[strong] = 255

    changed = True
    while changed:
        changed = False
        for y in range(1, edges.shape[0] - 1):
            for x in range(1, edges.shape[1] - 1):
                if weak[y, x] and edges[y, x] == 0:
                    if np.any(edges[y - 1 : y + 2, x - 1 : x + 2] == 255):
                        edges[y, x] = 255
                        changed = True
    return edges


# ---------------------------------------------------------------------------
# 5. Gaussian Pyramid
# ---------------------------------------------------------------------------

def build_gaussian_pyramid(
    image: NDArray[np.floating],
    levels: int,
) -> list[NDArray[np.float64]]:
    r"""Build a Gaussian image pyramid by repeated blur + downsample.

    At each level *l* the image is smoothed with a 5×5 Gaussian
    (σ = 1.0) and then sub-sampled by a factor of 2 in each dimension:

    .. math::

        G_{l+1}(x, y) = \bigl[\,G_l * g\,\bigr](2x,\; 2y)

    Parameters
    ----------
    image : ndarray
        Base image (level 0).
    levels : int
        Total number of pyramid levels (including the base).

    Returns
    -------
    list[ndarray]
        ``pyramid[0]`` is the original; ``pyramid[-1]`` is the coarsest.
    """
    image = image.astype(np.float64)
    gk = gaussian_kernel(5, 1.0)
    pyramid: list[NDArray[np.float64]] = [image]

    for _ in range(1, levels):
        blurred = convolve2d(pyramid[-1], gk)
        if blurred.ndim == 3:
            downsampled = blurred[::2, ::2, :]
        else:
            downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)
    return pyramid


# ---------------------------------------------------------------------------
# 6. Laplacian Pyramid
# ---------------------------------------------------------------------------

def _upsample(image: NDArray[np.float64], target_shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Upsample *image* to *target_shape* using zero-insertion + Gaussian blur.

    Insert zeros between every sample, then convolve with a 5×5 Gaussian
    scaled by 4 to compensate for the inserted zeros.
    """
    th, tw = target_shape[:2]
    if image.ndim == 3:
        upsampled = np.zeros((th, tw, image.shape[2]), dtype=np.float64)
        upsampled[::2, ::2, :] = image[: (th + 1) // 2, : (tw + 1) // 2, :]
    else:
        upsampled = np.zeros((th, tw), dtype=np.float64)
        upsampled[::2, ::2] = image[: (th + 1) // 2, : (tw + 1) // 2]

    gk = gaussian_kernel(5, 1.0) * 4.0
    return convolve2d(upsampled, gk)


def build_laplacian_pyramid(
    image: NDArray[np.floating],
    levels: int,
) -> list[NDArray[np.float64]]:
    r"""Build a Laplacian image pyramid.

    The Laplacian at level *l* captures the detail lost between two
    successive Gaussian levels:

    .. math::

        L_l = G_l - \operatorname{upsample}(G_{l+1})

    The last element of the returned list is the residual (smallest
    Gaussian level), which is needed for perfect reconstruction.

    Parameters
    ----------
    image : ndarray
        Base image.
    levels : int
        Number of Laplacian levels (the residual is appended as an
        extra entry, so ``len(result) == levels``).

    Returns
    -------
    list[ndarray]
        Laplacian pyramid with ``levels`` entries.  The last entry is the
        low-frequency residual from the Gaussian pyramid.
    """
    gpyr = build_gaussian_pyramid(image, levels)
    lpyr: list[NDArray[np.float64]] = []

    for i in range(len(gpyr) - 1):
        upsampled = _upsample(gpyr[i + 1], gpyr[i].shape)
        lpyr.append(gpyr[i] - upsampled)

    lpyr.append(gpyr[-1])
    return lpyr


def reconstruct_from_laplacian(
    laplacian_pyramid: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    r"""Reconstruct an image from its Laplacian pyramid.

    Starting from the coarsest residual, repeatedly upsample and add the
    corresponding Laplacian detail band:

    .. math::

        G_l = L_l + \operatorname{upsample}(G_{l+1})

    Parameters
    ----------
    laplacian_pyramid : list[ndarray]
        Pyramid produced by :func:`build_laplacian_pyramid`.

    Returns
    -------
    ndarray
        Reconstructed image (same shape and dtype as the original base).
    """
    current = laplacian_pyramid[-1].astype(np.float64)

    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        upsampled = _upsample(current, laplacian_pyramid[i].shape)
        current = laplacian_pyramid[i] + upsampled
    return current


# ---------------------------------------------------------------------------
# 7. Histogram Equalisation
# ---------------------------------------------------------------------------

def histogram_equalize(image: NDArray) -> NDArray[np.uint8]:
    r"""Histogram equalisation for contrast enhancement.

    For a grayscale image with *L* intensity levels the transform is:

    .. math::

        s_k = \operatorname{round}\!\Bigl((L - 1)\;\sum_{j=0}^{k}
              \frac{n_j}{N}\Bigr)

    where *n_j* is the count of pixels with intensity *j* and
    *N = H × W* is the total pixel count.  The CDF of the output
    histogram is approximately uniform, maximising contrast.

    For colour images the transform is applied to the *value* channel
    after converting to HSV-like decomposition (max of channels),
    preserving chrominance ratios.

    Parameters
    ----------
    image : ndarray, shape (H, W) or (H, W, C)
        Input image with integer intensities in [0, 255] or float in
        [0, 1] (auto-detected and converted).

    Returns
    -------
    ndarray, dtype uint8
        Equalised image in [0, 255].
    """
    if image.dtype.kind == "f":
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        return _equalize_channel(image)

    value = np.max(image, axis=2).astype(np.float64)
    eq_value = _equalize_channel(value.astype(np.uint8)).astype(np.float64)

    scale = np.ones_like(value)
    nonzero = value > 0
    scale[nonzero] = eq_value[nonzero] / value[nonzero]

    result = image.astype(np.float64) * scale[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)


def _equalize_channel(channel: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Equalise a single uint8 channel using CDF mapping."""
    hist = np.zeros(256, dtype=np.int64)
    for val in channel.ravel():
        hist[val] += 1

    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    total = channel.size

    lut = np.zeros(256, dtype=np.uint8)
    denom = total - cdf_min
    if denom > 0:
        lut = np.round((cdf - cdf_min) / denom * 255.0).astype(np.uint8)
        lut[cdf == 0] = 0

    return lut[channel]
