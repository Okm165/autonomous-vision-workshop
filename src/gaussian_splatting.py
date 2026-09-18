"""
3D Gaussian Splatting (3DGS) — Forward Model Implementation

Reference: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
           SIGGRAPH 2023.

The scene is represented as a set of 3D Gaussians, each with:
- μ ∈ R³: center position
- Σ ∈ R^{3×3}: 3D covariance (positive semi-definite, stored as scale + quaternion)
- c ∈ R³: color (can be SH coefficients for view-dependent color)
- α ∈ [0,1]: opacity

Rendering pipeline:
1. Project each 3D Gaussian to 2D (splatting)
2. Sort by depth
3. Alpha-composite front-to-back
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from src.transforms import quaternion_to_matrix
except ImportError:
    try:
        from .transforms import quaternion_to_matrix
    except ImportError:

        def quaternion_to_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
            r"""Fallback: convert unit quaternion [w,x,y,z] to 3×3 rotation matrix.

            R = \begin{pmatrix}
                1 - 2(y²+z²) & 2(xy - wz)   & 2(xz + wy)   \\
                2(xy + wz)   & 1 - 2(x²+z²) & 2(yz - wx)   \\
                2(xz - wy)   & 2(yz + wx)   & 1 - 2(x²+y²)
            \end{pmatrix}
            """
            q = np.asarray(q, dtype=np.float64).ravel()
            q = q / (np.linalg.norm(q) + 1e-12)
            w, x, y, z = q
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )


_TRANSMITTANCE_EPS = 1e-4


@dataclass
class Gaussian3D:
    """
    A single 3D Gaussian primitive.

    The 3D Gaussian is defined as:
        G(x) = exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))

    Covariance parameterization:
        Σ = R · S · Sᵀ · Rᵀ
    where R = quaternion_to_matrix(q) and S = diag(s₁, s₂, s₃).
    This ensures Σ is always positive semi-definite.
    """

    position: np.ndarray  # (3,) center μ
    scale: np.ndarray  # (3,) log-scale factors
    quaternion: np.ndarray  # (4,) rotation quaternion [w,x,y,z]
    color: np.ndarray  # (3,) RGB color
    opacity: float  # α ∈ [0,1]

    @property
    def scales(self) -> np.ndarray:
        """Actual scales (exp of log-scales for positive guarantee)."""
        return np.exp(self.scale)

    @property
    def covariance_3d(self) -> np.ndarray:
        """
        Compute 3×3 covariance matrix.
        Σ = R · diag(s²) · Rᵀ
        """
        return build_covariance_3d(self.scale, self.quaternion)


def build_covariance_3d(
    scale: NDArray[np.float64], quaternion: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""
    Build 3D covariance from scale and rotation.

    Σ = R · S · Sᵀ · Rᵀ
    where S = diag(exp(s₁), exp(s₂), exp(s₃))

    Expanding:
        Σ = R · diag(exp(2s₁), exp(2s₂), exp(2s₃)) · Rᵀ

    This parameterization guarantees Σ is positive semi-definite because
    for any vector v:
        vᵀ Σ v = vᵀ R diag(exp(2s)) Rᵀ v = ‖diag(exp(s)) Rᵀ v‖² ≥ 0

    Parameters
    ----------
    scale : ndarray, shape (3,)
        Log-scale factors [s₁, s₂, s₃].
    quaternion : ndarray, shape (4,)
        Rotation quaternion [w, x, y, z].

    Returns
    -------
    Sigma : ndarray, shape (3, 3)
        Positive semi-definite covariance matrix.
    """
    s = np.exp(np.asarray(scale, dtype=np.float64))
    S = np.diag(s)
    R = quaternion_to_matrix(quaternion)
    M = R @ S
    return M @ M.T


def project_gaussian_to_2d(
    gaussian: Gaussian3D,
    T_world_to_cam: NDArray[np.float64],
    K: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    r"""
    Project a 3D Gaussian to a 2D Gaussian on the image plane.

    The projection of a 3D Gaussian through a pinhole camera:

    1. Transform to camera frame:
       μ_cam = R_cam @ μ + t_cam

    2. Compute the Jacobian of the projective transformation:
       J = ∂π/∂p = [[fx/z, 0, -fx·x/z²],
                     [0, fy/z, -fy·y/z²]]
       (2×3 matrix, evaluated at μ_cam)

    3. Project the covariance to 2D:
       The "EWA splatting" formula (Zwicker et al. 2001):
       Σ_cam = R_cam @ Σ_world @ R_camᵀ
       Σ_2d = J @ Σ_cam @ Jᵀ  (2×2 matrix)

       A small isotropic term (0.3·I) is added for anti-aliasing
       (low-pass filter from Zwicker et al.).

    4. Project center to pixel:
       p_hom = K @ μ_cam
       μ_2d = p_hom[:2] / p_hom[2]

    Parameters
    ----------
    gaussian : Gaussian3D
        The 3D Gaussian to project.
    T_world_to_cam : ndarray, shape (4, 4)
        Camera extrinsics [R | t] mapping world → camera.
    K : ndarray, shape (3, 3)
        Camera intrinsics [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].

    Returns
    -------
    mu_2d : ndarray, shape (2,)
        Projected pixel center.
    cov_2d : ndarray, shape (2, 2)
        2D covariance in pixel space.
    depth : float
        z-depth in camera frame for depth sorting.
    """
    R_cam = T_world_to_cam[:3, :3]
    t_cam = T_world_to_cam[:3, 3]

    mu_cam = R_cam @ gaussian.position + t_cam
    depth = float(mu_cam[2])

    if depth < 0.1:
        return np.array([0.0, 0.0]), np.eye(2), depth

    fx, fy = K[0, 0], K[1, 1]
    x, y, z = mu_cam

    J = np.array(
        [
            [fx / z, 0.0, -fx * x / (z * z)],
            [0.0, fy / z, -fy * y / (z * z)],
        ]
    )

    Sigma_world = gaussian.covariance_3d
    Sigma_cam = R_cam @ Sigma_world @ R_cam.T

    cov_2d = J @ Sigma_cam @ J.T

    cov_2d[0, 0] += 0.3
    cov_2d[1, 1] += 0.3

    p_hom = K @ mu_cam
    mu_2d = p_hom[:2] / p_hom[2]

    return mu_2d, cov_2d, depth


def gaussian_2d_pdf(
    x: float | NDArray[np.float64],
    y: float | NDArray[np.float64],
    mu: NDArray[np.float64],
    cov: NDArray[np.float64],
) -> NDArray[np.float64] | np.float64:
    r"""
    Evaluate 2D Gaussian PDF on a grid.

    G(x,y) = (1 / (2π |Σ|^{1/2})) exp(-½ [x-μx, y-μy] Σ⁻¹ [x-μx, y-μy]ᵀ)

    For rendering we only need the unnormalized exponent (the normalization
    cancels with the opacity). We return the *unnormalized* Gaussian:

    G(x,y) = exp(-½ dᵀ Σ⁻¹ d)   where d = [x-μx, y-μy]ᵀ

    Using the analytic inverse for a 2×2 symmetric matrix:

    Σ = [[a, b], [b, c]]
    Σ⁻¹ = (1/det) [[c, -b], [-b, a]]
    det = ac - b²

    Parameters
    ----------
    x, y : float | ndarray
        Coordinate grids (same shape, e.g. from np.meshgrid); scalars
        evaluate the PDF at a single point.
    mu : ndarray, shape (2,)
        Mean [μx, μy].
    cov : ndarray, shape (2, 2)
        2×2 covariance matrix.

    Returns
    -------
    G : ndarray (same shape as x, y)
        Unnormalized Gaussian values in [0, 1].
    """
    a, b = cov[0, 0], cov[0, 1]
    c = cov[1, 1]
    det = a * c - b * b

    if det < 1e-12:
        return np.zeros_like(x)

    inv_det = 1.0 / det
    dx = x - mu[0]
    dy = y - mu[1]

    exponent = -0.5 * inv_det * (c * dx * dx - 2.0 * b * dx * dy + a * dy * dy)

    return np.exp(np.clip(exponent, -50.0, 0.0))


def render_gaussians(
    gaussians: list[Gaussian3D],
    T_world_to_cam: NDArray[np.float64],
    K: NDArray[np.float64],
    image_size: tuple[int, int],
) -> NDArray[np.float64]:
    r"""
    Render a set of 3D Gaussians to an image.

    Algorithm:
    1. Project all Gaussians to 2D
    2. Sort by depth (front-to-back)
    3. For each pixel, alpha-composite overlapping Gaussians:

       Alpha compositing (front-to-back):
           C_pixel = Σ_i  cᵢ · αᵢ · Gᵢ(pixel) · Πⱼ<ᵢ (1 - αⱼ · Gⱼ(pixel))

       where Gᵢ(pixel) = exp(-½ (pixel - μᵢ)ᵀ Σᵢ⁻¹ (pixel - μᵢ))

       Equivalently, the accumulated transmittance:
           Tᵢ = Πⱼ<ᵢ (1 - αⱼ · Gⱼ(pixel))
           C = Σᵢ cᵢ · αᵢ · Gᵢ · Tᵢ

       Stop when Tᵢ < ε (remaining contribution negligible).

    This is differentiable! The gradient ∂C/∂μᵢ, ∂C/∂Σᵢ, ∂C/∂cᵢ, ∂C/∂αᵢ
    can all be computed analytically, enabling gradient-based optimization.

    Parameters
    ----------
    gaussians : list of Gaussian3D
    T_world_to_cam : ndarray, shape (4, 4)
        Camera extrinsics.
    K : ndarray, shape (3, 3)
        Camera intrinsics.
    image_size : tuple (H, W)

    Returns
    -------
    image : ndarray, shape (H, W, 3)
        Rendered RGB image with values in [0, 1].
    """
    H, W = image_size
    image = np.zeros((H, W, 3), dtype=np.float64)
    transmittance = np.ones((H, W), dtype=np.float64)

    projected: list[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            float,
            NDArray[np.floating],
            float,
        ]
    ] = []
    for g in gaussians:
        mu_2d, cov_2d, depth = project_gaussian_to_2d(g, T_world_to_cam, K)
        if depth < 0.1:
            continue
        projected.append((mu_2d, cov_2d, depth, g.color, g.opacity))

    projected.sort(key=lambda t: t[2])

    yy, xx = np.mgrid[:H, :W]
    xx = xx.astype(np.float64)
    yy = yy.astype(np.float64)

    for mu_2d, cov_2d, _depth, color, opacity in projected:
        eigenvalues = np.linalg.eigvalsh(cov_2d)
        max_eigenval = max(eigenvalues.max(), 1e-6)
        radius = 3.0 * np.sqrt(max_eigenval)

        x_min = max(int(np.floor(mu_2d[0] - radius)), 0)
        x_max = min(int(np.ceil(mu_2d[0] + radius)), W)
        y_min = max(int(np.floor(mu_2d[1] - radius)), 0)
        y_max = min(int(np.ceil(mu_2d[1] + radius)), H)

        if x_min >= x_max or y_min >= y_max:
            continue

        patch_x = xx[y_min:y_max, x_min:x_max]
        patch_y = yy[y_min:y_max, x_min:x_max]

        G = gaussian_2d_pdf(patch_x, patch_y, mu_2d, cov_2d)

        alpha_map = opacity * G

        patch_T = transmittance[y_min:y_max, x_min:x_max]

        weight = alpha_map * patch_T

        for ch in range(3):
            image[y_min:y_max, x_min:x_max, ch] += weight * color[ch]

        transmittance[y_min:y_max, x_min:x_max] *= 1.0 - alpha_map

        if transmittance.max() < _TRANSMITTANCE_EPS:
            break

    return np.clip(image, 0.0, 1.0)


def compute_loss(
    rendered: NDArray[np.float64], target: NDArray[np.float64]
) -> dict[str, float]:
    r"""
    Compute training loss for 3DGS optimization.

    L = (1-λ) · L₁ + λ · L_SSIM

    where:
    - L₁ = mean(|rendered - target|)  (photometric L1 loss)
    - L_SSIM = 1 - SSIM(rendered, target)  (structural similarity)
    - λ = 0.2 (default weight from the paper)

    SSIM computation (per-channel, then averaged):
    SSIM(x, y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))

    where:
    - μₓ, μᵧ are local means computed via a Gaussian-weighted window
    - σₓ², σᵧ² are local variances
    - σₓᵧ is the local cross-covariance
    - C₁ = (k₁ · L)², C₂ = (k₂ · L)² with k₁=0.01, k₂=0.03, L=1.0

    Parameters
    ----------
    rendered : ndarray, shape (H, W, 3)
    target : ndarray, shape (H, W, 3)

    Returns
    -------
    dict with 'total', 'l1', 'ssim' losses.
    """
    l1_loss = float(np.mean(np.abs(rendered - target)))
    ssim_val = _compute_ssim(rendered, target)
    ssim_loss = 1.0 - ssim_val

    lam = 0.2
    total = (1.0 - lam) * l1_loss + lam * ssim_loss

    return {"total": total, "l1": l1_loss, "ssim": ssim_loss}


def _gaussian_kernel_1d(size: int, sigma: float) -> NDArray[np.float64]:
    """1D Gaussian kernel for SSIM local statistics."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def _apply_gaussian_filter(
    img: NDArray[np.float64], kernel_size: int = 11, sigma: float = 1.5
) -> NDArray[np.float64]:
    """Apply separable Gaussian filter (same-mode convolution)."""
    k1d = _gaussian_kernel_1d(kernel_size, sigma)

    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    H, W, C = img.shape
    out = np.zeros_like(img)

    for c in range(C):
        tmp = np.zeros((H, W), dtype=np.float64)
        for i in range(H):
            tmp[i, :] = np.convolve(img[i, :, c], k1d, mode="same")
        for j in range(W):
            out[:, j, c] = np.convolve(tmp[:, j], k1d, mode="same")

    return out


def _compute_ssim(
    img1: NDArray[np.float64],
    img2: NDArray[np.float64],
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> float:
    r"""
    Structural Similarity Index (SSIM).

    SSIM(x, y) = (2μₓμᵧ + C₁)(2σₓᵧ + C₂) / ((μₓ² + μᵧ² + C₁)(σₓ² + σᵧ² + C₂))

    Constants (Wang et al. 2004):
        C₁ = (0.01)² = 0.0001
        C₂ = (0.03)² = 0.0009

    Returns the mean SSIM over all pixels and channels.
    """
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = _apply_gaussian_filter(img1, kernel_size, sigma)
    mu2 = _apply_gaussian_filter(img2, kernel_size, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _apply_gaussian_filter(img1 * img1, kernel_size, sigma) - mu1_sq
    sigma2_sq = _apply_gaussian_filter(img2 * img2, kernel_size, sigma) - mu2_sq
    sigma12 = _apply_gaussian_filter(img1 * img2, kernel_size, sigma) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-12)

    return float(np.mean(ssim_map))


def create_random_gaussians(
    n: int,
    bounds: tuple[float, float] = (-1.0, 1.0),
    rng: np.random.Generator | None = None,
) -> list[Gaussian3D]:
    """
    Create N random Gaussians within given spatial bounds for testing.

    Parameters
    ----------
    n : int
        Number of Gaussians.
    bounds : tuple (lo, hi)
        Spatial extent for positions.
    rng : np.random.Generator, optional
        Random generator for reproducibility. ``None`` → global RNG state.

    Returns
    -------
    list of Gaussian3D
    """
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = bounds
    gaussians: list[Gaussian3D] = []
    for _ in range(n):
        pos = rng.uniform(lo, hi, size=3)
        scale = rng.uniform(-2.0, 0.0, size=3)

        q = rng.normal(size=4)
        q = q / (np.linalg.norm(q) + 1e-12)

        color = rng.uniform(0.0, 1.0, size=3)
        opacity = float(rng.uniform(0.3, 1.0))

        gaussians.append(
            Gaussian3D(
                position=pos,
                scale=scale,
                quaternion=q,
                color=color,
                opacity=opacity,
            )
        )
    return gaussians


def volume_render_equation(
    sigmas: NDArray[np.float64],
    colors: NDArray[np.float64],
    deltas: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""
    Volume rendering equation (Max 1995, Mildenhall et al. 2020).

    C = Σᵢ Tᵢ · (1 - exp(-σᵢδᵢ)) · cᵢ

    where:
    - Tᵢ = exp(-Σⱼ<ᵢ σⱼδⱼ) is the accumulated transmittance
    - σᵢ is the volume density at sample i
    - δᵢ is the distance between samples i and i+1
    - cᵢ is the color at sample i

    The connection to 3DGS:
    - NeRF samples points along rays and integrates
    - 3DGS projects Gaussians to 2D and alpha-composites
    - Both are discretizations of the same continuous rendering integral

    Specifically, the alpha value αᵢ in 3DGS compositing corresponds to
    (1 - exp(-σᵢδᵢ)) in the volume rendering equation. Both formulations
    accumulate transmittance T front-to-back and weight each sample's
    color by αᵢ · Tᵢ.

    Parameters
    ----------
    sigmas : ndarray, shape (N,)
        Volume densities σᵢ ≥ 0 at each sample.
    colors : ndarray, shape (N, 3)
        RGB colors cᵢ at each sample.
    deltas : ndarray, shape (N,)
        Step sizes δᵢ between consecutive samples.

    Returns
    -------
    C : ndarray, shape (3,)
        Composited RGB color for the ray.
    """
    sigmas = np.asarray(sigmas, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)

    N = len(sigmas)
    assert colors.shape == (N, 3), f"colors must be (N, 3), got {colors.shape}"
    assert deltas.shape == (N,), f"deltas must be (N,), got {deltas.shape}"

    alpha = 1.0 - np.exp(-sigmas * deltas)

    tau = np.concatenate([[0.0], np.cumsum(sigmas[:-1] * deltas[:-1])])
    T = np.exp(-tau)

    weights = alpha * T

    C = np.sum(weights[:, np.newaxis] * colors, axis=0)

    return C
