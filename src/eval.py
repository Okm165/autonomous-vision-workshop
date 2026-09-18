"""Evaluation metrics for visual odometry, SLAM, and depth estimation.

This module provides the standard metrics used to evaluate:

* **Trajectory accuracy** — Absolute Trajectory Error (ATE) and Relative
  Pose Error (RPE), with Sim(3) Umeyama alignment.
* **Depth map quality** — the seven standard monocular depth metrics from
  Eigen et al. (2014).
* **Trajectory visualisation** — 2-D (top-down) and 3-D comparison plots.

References
----------
[1] Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems",
    IROS 2012.
[2] Umeyama, "Least-Squares Estimation of Transformation Parameters Between
    Two Point Patterns", TPAMI 1991.
[3] Eigen et al., "Depth Map Prediction from a Single Image using a
    Multi-Scale Deep Network", NeurIPS 2014.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

# ======================================================================
#  Umeyama alignment
# ======================================================================


def _umeyama_rotation(
    H: np.ndarray,
    d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD of cross-covariance and build the rotation matrix."""
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(d, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        D[d - 1, d - 1] = -1.0
    R = U @ D @ Vt
    return R, S, D


def _umeyama_scale(
    S: np.ndarray,
    D: np.ndarray,
    var_src: float,
) -> float:
    """Compute the optimal Umeyama scale from SVD singular values."""
    if var_src < 1e-15:
        return 1.0
    return float(np.trace(np.diag(S) @ D) / var_src)


def align_trajectories_umeyama(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    r"""Umeyama alignment (Sim(3)) of two point sets.

    Find the optimal scale *s*, rotation *R*, and translation
    :math:`\mathbf{t}` minimising:

    .. math::

        \min_{s, R, \mathbf{t}} \;
        \sum_{i=1}^{N}
        \bigl\lVert s \, R \, \mathbf{p}_i + \mathbf{t}
        - \mathbf{q}_i \bigr\rVert^2

    **Algorithm (Umeyama 1991):**

    1. Compute centroids:

       .. math::

           \mu_s = \frac{1}{N}\sum_i \mathbf{p}_i, \qquad
           \mu_t = \frac{1}{N}\sum_i \mathbf{q}_i

    2. Centre the points:

       .. math::

           \tilde{\mathbf{p}}_i = \mathbf{p}_i - \mu_s, \qquad
           \tilde{\mathbf{q}}_i = \mathbf{q}_i - \mu_t

    3. Cross-covariance matrix:

       .. math::

           H = \sum_i \tilde{\mathbf{q}}_i \, \tilde{\mathbf{p}}_i^T

    4. SVD:  :math:`H = U \Sigma V^T`.  Construct
       :math:`D = \text{diag}(1, \ldots, 1, \det(U)\det(V))` to ensure
       :math:`\det(R) = +1`.

    5. Recover optimal parameters:

       .. math::

           R &= U \, D \, V^T \\
           s &= \frac{\text{tr}(\Sigma \, D)}
                     {\sum_i \lVert \tilde{\mathbf{p}}_i \rVert^2} \\
           \mathbf{t} &= \mu_t - s \, R \, \mu_s

    Parameters
    ----------
    source : ndarray, shape (N, 3)
        Source points (estimated trajectory positions).
    target : ndarray, shape (N, 3)
        Target points (ground-truth trajectory positions).

    Returns
    -------
    s : float
        Optimal scale factor.
    R : ndarray, shape (3, 3)
        Optimal rotation matrix.
    t : ndarray, shape (3,)
        Optimal translation vector.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    N, d = source.shape

    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    src_c = source - mu_s
    tgt_c = target - mu_t

    var_src = np.sum(src_c**2) / N

    H = (tgt_c.T @ src_c) / N

    R, S, D = _umeyama_rotation(H, d)

    s = _umeyama_scale(S, D, var_src)

    t = mu_t - s * (R @ mu_s)

    return s, R, t


# ======================================================================
#  Absolute Trajectory Error
# ======================================================================


def compute_ate(
    estimated_poses: Sequence[NDArray[np.float64]],
    gt_poses: Sequence[NDArray[np.float64]],
) -> tuple[float, float, float, NDArray[np.float64], float]:
    r"""Absolute Trajectory Error after Sim(3) alignment.

    **Procedure:**

    1. Extract camera positions from the 4×4 pose matrices.
    2. Align via :func:`align_trajectories_umeyama`.
    3. Compute per-frame position errors:

       .. math::

           e_i = \bigl\lVert \mathbf{q}_i -
           (s \, R \, \mathbf{p}_i + \mathbf{t}) \bigr\rVert

    **Reported statistics:**

    * **ATE RMSE** — root-mean-square of :math:`\{e_i\}`.
    * **ATE mean** — arithmetic mean.
    * **ATE median** — median error.

    Parameters
    ----------
    estimated_poses : sequence of ndarray, each (4, 4)
        Estimated camera-to-world poses.
    gt_poses : sequence of ndarray, each (4, 4)
        Ground-truth camera-to-world poses (same length).

    Returns
    -------
    ate_rmse : float
        RMSE of aligned positional errors.
    ate_mean : float
        Mean of aligned positional errors.
    ate_median : float
        Median of aligned positional errors.
    aligned_poses : ndarray, shape (N, 3)
        Sim(3)-aligned estimated positions.
    scale : float
        Optimal Sim(3) scale factor from Umeyama alignment.
    """
    est_pos = np.array([T[:3, 3] for T in estimated_poses], dtype=np.float64)
    gt_pos = np.array([T[:3, 3] for T in gt_poses], dtype=np.float64)
    assert len(est_pos) == len(gt_pos), "Trajectory lengths must match."

    s, R, t = align_trajectories_umeyama(est_pos, gt_pos)
    aligned = s * (est_pos @ R.T) + t

    errors = np.linalg.norm(gt_pos - aligned, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors**2)))
    ate_mean = float(np.mean(errors))
    ate_median = float(np.median(errors))

    return ate_rmse, ate_mean, ate_median, aligned, s


# ======================================================================
#  Scale Error (monocular systems)
# ======================================================================


def compute_scale_error(scale: float) -> tuple[float, float]:
    r"""Scale error from Umeyama alignment.

    For monocular VO/SLAM the estimated trajectory has unknown global
    scale.  Umeyama alignment recovers the optimal scale :math:`s^*`.
    The scale error quantifies how far :math:`s^*` deviates from unity:

    .. math::

        e_s        &= |1 - s^*| \\
        e_s^{\%}   &= |1 - s^*| \times 100\%

    Parameters
    ----------
    scale : float
        Optimal scale factor from :func:`align_trajectories_umeyama` or
        :func:`compute_ate`.

    Returns
    -------
    scale_error : float
        Absolute scale error :math:`|1 - s^*|`.
    scale_error_pct : float
        Percentage scale error.
    """
    err = abs(1.0 - scale)
    return err, err * 100.0


# ======================================================================
#  Relative Pose Error
# ======================================================================


def compute_rpe(
    estimated_poses: Sequence[NDArray[np.float64]],
    gt_poses: Sequence[NDArray[np.float64]],
    delta: int = 1,
) -> tuple[float, float, float, float]:
    r"""Relative Pose Error over pairs separated by *delta* frames.

    For each pair :math:`(i, i + \delta)`:

    .. math::

        \Delta T_{\text{gt}} &= T_{\text{gt},i}^{-1} \, T_{\text{gt},i+\delta} \\
        \Delta T_{\text{est}} &= T_{\text{est},i}^{-1} \, T_{\text{est},i+\delta} \\
        E_i &= \Delta T_{\text{gt}}^{-1} \, \Delta T_{\text{est}}

    The translational and rotational components of :math:`E_i` are
    extracted:

    * **Translation error**: :math:`\lVert \mathbf{t}(E_i) \rVert`
    * **Rotation error**:
      :math:`\arccos\!\bigl(\frac{\text{tr}(R(E_i)) - 1}{2}\bigr)`

    Parameters
    ----------
    estimated_poses : sequence of ndarray, each (4, 4)
    gt_poses : sequence of ndarray, each (4, 4)
    delta : int
        Frame gap between paired poses.

    Returns
    -------
    rpe_trans_rmse : float
        RMSE of translational RPE.
    rpe_trans_mean : float
        Mean of translational RPE.
    rpe_rot_rmse : float
        RMSE of rotational RPE (radians).
    rpe_rot_mean : float
        Mean of rotational RPE (radians).
    """
    N = len(estimated_poses)
    assert len(gt_poses) == N, "Trajectory lengths must match."

    trans_errors: list[float] = []
    rot_errors: list[float] = []

    for i in range(N - delta):
        T_gt_i = np.asarray(gt_poses[i], dtype=np.float64)
        T_gt_j = np.asarray(gt_poses[i + delta], dtype=np.float64)
        T_est_i = np.asarray(estimated_poses[i], dtype=np.float64)
        T_est_j = np.asarray(estimated_poses[i + delta], dtype=np.float64)

        delta_gt = np.linalg.inv(T_gt_i) @ T_gt_j
        delta_est = np.linalg.inv(T_est_i) @ T_est_j

        E = np.linalg.inv(delta_gt) @ delta_est

        # Translation error
        trans_errors.append(float(np.linalg.norm(E[:3, 3])))

        # Rotation error: geodesic distance on SO(3)
        cos_angle = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rot_errors.append(float(np.arccos(cos_angle)))

    trans_arr = np.array(trans_errors)
    rot_arr = np.array(rot_errors)

    rpe_trans_rmse = float(np.sqrt(np.mean(trans_arr**2)))
    rpe_trans_mean = float(np.mean(trans_arr))
    rpe_rot_rmse = float(np.sqrt(np.mean(rot_arr**2)))
    rpe_rot_mean = float(np.mean(rot_arr))

    return rpe_trans_rmse, rpe_trans_mean, rpe_rot_rmse, rpe_rot_mean


# ======================================================================
#  Depth evaluation metrics
# ======================================================================


def compute_depth_metrics(
    predicted: NDArray[np.floating],
    ground_truth: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    r"""Standard monocular depth evaluation metrics (Eigen et al. 2014).

    All metrics are computed over valid pixels only.

    **Error metrics** (lower is better):

    * ``abs_rel``:  :math:`\frac{1}{N}\sum_i |d_i - d_i^*| / d_i^*`
    * ``sq_rel``:   :math:`\frac{1}{N}\sum_i (d_i - d_i^*)^2 / d_i^*`
    * ``rmse``:     :math:`\sqrt{\frac{1}{N}\sum_i (d_i - d_i^*)^2}`
    * ``rmse_log``: :math:`\sqrt{\frac{1}{N}\sum_i (\log d_i - \log d_i^*)^2}`

    **Accuracy metrics** (higher is better):

    * :math:`\delta_1`:  % of pixels where
      :math:`\max(d / d^*, d^* / d) < 1.25`
    * :math:`\delta_2`:  ``< 1.25²``
    * :math:`\delta_3`:  ``< 1.25³``

    Parameters
    ----------
    predicted : ndarray, shape (H, W)
        Predicted metric depth map.
    ground_truth : ndarray, shape (H, W)
        Ground-truth metric depth map.
    mask : ndarray, shape (H, W), dtype bool, optional
        Valid-pixel mask.  ``None`` → use pixels where ``gt > 0``.

    Returns
    -------
    dict
        Dictionary with keys ``abs_rel``, ``sq_rel``, ``rmse``,
        ``rmse_log``, ``delta_1``, ``delta_2``, ``delta_3``.
    """
    if mask is None:
        mask = ground_truth > 0

    pred = predicted[mask].astype(np.float64)
    gt = ground_truth[mask].astype(np.float64)

    if len(pred) == 0:
        return dict.fromkeys(
            ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta_1", "delta_2", "delta_3"),
            0.0,
        )

    pred = np.clip(pred, 1e-6, None)
    gt = np.clip(gt, 1e-6, None)

    diff = np.abs(pred - gt)

    abs_rel = float(np.mean(diff / gt))
    sq_rel = float(np.mean(diff**2 / gt))
    rmse = float(np.sqrt(np.mean(diff**2)))
    rmse_log = float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2)))

    ratio = np.maximum(pred / gt, gt / pred)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < 1.25**2))
    delta_3 = float(np.mean(ratio < 1.25**3))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
    }


# ======================================================================
#  3-D Reconstruction Metrics
# ======================================================================


def compute_reconstruction_metrics(
    pred_points: NDArray[np.float64],
    gt_points: NDArray[np.float64],
    f_score_threshold: float = 0.05,
) -> dict[str, float]:
    r"""Compute Chamfer distance, accuracy, completeness, and F-score.

    **Chamfer distance** (bidirectional):

    .. math::

        d_{\text{CD}} = \frac{1}{|S_1|} \sum_{x \in S_1} \min_{y \in S_2} \|x-y\|
                      + \frac{1}{|S_2|} \sum_{y \in S_2} \min_{x \in S_1} \|y-x\|

    * First term = **accuracy** (how close reconstruction is to GT)
    * Second term = **completeness** (how much GT is covered)

    **F-score** at threshold :math:`\tau`:

    .. math::

        F = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}

    Parameters
    ----------
    pred_points : (N, 3) predicted point cloud
    gt_points : (M, 3) ground-truth point cloud
    f_score_threshold : distance threshold for F-score

    Returns
    -------
    dict with keys: chamfer, accuracy, completeness, f_score, precision, recall
    """
    from scipy.spatial import KDTree

    tree_pred = KDTree(pred_points)
    tree_gt = KDTree(gt_points)

    d_pred_to_gt, _ = tree_gt.query(pred_points)
    d_gt_to_pred, _ = tree_pred.query(gt_points)

    accuracy = float(np.mean(d_pred_to_gt))
    completeness = float(np.mean(d_gt_to_pred))
    chamfer = accuracy + completeness

    precision = float(np.mean(d_pred_to_gt < f_score_threshold))
    recall = float(np.mean(d_gt_to_pred < f_score_threshold))
    f_score = float(2 * precision * recall / max(precision + recall, 1e-8))

    return {
        "chamfer": chamfer,
        "accuracy": accuracy,
        "completeness": completeness,
        "f_score": f_score,
        "precision": precision,
        "recall": recall,
    }


# ======================================================================
#  Trajectory comparison plot
# ======================================================================


def plot_trajectory_comparison(
    est_poses: Sequence[NDArray[np.float64]],
    gt_poses: Sequence[NDArray[np.float64]],
    title: str = "",
    align: bool = True,
) -> None:
    r"""Plot estimated vs. ground-truth trajectory in 2-D and 3-D.

    Generates a side-by-side figure:

    * **Left panel** — top-down (XZ) view.
    * **Right panel** — full 3-D view.

    If *align* is ``True``, the estimated trajectory is first aligned to
    the ground truth via Umeyama Sim(3) alignment.

    Parameters
    ----------
    est_poses : sequence of ndarray, each (4, 4)
        Estimated camera-to-world poses.
    gt_poses : sequence of ndarray, each (4, 4)
        Ground-truth camera-to-world poses.
    title : str
        Optional super-title for the figure.
    align : bool
        Whether to apply Umeyama alignment before plotting.
    """
    import matplotlib.pyplot as plt

    est_pos = np.array([T[:3, 3] for T in est_poses], dtype=np.float64)
    gt_pos = np.array([T[:3, 3] for T in gt_poses], dtype=np.float64)

    if align and len(est_pos) >= 3:
        s, R, t = align_trajectories_umeyama(est_pos, gt_pos)
        est_pos = s * (est_pos @ R.T) + t
        suffix = " (aligned)"
    else:
        suffix = ""

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title or "Trajectory Comparison", fontsize=14)

    # 2-D top-down (XZ)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(gt_pos[:, 0], gt_pos[:, 2], "k-", label="Ground truth", linewidth=2)
    ax1.plot(
        est_pos[:, 0], est_pos[:, 2], "r--", label=f"Estimated{suffix}", linewidth=1.5
    )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_title("Top-down (XZ)")
    ax1.legend()
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # 3-D
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot3D(
        gt_pos[:, 0],
        gt_pos[:, 1],
        gt_pos[:, 2],
        "k-",
        label="Ground truth",
        linewidth=2,
    )
    ax2.plot3D(
        est_pos[:, 0],
        est_pos[:, 1],
        est_pos[:, 2],
        "r--",
        label=f"Estimated{suffix}",
        linewidth=1.5,
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("3-D View")
    ax2.legend()

    plt.tight_layout()
    plt.show()
