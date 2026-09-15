"""Visual SLAM components: pose-graph optimisation, bundle adjustment, and loop closure.

This module provides fully functional, pure-NumPy / SciPy implementations of
the core algorithmic building blocks of a visual SLAM system:

* **PoseGraph** -- pose-graph optimisation via Gauss-Newton on the SE(3)
  manifold.
* **BundleAdjuster** -- joint refinement of camera poses and 3-D landmarks
  with the Schur complement trick for efficient solving.
* **LoopDetector** -- bag-of-visual-words loop-closure detection.
* **Robust kernels** -- Huber and Cauchy (Lorentzian) loss functions for
  outlier-resistant estimation.

Coordinate / Lie-group conventions
-----------------------------------
* Camera poses are 4x4 homogeneous SE(3) matrices (**camera-to-world**),
  consistent with :mod:`src.odometry`.
* Twist vectors follow the ``[rho; omega]`` ordering (translation first,
  rotation second), matching :mod:`src.transforms` (Barfoot convention).
* Optimisation uses **right perturbation**:  ``T <- T . Exp(d_xi)``.

References
----------
[1] Kuemmerle et al., "g2o: A General Framework for Graph Optimization",
    ICRA 2011.
[2] Triggs et al., "Bundle Adjustment -- A Modern Synthesis", 2000.
[3] Grisetti et al., "A Tutorial on Graph-Based SLAM", IEEE Intelligent
    Transportation Systems Magazine, 2010.
[4] Galvez-Lopez & Tardos, "Bags of Binary Words for Fast Place
    Recognition in Image Sequences", IEEE T-RO, 2012.
[5] Barfoot, *State Estimation for Robotics*, Cambridge Univ. Press, 2017.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.cluster.vq import kmeans2, vq
from scipy.sparse.linalg import spsolve

from src.transforms import (
    se3_adjoint,
    se3_compose,
    se3_exp,
    se3_inverse,
    se3_log,
    se3_to_Rt,
    skew,
)

_EPS: float = 1e-10


# ===================================================================
#  Helpers: SE(3) Lie-algebra adjoint and right-Jacobian inverse
# ===================================================================


def _se3_ad_matrix(xi: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""6x6 adjoint representation of an se(3) Lie-algebra element.

    For xi = [rho; omega] in R^6 the *small* adjoint (lowercase "ad") is

    .. math::

        \mathrm{ad}(\xi) =
        \begin{pmatrix}
            [\omega]_\times & [\rho]_\times  \\
            0_{3 \times 3}  & [\omega]_\times
        \end{pmatrix}

    This matrix appears in the Baker-Campbell-Hausdorff (BCH) formula and
    its truncations.

    Parameters
    ----------
    xi : array_like, shape (6,)
        Twist vector [rho_1, rho_2, rho_3, omega_1, omega_2, omega_3].

    Returns
    -------
    ad : ndarray, shape (6, 6)
    """
    rho, omega = xi[:3], xi[3:]
    ad = np.zeros((6, 6))
    ad[:3, :3] = skew(omega)
    ad[:3, 3:] = skew(rho)
    ad[3:, 3:] = skew(omega)
    return ad


def _se3_right_jacobian_inv(xi: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Approximate inverse right Jacobian of SE(3) via the BCH series.

    The inverse *left* Jacobian admits the Bernoulli-number series

    .. math::

        \mathcal{J}_l^{-1}(\xi) =
        \sum_{k=0}^{\infty} \frac{B_k}{k!}\,\mathrm{ad}(\xi)^k

    with standard Bernoulli numbers B_0=1, B_1=-1/2, B_2=1/6, ...
    The *right* Jacobian inverse is obtained via
    :math:`\mathcal{J}_r^{-1}(\xi) = \mathcal{J}_l^{-1}(-\xi)`, which
    flips the sign of odd-order terms.  Truncating at second order gives:

    .. math::

        \mathcal{J}_r^{-1}(\xi) \approx
        I_6 + \tfrac{1}{2}\,\mathrm{ad}(\xi)
            + \tfrac{1}{12}\,\mathrm{ad}(\xi)^2

    This approximation is accurate to O(||xi||^3) and is the standard
    choice in iterative graph-SLAM solvers where the error xi is driven
    toward zero during optimisation.

    Parameters
    ----------
    xi : array_like, shape (6,)

    Returns
    -------
    Jr_inv : ndarray, shape (6, 6)
    """
    xi = np.asarray(xi, dtype=np.float64).ravel()
    ad = _se3_ad_matrix(xi)
    return np.eye(6) + 0.5 * ad + (1.0 / 12.0) * (ad @ ad)


# ===================================================================
#  Robust kernels
# ===================================================================


def huber_kernel(
    residual: float, delta: float = 1.0
) -> Tuple[float, float]:
    r"""Huber robust cost function.

    .. math::

        \rho(r) =
        \begin{cases}
            \tfrac{1}{2}\,r^2                            & |r| \le \delta \\
            \delta\,\bigl(|r| - \tfrac{\delta}{2}\bigr)  & |r| > \delta
        \end{cases}

    The equivalent IRLS weight (for converting to iteratively re-weighted
    least squares) is w(r) = rho'(r) / r:

    .. math::

        w(r) =
        \begin{cases}
            1              & |r| \le \delta \\
            \delta / |r|   & |r| > \delta
        \end{cases}

    The Huber kernel is **convex** and transitions smoothly from an L2
    (quadratic) regime near the origin to L1 (linear) growth in the tails,
    bounding the influence of large outliers while preserving the
    efficiency of least-squares for inliers.

    Parameters
    ----------
    residual : float
        Scalar residual r.
    delta : float
        Threshold separating the quadratic and linear regions.

    Returns
    -------
    cost : float
        rho(r).
    weight : float
        w(r) = rho'(r) / r, for use in IRLS.
    """
    abs_r = abs(residual)
    if abs_r <= delta:
        return 0.5 * residual * residual, 1.0
    return delta * (abs_r - 0.5 * delta), delta / abs_r


def cauchy_kernel(
    residual: float, c: float = 1.0
) -> Tuple[float, float]:
    r"""Cauchy (Lorentzian) robust cost function.

    .. math::

        \rho(r) = \frac{c^2}{2}\,\ln\!\bigl(1 + (r/c)^2\bigr)

    Equivalent IRLS weight:

    .. math::

        w(r) = \frac{1}{1 + (r/c)^2}

    The Cauchy kernel is **non-convex** and more aggressive than Huber at
    suppressing outliers.  Its influence function is *redescending*:
    extreme residuals contribute almost zero gradient, effectively
    ignoring gross outliers.

    Parameters
    ----------
    residual : float
        Scalar residual r.
    c : float
        Scale parameter controlling the width of the inlier region.

    Returns
    -------
    cost : float
        rho(r).
    weight : float
        w(r) = rho'(r) / r.
    """
    r_over_c_sq = (residual / c) ** 2
    cost = 0.5 * c * c * np.log1p(r_over_c_sq)
    weight = 1.0 / (1.0 + r_over_c_sq)
    return cost, weight


def geman_mcclure_kernel(
    residual: float, c: float = 1.0,
) -> tuple[float, float]:
    r"""Geman–McClure robust cost.

    .. math::

        \rho(r) = \frac{r^2 / 2}{c^2 + r^2}

    The IRLS weight:

    .. math::

        w(r) = \frac{c^2}{(c^2 + r^2)^2}

    This kernel has a *redescending* influence function — residuals much
    larger than *c* are almost completely suppressed, making it very
    aggressive against outliers.

    Parameters
    ----------
    residual : scalar residual value
    c : scale parameter

    Returns
    -------
    cost : robust cost ρ(r)
    weight : IRLS weight w(r)
    """
    r2 = residual ** 2
    c2 = c ** 2
    cost = 0.5 * r2 / (c2 + r2)
    # IRLS weight: w(r) = ρ'(r)/r = c² / (c² + r²)²
    weight = c2 / (c2 + r2) ** 2
    return cost, weight


# ===================================================================
#  Pose Graph
# ===================================================================


class PoseGraph:
    r"""Pose graph for SLAM optimisation.

    Nodes: camera poses T_i in SE(3) (camera-to-world).

    Edges: relative-pose constraints T_ij with information matrix
    Omega_ij = Sigma_ij^{-1}.

    The pose-graph optimisation problem seeks the node poses that best
    satisfy all measured constraints:

    .. math::

        \min_{\{T_i\}} \sum_{(i,j) \in \mathcal{E}}
        \bigl\lVert
            \operatorname{Log}\!\bigl(T_{ij}^{-1}\,T_i^{-1}\,T_j\bigr)
        \bigr\rVert_{\Omega_{ij}}^{2}

    This is a nonlinear least-squares problem on the SE(3) manifold,
    solved via Gauss-Newton with sparse linear algebra.

    Error and Jacobians (right perturbation)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The 6-vector error for edge (i, j) is

    .. math::

        e_{ij}
        = \operatorname{Log}\!\bigl(T_{ij}^{-1}\,T_i^{-1}\,T_j\bigr)
        \in \mathbb{R}^6

    Under right perturbation T_k <- T_k . Exp(d_xi_k), the linearised
    Jacobians are (via first-order BCH):

    .. math::

        \frac{\partial e}{\partial \delta\xi_i}
        = -\mathcal{J}_r^{-1}(e_{ij})\;
           \operatorname{Ad}(T_j^{-1}\,T_i)

    .. math::

        \frac{\partial e}{\partial \delta\xi_j}
        = \mathcal{J}_r^{-1}(e_{ij})

    **Derivation sketch** for d_e / d_xi_i:

    Perturb T_i -> T_i . Exp(d_xi_i), keeping T_j fixed:

    e(d_xi_i) = Log(T_ij^{-1} . Exp(-d_xi_i) . T_i^{-1} . T_j)

    Move Exp(-d_xi_i) to the right of T_i^{-1} T_j using
    Exp(xi) . A = A . Exp(Ad(A^{-1}).xi):

    = Log(T_ij^{-1} . (T_i^{-1} T_j) . Exp(-Ad(T_j^{-1} T_i) . d_xi_i))
    = Log(Delta . Exp(-Ad(T_j^{-1} T_i) . d_xi_i))

    where Delta = T_ij^{-1} T_i^{-1} T_j,  e_0 = Log(Delta).

    Apply the BCH approximation Log(Exp(a).Exp(b)) ~ a + J_r^{-1}(a).b:

    ~ e_0 + J_r^{-1}(e_0).(-Ad(T_j^{-1} T_i) . d_xi_i)

    Gauss-Newton iteration
    ~~~~~~~~~~~~~~~~~~~~~~
    1. For every edge, compute e_ij and its Jacobians.
    2. Assemble the sparse normal equation H.dx = -b.
    3. Solve for dx (sparse Cholesky / LU).
    4. Update: T_k <- T_k . Exp(dx_k).
    5. Repeat until ||dx|| < threshold.
    """

    def __init__(self) -> None:
        """Initialise an empty pose graph with no nodes or edges."""
        self.poses: List[NDArray[np.float64]] = []
        self.edges: List[
            Tuple[int, int, NDArray[np.float64], NDArray[np.float64]]
        ] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, pose: NDArray[np.float64]) -> int:
        """Add a pose node to the graph.

        Parameters
        ----------
        pose : ndarray, shape (4, 4)
            Camera-to-world SE(3) transformation.

        Returns
        -------
        idx : int
            Index of the newly added node.
        """
        pose = np.asarray(pose, dtype=np.float64)
        self.poses.append(pose.copy())
        return len(self.poses) - 1

    def add_loop_closure(
        self,
        i: int,
        j: int,
        relative_pose: NDArray[np.float64],
        information: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Add a loop closure edge to the pose graph."""
        if information is None:
            information = np.eye(6) * 100
        self.add_edge(i, j, relative_pose, information)

    def add_edge(
        self,
        i: int,
        j: int,
        T_ij: NDArray[np.float64],
        information: Optional[NDArray[np.float64]] = None,
    ) -> None:
        r"""Add a relative-pose constraint between nodes i and j.

        Parameters
        ----------
        i, j : int
            Source and target node indices.
        T_ij : ndarray, shape (4, 4)
            Measured relative transformation from frame i to frame j,
            such that T_j ~ T_i . T_ij.
        information : ndarray, shape (6, 6), optional
            Information matrix Omega_ij = Sigma_ij^{-1}.
            Defaults to I_6.
        """
        T_ij = np.asarray(T_ij, dtype=np.float64)
        if information is None:
            information = np.eye(6)
        else:
            information = np.asarray(information, dtype=np.float64)
        self.edges.append((i, j, T_ij.copy(), information.copy()))

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _compute_error(
        self, i: int, j: int, T_ij: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""Compute the 6-vector error for edge (i, j).

        .. math::

            e_{ij} = \operatorname{Log}(T_{ij}^{-1}\,T_i^{-1}\,T_j)
        """
        T_ij_inv = se3_inverse(T_ij)
        T_i_inv = se3_inverse(self.poses[i])
        return se3_log(
            se3_compose(T_ij_inv, se3_compose(T_i_inv, self.poses[j]))
        )

    def optimize(
        self,
        n_iterations: int = 10,
        fixed_nodes: Optional[List[int]] = None,
    ) -> List[NDArray[np.float64]]:
        r"""Optimise the pose graph via Gauss-Newton on SE(3).

        Algorithm
        ---------
        For each iteration:

        1. For every edge, compute the error e_ij and the 6x6 Jacobians
           J_i, J_j.
        2. Assemble the sparse normal equation H.dx = -b with

           H = sum J_ij^T . Omega_ij . J_ij
           b = sum J_ij^T . Omega_ij . e_ij

        3. Solve for dx via sparse Cholesky / LU.
        4. Update each pose: T_k <- T_k . Exp(dx_k).
        5. Repeat until ||dx|| < 1e-8.

        Parameters
        ----------
        n_iterations : int
            Maximum Gauss-Newton iterations.
        fixed_nodes : list of int, optional
            Node indices held fixed (gauge freedom).  Defaults to [0].

        Returns
        -------
        poses : list of ndarray, each (4, 4)
            Optimised camera-to-world poses.
        """
        if fixed_nodes is None:
            fixed_nodes = [0]

        n = len(self.poses)
        dim = 6 * n
        fixed_set = set(fixed_nodes)

        for _ in range(n_iterations):
            H = np.zeros((dim, dim))
            b = np.zeros(dim)

            for i, j, T_ij, info in self.edges:
                e_ij = self._compute_error(i, j, T_ij)
                Jr_inv = _se3_right_jacobian_inv(e_ij)

                Ad_ji = se3_adjoint(
                    se3_compose(se3_inverse(self.poses[j]), self.poses[i])
                )

                Ji = -(Jr_inv @ Ad_ji)  # (6, 6)
                Jj = Jr_inv             # (6, 6)

                si, sj = 6 * i, 6 * j

                H[si:si + 6, si:si + 6] += Ji.T @ info @ Ji
                H[si:si + 6, sj:sj + 6] += Ji.T @ info @ Jj
                H[sj:sj + 6, si:si + 6] += Jj.T @ info @ Ji
                H[sj:sj + 6, sj:sj + 6] += Jj.T @ info @ Jj

                info_e = info @ e_ij
                b[si:si + 6] += Ji.T @ info_e
                b[sj:sj + 6] += Jj.T @ info_e

            for idx in fixed_set:
                s = 6 * idx
                H[s:s + 6, :] = 0.0
                H[:, s:s + 6] = 0.0
                H[s:s + 6, s:s + 6] = np.eye(6)
                b[s:s + 6] = 0.0

            H_sp = sparse.csc_matrix(H)
            dx = spsolve(H_sp, -b)

            for k in range(n):
                if k not in fixed_set:
                    self.poses[k] = se3_compose(
                        self.poses[k], se3_exp(dx[6 * k : 6 * k + 6])
                    )

            if np.linalg.norm(dx) < 1e-8:
                break

        return [p.copy() for p in self.poses]


# ===================================================================
#  Bundle Adjuster
# ===================================================================


class BundleAdjuster:
    r"""Bundle Adjustment -- joint optimisation of camera poses and 3-D points.

    The name *bundle adjustment* refers to adjusting the bundles of light
    rays connecting each camera centre to the 3-D landmarks it observes.

    Cost function
    -------------
    .. math::

        \min_{\{T_i,\,X_j\}}
        \sum_{(i,j)}
        \rho\!\Bigl(
            \bigl\lVert
                \pi(T_i,\,X_j) - z_{ij}
            \bigr\rVert_{\Omega_{ij}}^2
        \Bigr)

    where:

    * T_i -- camera-to-world pose of camera i.
    * X_j -- 3-D landmark j in world coordinates.
    * z_ij -- observed 2-D projection of landmark j in camera i.
    * pi -- perspective projection function.
    * rho -- optional robust kernel (Huber, Cauchy, ...) for outlier
      robustness (identity when None).

    The Schur Complement Trick
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    The Hessian of the Gauss-Newton system has a characteristic block
    structure arising from the bipartite camera-point graph:

    .. math::

        H = \begin{pmatrix}
            H_{cc} & H_{cp} \\
            H_{pc} & H_{pp}
        \end{pmatrix}

    where H_pp is **block-diagonal** -- each 3-D point only interacts
    with itself in the point-point block.  The Schur complement

    .. math::

        S = H_{cc} - H_{cp}\,H_{pp}^{-1}\,H_{pc}

    is formed cheaply because H_pp^{-1} factorises into independent
    3x3 inversions.  The reduced system

    .. math::

        S\,\Delta x_c = -(b_c - H_{cp}\,H_{pp}^{-1}\,b_p)

    has dimension 6m (with m cameras) instead of 6m + 3n.  After
    solving for camera updates, point updates are recovered via
    back-substitution:

    .. math::

        \Delta x_p = H_{pp}^{-1}\,(-b_p - H_{pc}\,\Delta x_c)

    This reduces complexity from O((6m+3n)^3) to O((6m)^3 + n), since
    typically m << n.

    Projection Jacobians
    ~~~~~~~~~~~~~~~~~~~~
    With P = R.X + t (camera-frame point, computed from the
    **world-to-camera** transform), the projected pixel is

    .. math::

        u = \pi(P) = (f_x P_x / P_z + c_x,\; f_y P_y / P_z + c_y)

    The 2x3 Jacobian of pi w.r.t. the camera-frame point is

    .. math::

        \frac{\partial\pi}{\partial P}
        = \begin{pmatrix}
            f_x/Z & 0     & -f_x X_c/Z^2 \\
            0     & f_y/Z & -f_y Y_c/Z^2
          \end{pmatrix}

    Under **right perturbation** of the *camera-to-world* pose
    T <- T . Exp(d_xi), the camera-frame point changes as:

    .. math::

        \frac{\partial P}{\partial \delta\xi}
        = [-I_3 \;|\; [P]_\times]
        \in \mathbb{R}^{3 \times 6}

    so the camera Jacobian is

    .. math::

        J_{cam} = \frac{\partial\pi}{\partial P}\;[-I_3 \;|\; [P]_\times]

    and the point Jacobian is

    .. math::

        J_{point} = \frac{\partial\pi}{\partial P}\;R
    """

    def __init__(self) -> None:
        """Initialise an empty bundle adjustment problem."""
        self.cameras: List[NDArray[np.float64]] = []
        self.points: List[NDArray[np.float64]] = []
        self.observations: List[Tuple[int, int, float, float]] = []
        self.K: Optional[NDArray[np.float64]] = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_camera(
        self,
        pose: NDArray[np.float64],
        K: NDArray[np.float64],
    ) -> int:
        """Add a camera with its pose and intrinsic matrix.

        Parameters
        ----------
        pose : ndarray, shape (4, 4)
            Camera-to-world SE(3) transformation.
        K : ndarray, shape (3, 3)
            Camera intrinsic (calibration) matrix.

        Returns
        -------
        idx : int
            Index of the newly added camera.
        """
        pose = np.asarray(pose, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        if self.K is None:
            self.K = K.copy()
        self.cameras.append(pose.copy())
        return len(self.cameras) - 1

    def add_point(self, point_3d: NDArray[np.float64]) -> int:
        """Add a 3-D landmark.

        Parameters
        ----------
        point_3d : array_like, shape (3,)
            World-frame coordinates of the 3-D point.

        Returns
        -------
        idx : int
            Index of the newly added point.
        """
        pt = np.asarray(point_3d, dtype=np.float64).ravel()
        self.points.append(pt.copy())
        return len(self.points) - 1

    def add_observation(
        self,
        cam_idx: int,
        pt_idx: int,
        pixel: NDArray[np.float64],
    ) -> None:
        """Add a 2-D observation of a landmark in a camera.

        Parameters
        ----------
        cam_idx : int
            Camera index.
        pt_idx : int
            Point index.
        pixel : array_like, shape (2,)
            Observed pixel coordinates (u, v).
        """
        pixel = np.asarray(pixel, dtype=np.float64).ravel()
        self.observations.append(
            (cam_idx, pt_idx, float(pixel[0]), float(pixel[1]))
        )

    # ------------------------------------------------------------------
    # Projection Jacobian
    # ------------------------------------------------------------------

    @staticmethod
    def projection_jacobian(
        K: NDArray[np.float64],
        R: NDArray[np.float64],
        t: NDArray[np.float64],
        X: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Jacobians of the projection w.r.t. camera pose and 3-D point.

        Given the **world-to-camera** rotation R and translation t, the
        camera-frame point is P = R.X + t and the projection is
        pi(P) = K . P / P_z.

        This method returns two Jacobians:

        .. math::

            J_{cam} = \frac{\partial\pi}{\partial P}\;
            [-I_3 \;|\; [P]_\times]
            \in \mathbb{R}^{2 \times 6}

        .. math::

            J_{point} = \frac{\partial\pi}{\partial P}\;R
            \in \mathbb{R}^{2 \times 3}

        The camera Jacobian is computed w.r.t. a right perturbation of the
        **camera-to-world** pose.

        Derivation: under right perturbation T_cw -> T_cw . Exp(d_xi),
        the world-to-camera transform becomes
        T_wc_new = Exp(-d_xi) . T_wc.  Expanding to first order:

            P_new = (I - [d_omega]x) . P - d_rho

        so dP/d(d_xi) = [-I_3 | [P]x] where xi = [rho; omega].

        Parameters
        ----------
        K : ndarray, shape (3, 3)
            Camera intrinsic matrix.
        R : ndarray, shape (3, 3)
            World-to-camera rotation.
        t : ndarray, shape (3,)
            World-to-camera translation.
        X : ndarray, shape (3,)
            3-D point in world coordinates.

        Returns
        -------
        J_cam : ndarray, shape (2, 6)
            Jacobian w.r.t. the camera-to-world twist perturbation
            d_xi = [d_rho; d_omega].
        J_point : ndarray, shape (2, 3)
            Jacobian w.r.t. the 3-D point coordinates.
        """
        P = R @ X + t
        Xc, Yc, Zc = P
        fx, fy = K[0, 0], K[1, 1]

        Z_inv = 1.0 / Zc
        Z_inv2 = Z_inv * Z_inv

        dpi_dP = np.array([
            [fx * Z_inv, 0.0,        -fx * Xc * Z_inv2],
            [0.0,        fy * Z_inv, -fy * Yc * Z_inv2],
        ])

        dP_dxi = np.zeros((3, 6))
        dP_dxi[:, :3] = -np.eye(3)
        dP_dxi[:, 3:] = skew(P)

        J_cam = dpi_dP @ dP_dxi       # (2, 6)
        J_point = dpi_dP @ R           # (2, 3)

        return J_cam, J_point

    # ------------------------------------------------------------------
    # Optimisation helpers
    # ------------------------------------------------------------------

    def _project_observation(
        self,
        cam_idx: int,
        pt_idx: int,
        u_obs: float,
        v_obs: float,
        robust_kernel: Optional[Callable[[float], Tuple[float, float]]],
    ) -> Optional[
        Tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], int, int]
    ]:
        T_cw = self.cameras[cam_idx]
        T_wc = se3_inverse(T_cw)
        R_wc, t_wc = se3_to_Rt(T_wc)
        X = self.points[pt_idx]

        P = R_wc @ X + t_wc
        if P[2] <= _EPS:
            return None

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        proj = np.array([
            fx * P[0] / P[2] + cx,
            fy * P[1] / P[2] + cy,
        ])

        e = proj - np.array([u_obs, v_obs])

        w = 1.0
        if robust_kernel is not None:
            r_norm = float(np.linalg.norm(e))
            if r_norm > _EPS:
                _, w = robust_kernel(r_norm)

        Jc, Jp = self.projection_jacobian(self.K, R_wc, t_wc, X)
        return e, w, Jc, Jp, 6 * cam_idx, 3 * pt_idx

    def _schur_complement_solve(
        self,
        H_cc: NDArray[np.float64],
        H_cp: NDArray[np.float64],
        H_pp_blocks: List[NDArray[np.float64]],
        b_c: NDArray[np.float64],
        b_p: NDArray[np.float64],
        n_p: int,
        cam_dim: int,
        fix_set: set,
        damping: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        S = H_cc.copy()
        Hpp_inv_bp = np.zeros(3 * n_p)

        for k in range(n_p):
            s = 3 * k
            block_inv = np.linalg.inv(H_pp_blocks[k] + damping * np.eye(3))
            Hcp_k = H_cp[:, s:s + 3]
            S -= Hcp_k @ (block_inv @ Hcp_k.T)
            Hpp_inv_bp[s:s + 3] = block_inv @ b_p[s:s + 3]

        rhs_c = -(b_c - H_cp @ Hpp_inv_bp)

        for idx in fix_set:
            s = 6 * idx
            S[s:s + 6, :] = 0.0
            S[:, s:s + 6] = 0.0
            S[s:s + 6, s:s + 6] = np.eye(6)
            rhs_c[s:s + 6] = 0.0

        S += damping * np.eye(cam_dim)
        dx_c = np.linalg.solve(S, rhs_c)

        H_pc = H_cp.T
        dx_p = np.zeros(3 * n_p)
        for k in range(n_p):
            s = 3 * k
            block_inv = np.linalg.inv(H_pp_blocks[k] + damping * np.eye(3))
            dx_p[s:s + 3] = block_inv @ (
                -b_p[s:s + 3] - H_pc[s:s + 3, :] @ dx_c
            )

        return dx_c, dx_p

    def _apply_pose_updates(
        self,
        dx_c: NDArray[np.float64],
        dx_p: NDArray[np.float64],
        n_c: int,
        n_p: int,
        fix_set: set,
    ) -> None:
        for k in range(n_c):
            if k not in fix_set:
                self.cameras[k] = se3_compose(
                    self.cameras[k],
                    se3_exp(dx_c[6 * k : 6 * k + 6]),
                )
        for k in range(n_p):
            self.points[k] = self.points[k] + dx_p[3 * k : 3 * k + 3]

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_iterations: int = 10,
        fix_cameras: Optional[List[int]] = None,
        robust_kernel: Optional[Callable[[float], Tuple[float, float]]] = None,
        damping: float = 1e-6,
    ) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
        r"""Run Gauss-Newton bundle adjustment with the Schur complement.

        Parameters
        ----------
        n_iterations : int
            Maximum Gauss-Newton iterations.
        fix_cameras : list of int, optional
            Camera indices held fixed during optimisation.
        robust_kernel : callable, optional
            A function (residual) -> (cost, weight) such as
            :func:`huber_kernel` or :func:`cauchy_kernel`.  When
            None, plain squared loss is used.
        damping : float
            Small diagonal regularisation added to point blocks for
            numerical stability (Levenberg-Marquardt style).

        Returns
        -------
        cameras : list of ndarray, each (4, 4)
            Optimised camera-to-world poses.
        points : list of ndarray, each (3,)
            Optimised 3-D landmark positions.
        """
        if fix_cameras is None:
            fix_cameras = []
        fix_set = set(fix_cameras)

        n_c = len(self.cameras)
        n_p = len(self.points)
        cam_dim = 6 * n_c

        for _ in range(n_iterations):
            H_cc = np.zeros((cam_dim, cam_dim))
            H_cp = np.zeros((cam_dim, 3 * n_p))
            H_pp_blocks: List[NDArray[np.float64]] = [
                np.zeros((3, 3)) for _ in range(n_p)
            ]
            b_c = np.zeros(cam_dim)
            b_p = np.zeros(3 * n_p)

            for cam_idx, pt_idx, u_obs, v_obs in self.observations:
                result = self._project_observation(
                    cam_idx, pt_idx, u_obs, v_obs, robust_kernel,
                )
                if result is None:
                    continue
                e, w, Jc, Jp, si, pi = result

                H_cc[si:si + 6, si:si + 6] += w * (Jc.T @ Jc)
                H_cp[si:si + 6, pi:pi + 3] += w * (Jc.T @ Jp)
                H_pp_blocks[pt_idx] += w * (Jp.T @ Jp)
                b_c[si:si + 6] += w * (Jc.T @ e)
                b_p[pi:pi + 3] += w * (Jp.T @ e)

            dx_c, dx_p = self._schur_complement_solve(
                H_cc, H_cp, H_pp_blocks, b_c, b_p,
                n_p, cam_dim, fix_set, damping,
            )

            self._apply_pose_updates(dx_c, dx_p, n_c, n_p, fix_set)

            if np.linalg.norm(np.concatenate([dx_c, dx_p])) < 1e-8:
                break

        return (
            [c.copy() for c in self.cameras],
            [p.copy() for p in self.points],
        )


# ===================================================================
#  Loop Detector
# ===================================================================


class LoopDetector:
    r"""Loop-closure detection using a bag-of-visual-words (BoVW) approach.

    When a camera revisits a previously mapped location, the accumulated
    drift in odometry must be corrected.  Detecting such *loop closures*
    is a visual place-recognition problem.

    Algorithm
    ---------
    1. **Build a visual vocabulary** by clustering a large set of local
       descriptors (e.g. ORB) into K visual words via k-means.
    2. **Represent each keyframe** as a normalised histogram (L2) over
       visual words -- its *bag-of-words* (BoW) vector.
    3. **Query**: for a new frame, compute its BoW vector and compare
       against all stored keyframes using **cosine similarity**:

       .. math::

           \mathrm{sim}(h_a, h_b)
           = \frac{h_a \cdot h_b}{\|h_a\|\;\|h_b\|}

       (Since histograms are L2-normalised, this reduces to a dot
       product.)

    4. Keyframes with similarity above a threshold are returned as
       **loop-closure candidates**, to be verified downstream via
       geometric consistency (e.g. fundamental / homography estimation).

    Parameters
    ----------
    vocab_size : int
        Number of visual words K.
    similarity_threshold : float
        Minimum cosine similarity to consider a candidate loop closure.

    Notes
    -----
    This is a simplified educational implementation.  Production systems
    use hierarchical vocabularies (vocabulary trees), TF-IDF weighting,
    and temporal consistency checks; see DBoW2 / DBoW3 [4].
    """

    def __init__(
        self,
        vocab_size: int = 100,
        similarity_threshold: float = 0.3,
    ) -> None:
        """Initialise the loop detector with vocabulary size and threshold."""
        self.vocab_size: int = vocab_size
        self.similarity_threshold: float = similarity_threshold

        self.vocabulary: Optional[NDArray[np.float64]] = None
        self.keyframes: Dict[int, NDArray[np.float64]] = {}

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def build_vocabulary(
        self, all_descriptors: NDArray[np.float64]
    ) -> None:
        r"""Build the visual vocabulary via k-means clustering.

        Partitions the descriptor space into :attr:`vocab_size` Voronoi
        cells.  Each cell centre becomes a *visual word*.

        The k-means objective minimises the total within-cluster variance:

        .. math::

            \min_{\{\mu_k\}}
            \sum_{i=1}^{N}
            \min_{k}
            \| d_i - \mu_k \|^2

        Parameters
        ----------
        all_descriptors : ndarray, shape (N, D)
            Pooled descriptors from training images.  For binary
            descriptors (e.g. ORB) these should be cast to float64
            beforehand; k-means uses Euclidean distance which serves as
            a reasonable proxy for Hamming distance in R^D.
        """
        data = np.asarray(all_descriptors, dtype=np.float64)
        k = min(self.vocab_size, len(data))
        centroids, _ = kmeans2(data, k, minit="points", iter=20)
        self.vocabulary = centroids

    # ------------------------------------------------------------------
    # Keyframe management
    # ------------------------------------------------------------------

    def _to_histogram(
        self, descriptors: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Convert a set of descriptors to an L2-normalised BoW histogram."""
        codes, _ = vq(
            np.asarray(descriptors, dtype=np.float64), self.vocabulary
        )
        hist = np.bincount(
            codes, minlength=len(self.vocabulary)
        ).astype(np.float64)
        norm = np.linalg.norm(hist)
        if norm > _EPS:
            hist /= norm
        return hist

    def add_keyframe(
        self, frame_id: int, descriptors: NDArray[np.float64]
    ) -> None:
        """Register a keyframe's descriptors in the database.

        Parameters
        ----------
        frame_id : int
            Unique identifier for this keyframe.
        descriptors : ndarray, shape (M, D)
            Local feature descriptors for the keyframe.

        Raises
        ------
        RuntimeError
            If :meth:`build_vocabulary` has not been called yet.
        """
        if self.vocabulary is None:
            raise RuntimeError(
                "Visual vocabulary has not been built.  "
                "Call build_vocabulary() first."
            )
        self.keyframes[frame_id] = self._to_histogram(descriptors)

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------

    def detect_loop(
        self, descriptors: NDArray[np.float64]
    ) -> List[Tuple[int, float]]:
        r"""Check whether the current frame closes a loop.

        Computes the BoW histogram of the query descriptors and compares
        it against every stored keyframe via cosine similarity.

        Parameters
        ----------
        descriptors : ndarray, shape (M, D)
            Descriptors of the current (query) frame.

        Returns
        -------
        candidates : list of (frame_id, similarity)
            Keyframes whose similarity exceeds
            :attr:`similarity_threshold`, sorted in descending order of
            similarity score.
        """
        if self.vocabulary is None or not self.keyframes:
            return []

        query_hist = self._to_histogram(descriptors)
        candidates: List[Tuple[int, float]] = []

        for fid, stored_hist in self.keyframes.items():
            sim = float(np.dot(query_hist, stored_hist))
            if sim >= self.similarity_threshold:
                candidates.append((fid, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
