"""Rigid-body transformation utilities for 3-D vision and mapping.

This module provides numerically robust, pure-NumPy implementations of:

* Rotation representations  – Euler angles, Rodrigues, quaternions
* Lie-group operations      – SO(3) and SE(3) exponential / logarithm maps
* Pose composition          – SE(3) homogeneous-matrix algebra
* Interpolation             – quaternion SLERP

Convention summary
------------------
* **Euler angles** follow the ZYX (yaw-pitch-roll) intrinsic convention,
  equivalent to the extrinsic XYZ convention:  R = Rz(ψ) Ry(θ) Rx(φ).
* **Quaternions** use the *scalar-first* Hamilton convention  q = [w, x, y, z].
* **SE(3) twist vectors** are ordered  ξ = [ρ; ω]  (translation part first,
  rotation part second), matching the Lie-algebra basis used in Barfoot's
  *State Estimation for Robotics*.
* All angles are in **radians**.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Small-angle threshold used by Taylor-expansion fallbacks.
# ---------------------------------------------------------------------------
_EPS = 1e-10
_SMALL_ANGLE = 1e-7
# Threshold (on pi - theta) below which the symmetric eigenvector form is used
# to recover the rotation axis, because the antisymmetric part has lost
# precision.  See so3_log.
_PI_TOL = 1e-2


# ===================================================================
#  1. Rotation matrix from Euler angles (ZYX intrinsic convention)
# ===================================================================


def rotation_matrix_from_euler(
    roll: float, pitch: float, yaw: float
) -> NDArray[np.float64]:
    r"""Build a 3×3 rotation matrix from ZYX Euler angles.

    The rotation is applied in the order  **R = Rz(yaw) · Ry(pitch) · Rx(roll)**,
    which corresponds to an intrinsic ZYX (or equivalently extrinsic XYZ) sequence.

    Individual axis rotations
    -------------------------
    .. math::

        R_x(\phi) = \begin{pmatrix}
            1 & 0        & 0         \\
            0 & \cos\phi & -\sin\phi \\
            0 & \sin\phi &  \cos\phi
        \end{pmatrix}, \quad
        R_y(\theta) = \begin{pmatrix}
             \cos\theta & 0 & \sin\theta \\
             0          & 1 & 0          \\
            -\sin\theta & 0 & \cos\theta
        \end{pmatrix}, \quad
        R_z(\psi) = \begin{pmatrix}
            \cos\psi & -\sin\psi & 0 \\
            \sin\psi &  \cos\psi & 0 \\
            0        &  0        & 1
        \end{pmatrix}

    Parameters
    ----------
    roll : float
        Rotation about the x-axis (φ), in radians.
    pitch : float
        Rotation about the y-axis (θ), in radians.
    yaw : float
        Rotation about the z-axis (ψ), in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Proper rotation matrix (det = +1, R^T R = I).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ]
    )
    Ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ]
    )
    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return Rz @ Ry @ Rx


# ===================================================================
#  2. Skew-symmetric matrix  /  unskew
# ===================================================================


def skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Return the 3×3 skew-symmetric (hat) matrix of a 3-vector.

    .. math::

        [v]_\times = \begin{pmatrix}
             0   & -v_3 &  v_2 \\
             v_3 &  0   & -v_1 \\
            -v_2 &  v_1 &  0
        \end{pmatrix}

    so that  ``skew(a) @ b == np.cross(a, b)``  for any 3-vectors *a*, *b*.

    Parameters
    ----------
    v : array_like, shape (3,)

    Returns
    -------
    S : ndarray, shape (3, 3)
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def unskew(S: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Extract the 3-vector from a skew-symmetric matrix.

    Inverse of :func:`skew`:  ``unskew(skew(v)) == v``.

    Uses the off-diagonal entries:

    .. math::

        v = \bigl(S_{32},\; S_{13},\; S_{21}\bigr)

    Parameters
    ----------
    S : array_like, shape (3, 3)

    Returns
    -------
    v : ndarray, shape (3,)
    """
    S = np.asarray(S, dtype=np.float64)
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# ===================================================================
#  3. Rodrigues' rotation formula  (axis-angle ↔ matrix)
# ===================================================================


def rodrigues(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    r"""Compute a rotation matrix via Rodrigues' rotation formula.

    Given a *unit* axis  ω̂  and angle  θ  the rotation matrix is

    .. math::

        R = I + \sin\theta\;[\hat\omega]_\times
            + (1 - \cos\theta)\;[\hat\omega]_\times^{2}

    The axis is normalised internally, so it need not be unit-length on input
    (its direction is what matters).

    Parameters
    ----------
    axis : array_like, shape (3,)
        Rotation axis (will be normalised).
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    axis = np.asarray(axis, dtype=np.float64).ravel()
    n = np.linalg.norm(axis)
    if n < _EPS:
        return np.eye(3)
    omega_hat = axis / n
    K = skew(omega_hat)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rodrigues_inv(R: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    r"""Extract axis and angle from a rotation matrix (inverse Rodrigues).

    The angle is recovered as

    .. math::

        \theta = \arccos\!\Bigl(\frac{\operatorname{tr}(R) - 1}{2}\Bigr)

    and the axis from the antisymmetric part of *R*:

    .. math::

        \hat\omega = \frac{1}{2\sin\theta}
        \begin{pmatrix} R_{32}-R_{23} \\ R_{13}-R_{31} \\ R_{21}-R_{12} \end{pmatrix}

    Edge cases:
      * **θ ≈ 0** : axis is undefined; we return  [0, 0, 1]  by convention.
      * **θ ≈ π** : the antisymmetric part vanishes.  We read the axis from the
        dominant diagonal entry of  R + I = 2 ω̂ω̂ᵀ,  which is numerically
        robust where the column heuristic is not.

    Parameters
    ----------
    R : array_like, shape (3, 3)

    Returns
    -------
    axis : ndarray, shape (3,)
        Unit rotation axis.
    angle : float
        Rotation angle in [0, π].
    """
    R = np.asarray(R, dtype=np.float64)
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))

    if angle < _SMALL_ANGLE:
        return np.array([0.0, 0.0, 1.0]), 0.0

    if np.pi - angle < _PI_TOL:
        # θ ≈ π  →  the axis is the eigenvalue-1 eigenvector of the symmetric
        # matrix (R + Rᵀ)/2, with the sign fixed by the antisymmetric part.
        S = 0.5 * (R + R.T)
        _, evecs = np.linalg.eigh(S)
        axis = evecs[:, 2]
        A = 0.5 * (R - R.T)
        k = int(np.argmin(np.abs(axis)))
        q = np.cross(axis, np.eye(3)[k])
        nq = np.linalg.norm(q)
        if nq < _EPS:
            return axis, float(angle)
        q = q / nq
        if np.dot(A @ q, np.cross(axis, q)) < 0.0:
            axis = -axis
        return axis, float(angle)

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    )
    axis = axis / (2.0 * np.sin(angle))
    return axis, float(angle)


# ===================================================================
#  4. Quaternion operations  (scalar-first  [w, x, y, z])
# ===================================================================


def normalize_quaternion(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the unit quaternion  q / ‖q‖.

    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion [w, x, y, z].

    Returns
    -------
    q_unit : ndarray, shape (4,)
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quaternion_conjugate(q: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Quaternion conjugate (which equals the inverse for unit quaternions).

    .. math::

        q^* = (w,\; -x,\; -y,\; -z)

    Parameters
    ----------
    q : array_like, shape (4,)

    Returns
    -------
    q_conj : ndarray, shape (4,)
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(
    q1: NDArray[np.float64], q2: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Hamilton product of two quaternions (scalar-first convention).

    .. math::

        q_1 \otimes q_2 =
        \begin{pmatrix}
            w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
            w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
            w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
            w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
        \end{pmatrix}

    Parameters
    ----------
    q1, q2 : array_like, shape (4,)

    Returns
    -------
    q : ndarray, shape (4,)
    """
    q1 = np.asarray(q1, dtype=np.float64).ravel()
    q2 = np.asarray(q2, dtype=np.float64).ravel()
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_to_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Convert a unit quaternion to a 3×3 rotation matrix.

    For  q = (w, x, y, z)  the rotation matrix is

    .. math::

        R = \begin{pmatrix}
            1 - 2(y^2+z^2) & 2(xy - wz)     & 2(xz + wy)     \\
            2(xy + wz)     & 1 - 2(x^2+z^2) & 2(yz - wx)     \\
            2(xz - wy)     & 2(yz + wx)     & 1 - 2(x^2+y^2)
        \end{pmatrix}

    The quaternion is normalised internally.

    Parameters
    ----------
    q : array_like, shape (4,)

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    q = normalize_quaternion(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quaternion(R: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Convert a 3×3 rotation matrix to a unit quaternion via Shepperd's method.

    Shepperd's method avoids the numerical instability of the naive
    ``θ = arccos(…)`` approach by choosing the quaternion component with the
    largest magnitude first.

    The four candidate magnitudes are:

    .. math::

        4w^2 = 1 + R_{00} + R_{11} + R_{22}  \\
        4x^2 = 1 + R_{00} - R_{11} - R_{22}  \\
        4y^2 = 1 - R_{00} + R_{11} - R_{22}  \\
        4z^2 = 1 - R_{00} - R_{11} + R_{22}

    We compute the component with the largest squared magnitude and derive the
    remaining three from symmetric off-diagonal sums / differences.

    The returned quaternion is normalised and has  w ≥ 0  (canonical sign).

    Parameters
    ----------
    R : array_like, shape (3, 3)

    Returns
    -------
    q : ndarray, shape (4,)
        Unit quaternion [w, x, y, z].
    """
    R = np.asarray(R, dtype=np.float64)

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    candidates = np.array(
        [
            trace,  # proportional to 4w²-1
            R[0, 0] - R[1, 1] - R[2, 2],  # 4x²-1
            -R[0, 0] + R[1, 1] - R[2, 2],  # 4y²-1
            -R[0, 0] - R[1, 1] + R[2, 2],  # 4z²-1
        ]
    )
    idx = int(np.argmax(candidates))

    if idx == 0:
        w = np.sqrt(1.0 + trace) / 2.0
        f = 1.0 / (4.0 * w)
        x = (R[2, 1] - R[1, 2]) * f
        y = (R[0, 2] - R[2, 0]) * f
        z = (R[1, 0] - R[0, 1]) * f
    elif idx == 1:
        x = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) / 2.0
        f = 1.0 / (4.0 * x)
        w = (R[2, 1] - R[1, 2]) * f
        y = (R[0, 1] + R[1, 0]) * f
        z = (R[0, 2] + R[2, 0]) * f
    elif idx == 2:
        y = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) / 2.0
        f = 1.0 / (4.0 * y)
        w = (R[0, 2] - R[2, 0]) * f
        x = (R[0, 1] + R[1, 0]) * f
        z = (R[1, 2] + R[2, 1]) * f
    else:
        z = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) / 2.0
        f = 1.0 / (4.0 * z)
        w = (R[1, 0] - R[0, 1]) * f
        x = (R[0, 2] + R[2, 0]) * f
        y = (R[1, 2] + R[2, 1]) * f

    q = np.array([w, x, y, z])
    if q[0] < 0.0:
        q = -q
    return normalize_quaternion(q)


def slerp(
    q0: NDArray[np.float64],
    q1: NDArray[np.float64],
    t: float,
) -> NDArray[np.float64]:
    r"""Spherical linear interpolation (SLERP) between two unit quaternions.

    .. math::

        \operatorname{slerp}(q_0, q_1, t) =
        q_0 \,\frac{\sin\bigl((1-t)\,\Omega\bigr)}{\sin\Omega}
        + q_1 \,\frac{\sin(t\,\Omega)}{\sin\Omega}

    where  cos Ω = q₀ · q₁.

    Special cases
    -------------
    * If  cos Ω < 0  we negate  q₁  so the interpolation takes the short arc.
    * If  |cos Ω| ≈ 1  (quaternions nearly identical or antipodal) we fall back
      to normalised linear interpolation (NLERP) to avoid division by zero.

    Parameters
    ----------
    q0, q1 : array_like, shape (4,)
        Unit quaternions (scalar-first).
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    q : ndarray, shape (4,)
        Interpolated unit quaternion.
    """
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = float(np.dot(q0, q1))

    # Ensure shortest path.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 1.0 - _SMALL_ANGLE:
        # Nearly identical – fall back to NLERP.
        q = (1.0 - t) * q0 + t * q1
        return normalize_quaternion(q)

    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    s0 = np.sin((1.0 - t) * omega) / sin_omega
    s1 = np.sin(t * omega) / sin_omega
    return normalize_quaternion(s0 * q0 + s1 * q1)


# ===================================================================
#  5. SE(3) pose operations (4×4 homogeneous matrices)
# ===================================================================


def se3_from_Rt(R: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Construct a 4×4 homogeneous transformation from *R* and *t*.

    .. math::

        T = \begin{pmatrix} R & t \\ 0^T & 1 \end{pmatrix}

    Parameters
    ----------
    R : array_like, shape (3, 3)
        Rotation matrix.
    t : array_like, shape (3,) or (3, 1)
        Translation vector.

    Returns
    -------
    T : ndarray, shape (4, 4)
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_to_Rt(
    T: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract the rotation matrix and translation vector from a 4×4 pose.

    Parameters
    ----------
    T : array_like, shape (4, 4)

    Returns
    -------
    R : ndarray, shape (3, 3)
    t : ndarray, shape (3,)
    """
    T = np.asarray(T, dtype=np.float64)
    return T[:3, :3].copy(), T[:3, 3].copy()


def se3_compose(
    T1: NDArray[np.float64], T2: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Compose two SE(3) transformations  T₁ · T₂.

    .. math::

        T_1 T_2 = \begin{pmatrix}
            R_1 R_2 & R_1 t_2 + t_1 \\
            0^T     & 1
        \end{pmatrix}

    Parameters
    ----------
    T1, T2 : array_like, shape (4, 4)

    Returns
    -------
    T : ndarray, shape (4, 4)
    """
    T1 = np.asarray(T1, dtype=np.float64)
    T2 = np.asarray(T2, dtype=np.float64)
    return T1 @ T2


def se3_inverse(T: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Efficient inverse of an SE(3) transformation.

    .. math::

        T^{-1} = \begin{pmatrix}
            R^T  & -R^T t \\
            0^T  & 1
        \end{pmatrix}

    This avoids a general 4×4 matrix inversion by exploiting the orthogonality
    of *R*.

    Parameters
    ----------
    T : array_like, shape (4, 4)

    Returns
    -------
    T_inv : ndarray, shape (4, 4)
    """
    T = np.asarray(T, dtype=np.float64)
    R, t = se3_to_Rt(T)
    Rt = R.T
    T_inv = np.eye(4)
    T_inv[:3, :3] = Rt
    T_inv[:3, 3] = -Rt @ t
    return T_inv


# ===================================================================
#  6. SO(3) Lie-group exponential / logarithm / Jacobians
# ===================================================================


def so3_exp(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""SO(3) exponential map: axis-angle 3-vector → rotation matrix.

    .. math::

        \exp([\omega]_\times) = I
            + \frac{\sin\theta}{\theta}\;[\omega]_\times
            + \frac{1 - \cos\theta}{\theta^2}\;[\omega]_\times^2

    where  θ = ‖ω‖.

    For small θ the coefficients are replaced by their second-order Taylor
    expansions to avoid catastrophic cancellation:

    .. math::

        \frac{\sin\theta}{\theta} \approx 1 - \tfrac{\theta^2}{6}, \qquad
        \frac{1-\cos\theta}{\theta^2} \approx \tfrac12 - \tfrac{\theta^2}{24}.

    Parameters
    ----------
    omega : array_like, shape (3,)
        Axis-angle vector (direction = axis, norm = angle).

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    omega = np.asarray(omega, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(omega))
    K = skew(omega)

    if theta < _SMALL_ANGLE:
        a = 1.0 - theta**2 / 6.0
        b = 0.5 - theta**2 / 24.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta**2)

    return np.eye(3) + a * K + b * (K @ K)


def so3_log(R: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""SO(3) logarithm map: rotation matrix → axis-angle 3-vector.

    The angle is

    .. math::

        \theta = \arccos\!\Bigl(\frac{\operatorname{tr}(R) - 1}{2}\Bigr)

    and the axis is

    .. math::

        \omega = \frac{\theta}{2\sin\theta}
        \begin{pmatrix} R_{32}-R_{23} \\ R_{13}-R_{31} \\ R_{21}-R_{12}
        \end{pmatrix}.

    Edge cases
    ----------
    * **θ ≈ 0** :  Use the first-order approximation
      ω ≈ ½ [R₃₂−R₂₃, R₁₃−R₃₁, R₂₁−R₁₂]ᵀ.
    * **θ ≈ π** :  The antisymmetric part vanishes.  We use the identity
      R + I = 2 ω̂ω̂ᵀ and read the axis off the **largest diagonal entry** of
      R + I, which avoids the loss of sign precision that occurs when the
      axis is taken from a merely near-dominant column.

    Notes
    -----
    The returned vector always satisfies ‖ω‖ = θ ∈ [0, π], i.e. it is the
    principal branch of the logarithm.

    Parameters
    ----------
    R : array_like, shape (3, 3)

    Returns
    -------
    omega : ndarray, shape (3,)
        Axis-angle vector with  ‖ω‖ = θ ∈ [0, π].
    """
    R = np.asarray(R, dtype=np.float64)
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_angle))

    if theta < _SMALL_ANGLE:
        return 0.5 * np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]
        )

    # Generic branch.  The antisymmetric part  (R - Rᵀ)/2 = sin θ · [ω̂]×
    # is exact but degrades as 1/sin θ near θ = π.  For θ > π/2 we instead
    # take the axis from the dominant eigenvector of the symmetric matrix
    #   (R + Rᵀ)/2 = I + (1 − cos θ)[ω̂]²_× ,
    # whose eigenvalue 1 eigenvector is exactly ω̂, and fix the sign with the
    # antisymmetric part.  This is uniformly accurate over θ ∈ (0, π).
    if theta < 0.5 * np.pi:
        a = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]
        )
        return (theta / (2.0 * np.sin(theta))) * a

    S = 0.5 * (R + R.T)
    _, evecs = np.linalg.eigh(S)
    axis = evecs[:, 2]  # eigenvector for the largest λ
    A = 0.5 * (R - R.T)
    k = int(np.argmin(np.abs(axis)))
    q = np.cross(axis, np.eye(3)[k])
    nq = np.linalg.norm(q)
    if nq < _EPS:
        return axis * theta
    q = q / nq
    if np.dot(A @ q, np.cross(axis, q)) < 0.0:
        axis = -axis
    return axis * theta


def _so3_left_jacobian_coeffs(
    theta: float,
) -> tuple[float, float]:
    r"""Compute the two scalar coefficients for the left Jacobian.

    Returns (a, b) where

    .. math::

        J_l = I + a\;[\omega]_\times + b\;[\omega]_\times^2

    with  a = (1 − cos θ)/θ²,  b = (θ − sin θ)/θ³.

    For small θ second-order Taylor expansions are used.
    """
    if theta < _SMALL_ANGLE:
        a = 0.5 - theta**2 / 24.0
        b = 1.0 / 6.0 - theta**2 / 120.0
    else:
        a = (1.0 - np.cos(theta)) / (theta**2)
        b = (theta - np.sin(theta)) / (theta**3)
    return a, b


def so3_left_jacobian(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Left Jacobian of SO(3).

    .. math::

        J_l(\omega) = I
            + \frac{1 - \cos\theta}{\theta^2}\;[\omega]_\times
            + \frac{\theta - \sin\theta}{\theta^3}\;[\omega]_\times^2

    where  θ = ‖ω‖.  The left Jacobian relates a small perturbation in the
    tangent space to the group:

    .. math::

        \exp\bigl([\omega + \delta\omega]_\times\bigr)
        \approx \exp\bigl([J_l\,\delta\omega]_\times\bigr)\;\exp([\omega]_\times)

    Parameters
    ----------
    omega : array_like, shape (3,)

    Returns
    -------
    J : ndarray, shape (3, 3)
    """
    omega = np.asarray(omega, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(omega))
    K = skew(omega)
    a, b = _so3_left_jacobian_coeffs(theta)
    return np.eye(3) + a * K + b * (K @ K)


def so3_right_jacobian(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Right Jacobian of SO(3).

    .. math::

        J_r(\omega) = J_l(-\omega)

    Parameters
    ----------
    omega : array_like, shape (3,)

    Returns
    -------
    J : ndarray, shape (3, 3)
    """
    return so3_left_jacobian(-np.asarray(omega, dtype=np.float64).ravel())


# ===================================================================
#  7. SE(3) Lie-group exponential / logarithm / adjoint
# ===================================================================


def _V_matrix(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Compute the *V* matrix used in the SE(3) exponential map.

    .. math::

        V = I + \frac{1-\cos\theta}{\theta^2}\;[\omega]_\times
              + \frac{\theta - \sin\theta}{\theta^3}\;[\omega]_\times^2

    This is identical in form to the SO(3) left Jacobian (with the *same*
    coefficients) because V = J_l(ω).  It maps the translational part ρ of the
    twist to the translation component of the resulting SE(3) matrix:  t = Vρ.
    """
    return so3_left_jacobian(omega)


def _V_matrix_inv(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Inverse of the *V* matrix.

    .. math::

        V^{-1} = I
            - \tfrac12\;[\omega]_\times
            + \frac{1}{\theta^2}\!\left(1
              - \frac{\theta\sin\theta}{2(1-\cos\theta)}\right)
              [\omega]_\times^2

    The reflection formula is used for large θ; for small θ the series

    .. math::

        c = \frac{1}{12} + \frac{\theta^2}{720} + \frac{\theta^4}{30240}

    is used instead, because the direct quotient suffers catastrophic
    cancellation as θ → 0.
    """
    omega = np.asarray(omega, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(omega))
    K = skew(omega)

    if theta < 1e-2:
        t2 = theta * theta
        c = 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
    else:
        c = (1.0 / (theta**2)) * (
            1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta)))
        )

    return np.eye(3) - 0.5 * K + c * (K @ K)


def se3_exp(xi: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""SE(3) exponential map: 6-vector twist → 4×4 transformation.

    The twist  ξ = [ρ; ω]  (translation part ρ ∈ ℝ³ first, rotation part ω ∈ ℝ³
    second) is mapped to

    .. math::

        T = \exp\!\bigl([\xi]_\wedge\bigr)
          = \begin{pmatrix}
              \exp([\omega]_\times) & V\rho \\
              0^T                    & 1
            \end{pmatrix}

    where  V = I + \frac{1-\cos\theta}{\theta^2}[\omega]_\times
                 + \frac{\theta-\sin\theta}{\theta^3}[\omega]_\times^2
    and  θ = ‖ω‖.

    Parameters
    ----------
    xi : array_like, shape (6,)
        Twist vector [ρ₁, ρ₂, ρ₃, ω₁, ω₂, ω₃].

    Returns
    -------
    T : ndarray, shape (4, 4)
    """
    xi = np.asarray(xi, dtype=np.float64).ravel()
    rho = xi[:3]
    omega = xi[3:]
    R = so3_exp(omega)
    V = _V_matrix(omega)
    t = V @ rho
    return se3_from_Rt(R, t)


def se3_log(T: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""SE(3) logarithm map: 4×4 transformation → 6-vector twist.

    Given  T = [R, t; 0, 1]  we compute

    .. math::

        \omega = \log(R), \quad \rho = V^{-1} t

    Parameters
    ----------
    T : array_like, shape (4, 4)

    Returns
    -------
    xi : ndarray, shape (6,)
        Twist [ρ₁, ρ₂, ρ₃, ω₁, ω₂, ω₃].
    """
    T = np.asarray(T, dtype=np.float64)
    R, t = se3_to_Rt(T)
    omega = so3_log(R)
    V_inv = _V_matrix_inv(omega)
    rho = V_inv @ t
    return np.concatenate([rho, omega])


def se3_adjoint(T: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""6×6 adjoint representation of an SE(3) element.

    .. math::

        \operatorname{Ad}_T =
        \begin{pmatrix}
            R        & [t]_\times R \\
            0_{3\times3} & R
        \end{pmatrix}

    The adjoint maps twists between frames:
    ξ_b = Ad_T · ξ_a  when  T  transforms from frame *a* to frame *b*.

    Parameters
    ----------
    T : array_like, shape (4, 4)

    Returns
    -------
    Ad : ndarray, shape (6, 6)
    """
    T = np.asarray(T, dtype=np.float64)
    R, t = se3_to_Rt(T)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(t) @ R
    Ad[3:, 3:] = R
    return Ad


# ===================================================================
#  8. Utility helpers
# ===================================================================


def is_rotation_matrix(R: NDArray[np.float64], tol: float = 1e-6) -> bool:
    r"""Check whether *R* is a proper rotation matrix.

    A proper rotation satisfies:

    1. Orthogonality:  R^T R = I  (within *tol* in Frobenius norm).
    2. Right-handedness:  det(R) = +1  (within *tol*).

    Parameters
    ----------
    R : array_like, shape (3, 3)
    tol : float
        Tolerance for both checks.

    Returns
    -------
    bool
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    err_orth = np.linalg.norm(R.T @ R - np.eye(3), ord="fro")
    err_det = abs(np.linalg.det(R) - 1.0)
    return bool(err_orth < tol and err_det < tol)


def angle_between_rotations(R1: NDArray[np.float64], R2: NDArray[np.float64]) -> float:
    r"""Geodesic angle between two rotation matrices.

    .. math::

        \Delta\theta = \arccos\!\Bigl(
            \frac{\operatorname{tr}(R_1^T R_2) - 1}{2}
        \Bigr)

    The result lies in [0, π].

    Parameters
    ----------
    R1, R2 : array_like, shape (3, 3)

    Returns
    -------
    angle : float
        Angle in radians.
    """
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    cos_angle = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_angle))
