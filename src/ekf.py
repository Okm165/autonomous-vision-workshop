r"""Extended Kalman Filter for Visual-Inertial Odometry.

Implements a 15-state error-state EKF for fusing IMU, GPS, barometer,
and magnetometer measurements. The state vector is:

.. math::

    \mathbf{x} = [p, v, q, b_a, b_g]^T \in \mathbb{R}^{15}

where *p* is position, *v* velocity, *q* orientation (unit quaternion),
*b_a* accelerometer bias, and *b_g* gyroscope bias.

References
----------
[1] Solà, "Quaternion kinematics for the error-state Kalman filter",
    arXiv:1711.02508, 2017.
[2] Trawny & Roumeliotis, "Indirect Kalman Filter for 3D Attitude
    Estimation", TR-2005-002, University of Minnesota, 2005.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _skew(v: NDArray) -> NDArray:
    """Return the 3×3 skew-symmetric matrix [v]× such that [v]× w = v × w."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def _quat_multiply(q: NDArray, r: NDArray) -> NDArray:
    """Hamilton quaternion product q ⊗ r with scalar-first convention [w,x,y,z]."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


def _quat_to_rotation(q: NDArray) -> NDArray:
    """Convert unit quaternion [w,x,y,z] to 3×3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rotation_to_quat(R: NDArray) -> NDArray:
    """Convert 3×3 rotation matrix to unit quaternion [w,x,y,z] (Shepperd's method)."""
    trace = np.trace(R)
    choices = np.array([trace, R[0, 0], R[1, 1], R[2, 2]])
    idx = np.argmax(choices)

    if idx == 0:
        w = 0.5 * np.sqrt(1.0 + trace)
        s = 0.25 / w
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif idx == 1:
        x = 0.5 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        s = 0.25 / x
        w = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 1] + R[1, 0]) * s
        z = (R[0, 2] + R[2, 0]) * s
    elif idx == 2:
        y = 0.5 * np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
        s = 0.25 / y
        w = (R[0, 2] - R[2, 0]) * s
        x = (R[0, 1] + R[1, 0]) * s
        z = (R[1, 2] + R[2, 1]) * s
    else:
        z = 0.5 * np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
        s = 0.25 / z
        w = (R[1, 0] - R[0, 1]) * s
        x = (R[0, 2] + R[2, 0]) * s
        y = (R[1, 2] + R[2, 1]) * s

    quat = np.array([w, x, y, z])
    return quat * np.sign(quat[0])  # enforce w >= 0


def _axis_angle_to_quat(v: NDArray) -> NDArray:
    """Convert rotation vector (axis-angle) v ∈ R³ to unit quaternion [w,x,y,z].

    q = [cos(θ/2), sin(θ/2)·v̂]  where θ = ‖v‖
    """
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / angle
    half = angle / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def _quat_to_axis_angle(q: NDArray) -> NDArray:
    """Convert unit quaternion [w,x,y,z] to rotation vector (axis-angle) ∈ R³.

    v = 2·arctan2(‖q_v‖, q_w) · q_v̂
    """
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    vec = q[1:]
    sin_half = np.linalg.norm(vec)
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, q[0])
    return angle * vec / sin_half


class VIO_EKF:
    """
    Extended Kalman Filter for Visual-Inertial Odometry.

    State vector (15-dimensional):
        x = [p, v, q, b_a, b_g]
    where:
        p ∈ R³  : position in world frame
        v ∈ R³  : velocity in world frame
        q ∈ R⁴  : orientation quaternion (body-to-world, i.e. R(q) maps body-frame vectors to world frame)
        b_a ∈ R³: accelerometer bias
        b_g ∈ R³: gyroscope bias

    Error-state formulation (multiplicative EKF):
    Instead of directly estimating q, we estimate a small rotation
    perturbation δθ ∈ R³ and compose: q_true = q_nominal ⊗ δq(δθ)

    This keeps the error state in a vector space (R¹⁵) where
    Kalman filter math works, avoiding manifold issues.

    IMU Process Model:
        ṗ = v
        v̇ = R(q) · (a_m - b_a - n_a) + g
        q̇ = ½ q ⊗ (ω_m - b_g - n_g)
        ḃ_a = n_ba  (random walk)
        ḃ_g = n_bg  (random walk)

    where a_m, ω_m are IMU measurements, n_* are noise terms.

    Discrete-time prediction (Euler integration):
        p_{k+1} = p_k + v_k·dt + ½(R_k(a_m - b_a) + g)·dt²
        v_{k+1} = v_k + (R_k(a_m - b_a) + g)·dt
        q_{k+1} = q_k ⊗ q((ω_m - b_g)·dt)
        b_{a,k+1} = b_{a,k}
        b_{g,k+1} = b_{g,k}

    Error-state transition matrix F (15×15):
        Derived from linearizing the error dynamics around the
        nominal state. Key blocks:
        - ∂δv/∂δθ = -R_k [a_m - b_a]×  (skew-symmetric!)
        - ∂δv/∂δb_a = -R_k
        - ∂δθ/∂δb_g = -I

    Process noise covariance Q (15×15):
        Diagonal blocks from IMU noise specifications:
        - σ_a² (accelerometer noise)
        - σ_g² (gyroscope noise)
        - σ_ba² (accel bias random walk)
        - σ_bg² (gyro bias random walk)

    Visual Measurement Model:
        When visual odometry provides a pose estimate T_vis:
        z = [p_vis; θ_vis]  (6D measurement)
        Innovation: δz = z - h(x̂)
        For rotation part: δθ = Log(R_nominal^T · R_vis)

        Measurement Jacobian H (6×15):
        H = [I₃ 0₃ 0₃ 0₃ 0₃]  (position rows)
            [0₃ 0₃ I₃ 0₃ 0₃]  (orientation rows)

    Kalman Update (Joseph form for numerical stability):
        K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹
        δx = K · δz
        P⁺ = (I - KH) P⁻ (I - KH)ᵀ + K R Kᵀ  (Joseph form)

        Apply δx to nominal state:
        p ← p + δp
        v ← v + δv
        q ← q ⊗ exp(δθ)
        b_a ← b_a + δb_a
        b_g ← b_g + δb_g

    Parameters
    ----------
    sigma_accel : float
        Accelerometer noise density (m/s²/√Hz).
    sigma_gyro : float
        Gyroscope noise density (rad/s/√Hz).
    sigma_accel_bias : float
        Accelerometer bias random walk.
    sigma_gyro_bias : float
        Gyroscope bias random walk.
    gravity : float
        Gravity magnitude (default 9.81).
    """

    _DIM_STATE: int = 15
    _DIM_MEAS: int = 6

    # Slice indices into the 15-D error state
    _P: slice = slice(0, 3)
    _V: slice = slice(3, 6)
    _THETA: slice = slice(6, 9)
    _BA: slice = slice(9, 12)
    _BG: slice = slice(12, 15)

    def __init__(
        self,
        sigma_accel: float = 0.1,
        sigma_gyro: float = 0.01,
        sigma_accel_bias: float = 0.001,
        sigma_gyro_bias: float = 0.0001,
        gravity: float = 9.81,
    ) -> None:
        """Initialise the EKF with IMU noise parameters and identity prior."""
        self.sigma_accel = sigma_accel
        self.sigma_gyro = sigma_gyro
        self.sigma_accel_bias = sigma_accel_bias
        self.sigma_gyro_bias = sigma_gyro_bias
        self.gravity = np.array([0.0, 0.0, -gravity])

        # Nominal state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        # Error-state covariance P (15×15)
        self.P = np.eye(self._DIM_STATE) * 1e-2
        self.P[self._BA, self._BA] = np.eye(3) * 1e-4
        self.P[self._BG, self._BG] = np.eye(3) * 1e-6

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, accel: np.ndarray, gyro: np.ndarray, dt: float) -> None:
        """
        IMU prediction step.

        1. Propagate nominal state using IMU measurements
        2. Compute error-state transition matrix F
        3. Propagate covariance: P = F @ P @ F.T + Q

        Discrete-time nominal propagation:
            a_body  = a_m - b_a
            ω_body  = ω_m - b_g
            R_k     = R(q_k)

            a_world = R_k·a_body + g
            p_{k+1} = p_k + v_k·dt + ½·a_world·dt²
            v_{k+1} = v_k + a_world·dt
            q_{k+1} = q_k ⊗ δq(ω_body·dt)

        Error-state transition (first-order):
            F = I + F_c·dt
        where F_c encodes the linearised continuous-time error dynamics.

        Parameters
        ----------
        accel : (3,) accelerometer reading in body frame (m/s²).
        gyro : (3,) gyroscope reading in body frame (rad/s).
        dt : float, time step in seconds.
        """
        a_body = accel - self.accel_bias
        w_body = gyro - self.gyro_bias
        R_k = _quat_to_rotation(self.quaternion)

        # --- Nominal state propagation ---
        a_world = R_k @ a_body + self.gravity
        self.position = (
            self.position + self.velocity * dt + 0.5 * a_world * dt * dt
        )
        self.velocity = self.velocity + a_world * dt

        dq = _axis_angle_to_quat(w_body * dt)
        self.quaternion = _quat_multiply(self.quaternion, dq)
        self.quaternion /= np.linalg.norm(self.quaternion)

        # --- Error-state transition matrix F (15×15) ---
        F = np.eye(self._DIM_STATE)

        F[self._P, self._V] = np.eye(3) * dt
        # Second-order position blocks from p += ½·R·a_body·dt²:
        #   ∂(R·a)/∂θ = -R·[a]×  →  F[P,θ] = -½·R·[a]×·dt²
        #   ∂(R·a)/∂b_a = -R      →  F[P,b_a] = -½·R·dt²
        F[self._P, self._THETA] = -0.5 * R_k @ _skew(a_body) * dt * dt
        F[self._P, self._BA] = -0.5 * R_k * dt * dt
        F[self._V, self._THETA] = -R_k @ _skew(a_body) * dt
        F[self._V, self._BA] = -R_k * dt
        F[self._THETA, self._THETA] = np.eye(3) - _skew(w_body) * dt
        F[self._THETA, self._BG] = -np.eye(3) * dt

        # --- Process noise Q (15×15) ---
        # sigma values are continuous-time noise densities (units/√Hz),
        # so discrete-time covariance = σ² · Δt (not (σ·Δt)²).
        Q = np.zeros((self._DIM_STATE, self._DIM_STATE))
        Q[self._V, self._V] = np.eye(3) * self.sigma_accel ** 2 * dt
        Q[self._THETA, self._THETA] = np.eye(3) * self.sigma_gyro ** 2 * dt
        Q[self._BA, self._BA] = np.eye(3) * self.sigma_accel_bias ** 2 * dt
        Q[self._BG, self._BG] = np.eye(3) * self.sigma_gyro_bias ** 2 * dt

        # --- Covariance propagation ---
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)  # enforce symmetry

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        R_meas: np.ndarray,
    ) -> None:
        """
        Visual measurement update.

        1. Compute innovation (difference between measurement and prediction)
        2. Compute Kalman gain K
        3. Update error state
        4. Apply correction to nominal state
        5. Reset error state to zero
        6. Update covariance (Joseph form)

        Innovation:
            δp = p_meas - p_nominal
            δθ = Log(R_nominal^T · R_meas)   (rotation error as axis-angle)

        Measurement Jacobian H (6×15):
            H = [I₃  0₃  0₃  0₃  0₃]   ← position rows
                [0₃  0₃  I₃  0₃  0₃]   ← orientation rows

        Joseph-form covariance update:
            P⁺ = (I - KH) P⁻ (I - KH)ᵀ + K R Kᵀ

        Parameters
        ----------
        position : (3,) measured position in world frame.
        orientation : (4,) measured orientation quaternion [w,x,y,z].
        R_meas : (6, 6) measurement noise covariance.
        """
        # --- Innovation ---
        dp = position - self.position

        R_nom = _quat_to_rotation(self.quaternion)
        R_vis = _quat_to_rotation(orientation)
        dR = R_nom.T @ R_vis
        dtheta = _quat_to_axis_angle(_rotation_to_quat(dR))

        innovation = np.concatenate([dp, dtheta])

        # --- Measurement Jacobian H (6×15) ---
        H = np.zeros((self._DIM_MEAS, self._DIM_STATE))
        H[0:3, self._P] = np.eye(3)
        H[3:6, self._THETA] = np.eye(3)

        # --- Kalman gain ---
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)

        # --- Error-state update ---
        dx = K @ innovation

        # --- Apply correction to nominal state ---
        self.position = self.position + dx[self._P]
        self.velocity = self.velocity + dx[self._V]

        dq_correction = _axis_angle_to_quat(dx[self._THETA])
        self.quaternion = _quat_multiply(self.quaternion, dq_correction)
        self.quaternion /= np.linalg.norm(self.quaternion)

        self.accel_bias = self.accel_bias + dx[self._BA]
        self.gyro_bias = self.gyro_bias + dx[self._BG]

        # --- Joseph-form covariance update ---
        I_KH = np.eye(self._DIM_STATE) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, NDArray]:
        """Return current state estimate as dict.

        Returns
        -------
        dict with keys ``position``, ``velocity``, ``quaternion``,
        ``accel_bias``, ``gyro_bias``, each a 1-D numpy array.
        """
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "quaternion": self.quaternion.copy(),
            "accel_bias": self.accel_bias.copy(),
            "gyro_bias": self.gyro_bias.copy(),
        }

    def get_covariance(self) -> NDArray:
        """Return 15×15 state covariance matrix."""
        return self.P.copy()


# ======================================================================
#  IMU data simulator
# ======================================================================

def simulate_imu(
    trajectory_fn,
    dt: float = 0.005,
    duration: float = 10.0,
    accel_noise_std: float = 0.1,
    gyro_noise_std: float = 0.01,
    accel_bias_drift: float = 0.001,
    gyro_bias_drift: float = 0.0001,
    gravity: NDArray = np.array([0.0, 0.0, -9.81]),
) -> dict:
    """Generate synthetic IMU measurements from a trajectory function.

    Parameters
    ----------
    trajectory_fn : callable
        ``trajectory_fn(t) -> (position(3,), rotation_matrix(3,3))``
        returning ground-truth pose at time *t*.
    dt : float
        IMU sampling period (seconds).
    duration : float
        Total simulation time (seconds).
    accel_noise_std : float
        Accelerometer white noise σ (m/s²).
    gyro_noise_std : float
        Gyroscope white noise σ (rad/s).
    accel_bias_drift : float
        Accelerometer bias random walk σ.
    gyro_bias_drift : float
        Gyroscope bias random walk σ.
    gravity : (3,) gravity vector in world frame.

    Returns
    -------
    dict with keys:
        timestamps : (N,) time array
        accel_meas : (N, 3) noisy accelerometer readings
        gyro_meas : (N, 3) noisy gyroscope readings
        gt_positions : (N, 3) ground-truth positions
        gt_rotations : list of (3, 3) rotation matrices
        accel_bias_true : (N, 3) true bias trajectory
        gyro_bias_true : (N, 3) true bias trajectory
    """
    N = int(duration / dt) + 1
    timestamps = np.arange(N) * dt

    gt_positions = np.zeros((N, 3))
    gt_rotations = []
    accel_meas = np.zeros((N, 3))
    gyro_meas = np.zeros((N, 3))
    accel_bias = np.zeros((N, 3))
    gyro_bias = np.zeros((N, 3))

    for i, t in enumerate(timestamps):
        pos, R = trajectory_fn(t)
        gt_positions[i] = pos
        gt_rotations.append(R.copy())

        if i > 0:
            accel_bias[i] = accel_bias[i-1] + np.random.normal(0, accel_bias_drift * np.sqrt(dt), 3)
            gyro_bias[i] = gyro_bias[i-1] + np.random.normal(0, gyro_bias_drift * np.sqrt(dt), 3)

    for i in range(1, N - 1):
        v_prev = (gt_positions[i] - gt_positions[i-1]) / dt
        v_next = (gt_positions[i+1] - gt_positions[i]) / dt
        accel_world = (v_next - v_prev) / dt
        R = gt_rotations[i]
        accel_body = R.T @ (accel_world - gravity)
        accel_meas[i] = accel_body + accel_bias[i] + np.random.normal(0, accel_noise_std, 3)

    for i in range(1, N):
        R_prev = gt_rotations[i-1]
        R_curr = gt_rotations[i]
        dR = R_prev.T @ R_curr
        angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if angle > 1e-10:
            axis = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]])
            axis = axis / (2 * np.sin(angle))
            omega = axis * angle / dt
        else:
            omega = np.zeros(3)
        gyro_meas[i] = omega + gyro_bias[i] + np.random.normal(0, gyro_noise_std, 3)

    accel_meas[0] = accel_meas[1]

    return {
        "timestamps": timestamps,
        "accel_meas": accel_meas,
        "gyro_meas": gyro_meas,
        "gt_positions": gt_positions,
        "gt_rotations": gt_rotations,
        "accel_bias_true": accel_bias,
        "gyro_bias_true": gyro_bias,
    }
