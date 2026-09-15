r"""Quadrotor dynamics, control, and trajectory generation.

Provides a complete, pure-NumPy simulation and control stack for a
quadrotor UAV in the "+" motor configuration:

* **Dynamics** — Newton–Euler rigid-body equations with gyroscopic torque
  and per-motor thrust/drag, integrated via RK4 on SO(3).
* **Motor mixing** — wrench ↔ individual thrust conversion.
* **PID controller** — cascaded position → attitude → body-rate loops.
* **SE(3) geometric controller** — Lyapunov-stable tracking on the full
  pose manifold (Lee, Leok & McClamroch, CDC 2010).
* **Trajectory generators** — hover, circle, figure-8, waypoint sequences.

Convention: world frame is NED with *z*-up; body frame has *x* forward,
*y* right, *z* up.

References
----------
[1] Lee, Leok, McClamroch, "Geometric Tracking Control of a Quadrotor UAV
    on SE(3)", CDC 2010.
[2] Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control for
    Quadrotors", ICRA 2011.
[3] Bouabdallah, "Design and Control of Quadrotors with Application to
    Autonomous Flying", EPFL PhD Thesis, 2007.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict


# ======================================================================
#  Constants
# ======================================================================

GRAVITY = 9.81  # m/s²

MAX_VELOCITY = 50.0        # m/s — clamp to prevent divergence
MAX_ANGULAR_VELOCITY = 100.0  # rad/s — clamp to prevent divergence
INTEGRAL_CLAMP = 2.0       # integral error saturation limit


# ======================================================================
#  Quadrotor Parameters
# ======================================================================

class QuadrotorParams:
    r"""Physical parameters for a quadrotor UAV.

    Both body and world frames use the robotics convention (z-up).
    Thrust is generated along the body +z axis.

    Standard "+" configuration::

            1 (front)
            |
        4 --+-- 2
            |
            3 (back)

    Each motor :math:`i` produces thrust :math:`f_i = k_f \omega_i^2`
    and drag torque :math:`\tau_i = k_\tau \omega_i^2`.

    Parameters
    ----------
    mass : float
        Total mass [kg]
    arm_length : float
        Distance from CoM to motor [m]
    I : (3,3) ndarray
        Inertia tensor in body frame [kg·m²]
    k_f : float
        Thrust coefficient [N/(rad/s)²]
    k_tau : float
        Drag torque coefficient [N·m/(rad/s)²]
    max_rpm : float
        Maximum motor RPM
    """

    def __init__(
        self,
        mass: float = 0.5,
        arm_length: float = 0.17,
        I_xx: float = 2.32e-3,
        I_yy: float = 2.32e-3,
        I_zz: float = 4.00e-3,
        k_f: float = 1.0e-5,
        k_tau: float = 1.0e-7,
        max_rpm: float = 25000.0,
    ):
        """Initialise quadrotor physical parameters and compute the mixer matrix."""
        self.mass = mass
        self.arm_length = arm_length
        self.I = np.diag([I_xx, I_yy, I_zz])
        self.I_inv = np.diag([1.0 / I_xx, 1.0 / I_yy, 1.0 / I_zz])
        self.k_f = k_f
        self.k_tau = k_tau
        self.max_rpm = max_rpm
        self.max_thrust_per_motor = k_f * (max_rpm * 2 * np.pi / 60) ** 2

        # Mixer matrix: maps [f1, f2, f3, f4] -> [F_total, tau_x, tau_y, tau_z]
        L = arm_length
        gamma = k_tau / k_f  # drag-to-thrust ratio
        self.mixer = np.array([
            [1.0,     1.0,    1.0,    1.0   ],
            [0.0,     L,      0.0,   -L     ],
            [-L,      0.0,    L,      0.0   ],
            [gamma,  -gamma,  gamma, -gamma ],
        ])
        self.mixer_inv = np.linalg.inv(self.mixer)

    @property
    def hover_rpm(self) -> float:
        """RPM required for each motor to hover."""
        f_per_motor = self.mass * GRAVITY / 4.0
        omega = np.sqrt(f_per_motor / self.k_f)
        return omega * 60 / (2 * np.pi)


# ======================================================================
#  Quadrotor State
# ======================================================================

class QuadrotorState:
    r"""Full 12-DOF state of a rigid-body quadrotor.

    The state vector is :math:`\mathbf{x} = [p, v, R, \omega]` where:

    - :math:`p \in \mathbb{R}^3` — position in world frame
    - :math:`v \in \mathbb{R}^3` — velocity in world frame
    - :math:`R \in SO(3)` — rotation from body to world
    - :math:`\omega \in \mathbb{R}^3` — angular velocity in body frame
    """

    def __init__(self) -> None:
        """Initialise the state to hover at the origin with zero velocity."""
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.rotation = np.eye(3)
        self.omega = np.zeros(3)

    def copy(self) -> "QuadrotorState":
        """Return a deep copy of this state."""
        s = QuadrotorState()
        s.position = self.position.copy()
        s.velocity = self.velocity.copy()
        s.rotation = self.rotation.copy()
        s.omega = self.omega.copy()
        return s

    @property
    def euler_angles(self) -> NDArray[np.float64]:
        """Extract ZYX Euler angles (roll, pitch, yaw) from rotation matrix."""
        R = self.rotation
        pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
        if np.abs(np.cos(pitch)) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0.0
        return np.array([roll, pitch, yaw])

    def to_vector(self) -> NDArray[np.float64]:
        """Flatten to a 18-element vector [pos(3), vel(3), R_flat(9), omega(3)]."""
        return np.concatenate([
            self.position, self.velocity,
            self.rotation.flatten(), self.omega
        ])

    @staticmethod
    def from_vector(x: NDArray[np.float64]) -> "QuadrotorState":
        """Reconstruct a state from an 18-element vector.

        Parameters
        ----------
        x : (18,) array — [pos(3), vel(3), R_flat(9), omega(3)].
        """
        s = QuadrotorState()
        s.position = x[0:3].copy()
        s.velocity = x[3:6].copy()
        s.rotation = x[6:15].reshape(3, 3).copy()
        s.omega = x[15:18].copy()
        return s


def hat(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Skew-symmetric (hat) map: R^3 -> so(3)."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def vee(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of hat map: so(3) -> R^3."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


# ======================================================================
#  Quadrotor Dynamics
# ======================================================================

def quadrotor_dynamics(
    state: QuadrotorState,
    thrusts: NDArray[np.float64],
    params: QuadrotorParams,
    wind: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute state derivatives from Newton-Euler equations.

    **Translational dynamics** (world frame):

    .. math::

        m \dot{v} = -mg \mathbf{e}_3 + R \, F_{\text{total}} \, \mathbf{e}_3 + \mathbf{f}_{\text{ext}}

    where :math:`F_{\text{total}} = \sum_i f_i` and :math:`\mathbf{e}_3 = [0, 0, 1]^T`.

    **Rotational dynamics** (body frame, Euler's equation):

    .. math::

        I \dot{\omega} = -\omega \times (I \omega) + \boldsymbol{\tau}

    where :math:`\boldsymbol{\tau} = [L(f_2 - f_4), \; L(f_3 - f_1), \; \gamma(f_1 - f_2 + f_3 - f_4)]`.

    **Kinematics**:

    .. math::

        \dot{R} = R [\omega]_\times, \qquad \dot{p} = v

    Parameters
    ----------
    state : QuadrotorState
    thrusts : (4,) per-motor thrust forces [N]
    params : QuadrotorParams
    wind : (3,) optional external force in world frame [N]

    Returns
    -------
    dp, dv, dR, domega : state derivatives
    """
    R = state.rotation
    omega = state.omega

    # Wrench from mixer
    wrench = params.mixer @ thrusts  # [F_total, tau_x, tau_y, tau_z]
    F_total = wrench[0]
    tau_body = wrench[1:4]

    # Translational dynamics
    gravity_world = np.array([0, 0, -GRAVITY])
    thrust_world = R @ np.array([0, 0, F_total])
    f_ext = wind if wind is not None else np.zeros(3)
    dv = gravity_world + thrust_world / params.mass + f_ext / params.mass

    # Rotational dynamics (Euler's equation)
    domega = params.I_inv @ (tau_body - np.cross(omega, params.I @ omega))

    # Kinematics
    dp = state.velocity.copy()
    dR = R @ hat(omega)

    return dp, dv, dR, domega


def simulate_step(
    state: QuadrotorState,
    thrusts: NDArray[np.float64],
    params: QuadrotorParams,
    dt: float,
    wind: Optional[NDArray[np.float64]] = None,
    integrator: str = "rk4",
) -> QuadrotorState:
    r"""Advance state by one timestep using RK4 or Euler integration.

    RK4 on SO(3) uses the exponential map:
    :math:`R_{k+1} = R_k \exp([\omega \, dt]_\times)`.

    After integration, the rotation matrix is re-orthogonalized via SVD
    to prevent numerical drift from SO(3).

    Parameters
    ----------
    state : current state
    thrusts : (4,) motor thrusts [N]
    params : QuadrotorParams
    dt : timestep [s]
    wind : external force [N]
    integrator : "euler" or "rk4"

    Returns
    -------
    QuadrotorState : new state
    """
    thrusts = np.clip(thrusts, 0, params.max_thrust_per_motor)

    if integrator == "euler":
        dp, dv, dR, dw = quadrotor_dynamics(state, thrusts, params, wind)
        new = state.copy()
        new.position += dp * dt
        new.velocity += dv * dt
        new.rotation = state.rotation @ _exp_so3(state.omega * dt)
        new.omega += dw * dt
    elif integrator == "rk4":
        new = _rk4_step(state, thrusts, params, dt, wind)
    else:
        raise ValueError(f"Unknown integrator: {integrator}")

    # Re-orthogonalize R via SVD projection onto SO(3)
    try:
        U, _, Vt = np.linalg.svd(new.rotation)
        det_sign = np.linalg.det(U @ Vt)
        new.rotation = U @ np.diag([1, 1, det_sign]) @ Vt
    except np.linalg.LinAlgError:
        new.rotation = np.eye(3)
        new.omega = np.zeros(3)

    v_norm = np.linalg.norm(new.velocity)
    if v_norm > MAX_VELOCITY:
        new.velocity *= MAX_VELOCITY / v_norm
    w_norm = np.linalg.norm(new.omega)
    if w_norm > MAX_ANGULAR_VELOCITY:
        new.omega *= MAX_ANGULAR_VELOCITY / w_norm

    return new


def _exp_so3(omega_dt: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rodrigues formula for exp: so(3) -> SO(3)."""
    theta = np.linalg.norm(omega_dt)
    if theta < 1e-10:
        return np.eye(3) + hat(omega_dt)
    K = hat(omega_dt / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def _rk4_step(
    state: QuadrotorState,
    thrusts: NDArray[np.float64],
    params: QuadrotorParams,
    dt: float,
    wind: Optional[NDArray[np.float64]],
) -> QuadrotorState:
    """4th-order Runge-Kutta integrator."""
    def derivatives(
        s: QuadrotorState,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64],
               NDArray[np.float64], NDArray[np.float64]]:
        """Compute (dp, dv, omega, domega) for the RK4 sub-step."""
        dp, dv, _, dw = quadrotor_dynamics(s, thrusts, params, wind)
        return dp, dv, s.omega.copy(), dw

    dp1, dv1, w1, dw1 = derivatives(state)

    s2 = state.copy()
    s2.position = state.position + 0.5 * dt * dp1
    s2.velocity = state.velocity + 0.5 * dt * dv1
    s2.rotation = state.rotation @ _exp_so3(0.5 * dt * w1)
    s2.omega = state.omega + 0.5 * dt * dw1
    dp2, dv2, w2, dw2 = derivatives(s2)

    s3 = state.copy()
    s3.position = state.position + 0.5 * dt * dp2
    s3.velocity = state.velocity + 0.5 * dt * dv2
    s3.rotation = state.rotation @ _exp_so3(0.5 * dt * w2)
    s3.omega = state.omega + 0.5 * dt * dw2
    dp3, dv3, w3, dw3 = derivatives(s3)

    s4 = state.copy()
    s4.position = state.position + dt * dp3
    s4.velocity = state.velocity + dt * dv3
    s4.rotation = state.rotation @ _exp_so3(dt * w3)
    s4.omega = state.omega + dt * dw3
    dp4, dv4, w4, dw4 = derivatives(s4)

    new = state.copy()
    new.position += dt / 6 * (dp1 + 2*dp2 + 2*dp3 + dp4)
    new.velocity += dt / 6 * (dv1 + 2*dv2 + 2*dv3 + dv4)
    w_avg = (w1 + 2*w2 + 2*w3 + w4) / 6
    new.rotation = state.rotation @ _exp_so3(dt * w_avg)
    new.omega += dt / 6 * (dw1 + 2*dw2 + 2*dw3 + dw4)

    return new


# ======================================================================
#  Motor Mixing
# ======================================================================

def wrench_to_thrusts(
    F_total: float,
    torques: NDArray[np.float64],
    params: QuadrotorParams,
) -> NDArray[np.float64]:
    r"""Convert desired wrench to individual motor thrusts.

    Solves :math:`M \mathbf{f} = \mathbf{w}` where :math:`M` is the
    mixer matrix and :math:`\mathbf{w} = [F, \tau_x, \tau_y, \tau_z]^T`.

    Clamps negative thrusts to zero and caps at max thrust.
    """
    wrench = np.array([F_total, torques[0], torques[1], torques[2]])
    thrusts = params.mixer_inv @ wrench
    return np.clip(thrusts, 0, params.max_thrust_per_motor)


# ======================================================================
#  PID Controller
# ======================================================================

class PIDController:
    r"""Cascaded PID controller for quadrotor position tracking.

    **Outer loop** (position):
    computes desired acceleration from position/velocity error.

    .. math::

        \mathbf{a}_{\text{des}} = K_p (\mathbf{p}_d - \mathbf{p})
        + K_d (\mathbf{v}_d - \mathbf{v})
        + K_i \int (\mathbf{p}_d - \mathbf{p}) \, dt
        + \mathbf{a}_{\text{ff}} + g \mathbf{e}_3

    **Inner loop** (attitude):
    extracts desired thrust and orientation from :math:`\mathbf{a}_{\text{des}}`,
    then computes body torques from attitude error.

    .. math::

        F = m \, \mathbf{a}_{\text{des}} \cdot \mathbf{z}_B

    where :math:`\mathbf{z}_B = R \mathbf{e}_3` is the body z-axis.
    """

    def __init__(
        self,
        Kp: NDArray[np.float64] = np.array([4.0, 4.0, 6.0]),
        Kd: NDArray[np.float64] = np.array([3.5, 3.5, 4.5]),
        Ki: NDArray[np.float64] = np.array([0.1, 0.1, 0.2]),
        Kp_att: NDArray[np.float64] = np.array([0.55, 0.55, 0.25]),
        Kd_att: NDArray[np.float64] = np.array([0.072, 0.072, 0.04]),
        max_tilt: float = np.radians(45),
    ):
        """Initialise PID gains for position and attitude control loops."""
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.Kp_att = Kp_att
        self.Kd_att = Kd_att
        self.max_tilt = max_tilt
        self.int_error = np.zeros(3)

    def compute(
        self,
        state: QuadrotorState,
        target_pos: NDArray[np.float64],
        target_vel: NDArray[np.float64] = np.zeros(3),
        target_acc: NDArray[np.float64] = np.zeros(3),
        target_yaw: float = 0.0,
        params: QuadrotorParams = QuadrotorParams(),
        dt: float = 0.01,
    ) -> NDArray[np.float64]:
        r"""Compute motor thrusts for trajectory tracking.

        Returns (4,) array of motor thrusts.
        """
        # Position error
        e_pos = target_pos - state.position
        e_vel = target_vel - state.velocity
        self.int_error += e_pos * dt
        self.int_error = np.clip(self.int_error, -INTEGRAL_CLAMP, INTEGRAL_CLAMP)

        # Desired acceleration (world frame)
        a_des = (self.Kp * e_pos + self.Kd * e_vel + self.Ki * self.int_error
                 + target_acc + np.array([0, 0, GRAVITY]))

        # --- Thrust magnitude ---
        z_body = state.rotation[:, 2]
        F_total = params.mass * np.dot(a_des, z_body)

        # --- Desired orientation from desired acceleration ---
        z_des = a_des / (np.linalg.norm(a_des) + 1e-8)
        # Limit tilt angle
        tilt = np.arccos(np.clip(z_des[2], -1, 1))
        if tilt > self.max_tilt:
            z_des_xy = z_des[:2]
            z_des_xy_norm = np.linalg.norm(z_des_xy)
            if z_des_xy_norm > 1e-8:
                z_des[:2] = z_des_xy / z_des_xy_norm * np.sin(self.max_tilt)
                z_des[2] = np.cos(self.max_tilt)
            z_des = z_des / (np.linalg.norm(z_des) + 1e-8)

        # Desired yaw direction
        x_c = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0])
        y_des = np.cross(z_des, x_c)
        y_des_norm = np.linalg.norm(y_des)
        if y_des_norm > 1e-8:
            y_des /= y_des_norm
        else:
            y_des = np.array([0, 1, 0])
        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack([x_des, y_des, z_des])

        # --- Attitude error (Lee et al. 2010, Eq. 9) ---
        e_R_mat = 0.5 * (R_des.T @ state.rotation - state.rotation.T @ R_des)
        e_R = vee(e_R_mat)
        e_omega = state.omega  # desired omega ~ 0 for position control

        # Body torques
        tau = -self.Kp_att * e_R - self.Kd_att * e_omega

        return wrench_to_thrusts(F_total, tau, params)


# ======================================================================
#  SE(3) Geometric Controller (Lee et al. 2010)
# ======================================================================

class SE3Controller:
    r"""Geometric tracking controller on SE(3).

    From Lee, Leok, McClamroch (CDC 2010):

    .. math::

        \mathbf{e}_R = \frac{1}{2} (R_d^T R - R^T R_d)^\vee

    .. math::

        \mathbf{e}_\omega = \omega - R^T R_d \omega_d

    .. math::

        \boldsymbol{\tau} = -k_R \mathbf{e}_R - k_\omega \mathbf{e}_\omega
        + \omega \times J \omega
        - J (\hat{\omega} R^T R_d \omega_d - R^T R_d \dot{\omega}_d)

    This controller has almost-global asymptotic stability on SO(3).
    """

    def __init__(
        self,
        kx: float = 4.0,
        kv: float = 3.5,
        kR: float = 0.55,
        kw: float = 0.072,
    ):
        """Initialise SE(3) controller gains for position and attitude."""
        self.kx = kx
        self.kv = kv
        self.kR = kR
        self.kw = kw

    def compute(
        self,
        state: QuadrotorState,
        target_pos: NDArray[np.float64],
        target_vel: NDArray[np.float64],
        target_acc: NDArray[np.float64],
        target_yaw: float,
        params: QuadrotorParams,
    ) -> NDArray[np.float64]:
        """Compute motor thrusts using SE(3) geometric control."""
        R = state.rotation
        m = params.mass
        e3 = np.array([0, 0, 1.0])

        # Position/velocity errors
        e_p = state.position - target_pos
        e_v = state.velocity - target_vel

        # Desired force vector (world frame)
        F_des = (-self.kx * e_p - self.kv * e_v
                 + m * GRAVITY * e3 + m * target_acc)

        # Desired body z-axis
        z_des = F_des / (np.linalg.norm(F_des) + 1e-8)

        # Desired rotation from yaw + z_des
        x_c = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0])
        y_des = np.cross(z_des, x_c)
        y_norm = np.linalg.norm(y_des)
        if y_norm > 1e-8:
            y_des /= y_norm
        else:
            y_des = np.array([0, 1, 0])
        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack([x_des, y_des, z_des])

        # Thrust magnitude
        F_total = np.dot(F_des, R[:, 2])

        # Attitude error
        e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = vee(e_R_mat)
        e_omega = state.omega  # desired omega ≈ 0

        # Torque (Lee et al. Eq. 12)
        I = params.I
        tau = (-self.kR * e_R - self.kw * e_omega
               + np.cross(state.omega, I @ state.omega))

        return wrench_to_thrusts(F_total, tau, params)


# ======================================================================
#  Trajectory Generation (Minimum Snap)
# ======================================================================

def generate_hover_trajectory(
    center: NDArray[np.float64],
    duration: float,
    dt: float,
) -> Dict[str, NDArray[np.float64]]:
    """Generate a stationary hover trajectory."""
    N = int(duration / dt)
    t = np.arange(N) * dt
    return {
        "t": t,
        "pos": np.tile(center, (N, 1)),
        "vel": np.zeros((N, 3)),
        "acc": np.zeros((N, 3)),
        "yaw": np.zeros(N),
    }


def generate_circle_trajectory(
    center: NDArray[np.float64],
    radius: float,
    height: float,
    period: float,
    duration: float,
    dt: float,
) -> Dict[str, NDArray[np.float64]]:
    r"""Generate a circular trajectory with analytic derivatives.

    .. math::

        p(t) = c + \begin{bmatrix}
        r \cos(\omega t) \\ r \sin(\omega t) \\ h
        \end{bmatrix}, \quad \omega = \frac{2\pi}{T}
    """
    N = int(duration / dt)
    t = np.arange(N) * dt
    omega = 2 * np.pi / period

    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))
    acc = np.zeros((N, 3))
    yaw = np.zeros(N)

    pos[:, 0] = center[0] + radius * np.cos(omega * t)
    pos[:, 1] = center[1] + radius * np.sin(omega * t)
    pos[:, 2] = center[2] + height

    vel[:, 0] = -radius * omega * np.sin(omega * t)
    vel[:, 1] = radius * omega * np.cos(omega * t)

    acc[:, 0] = -radius * omega**2 * np.cos(omega * t)
    acc[:, 1] = -radius * omega**2 * np.sin(omega * t)

    yaw = np.arctan2(vel[:, 1], vel[:, 0])

    return {"t": t, "pos": pos, "vel": vel, "acc": acc, "yaw": yaw}


def generate_figure8_trajectory(
    center: NDArray[np.float64],
    scale: float,
    period: float,
    duration: float,
    dt: float,
) -> Dict[str, NDArray[np.float64]]:
    r"""Lissajous figure-8 trajectory.

    .. math::

        p(t) = c + \begin{bmatrix}
        A \sin(\omega t) \\ A \sin(2\omega t) \\ 0
        \end{bmatrix}
    """
    N = int(duration / dt)
    t = np.arange(N) * dt
    omega = 2 * np.pi / period

    pos = np.zeros((N, 3))
    vel = np.zeros((N, 3))
    acc = np.zeros((N, 3))

    pos[:, 0] = center[0] + scale * np.sin(omega * t)
    pos[:, 1] = center[1] + scale * np.sin(2 * omega * t)
    pos[:, 2] = center[2]

    vel[:, 0] = scale * omega * np.cos(omega * t)
    vel[:, 1] = scale * 2 * omega * np.cos(2 * omega * t)

    acc[:, 0] = -scale * omega**2 * np.sin(omega * t)
    acc[:, 1] = -scale * 4 * omega**2 * np.sin(2 * omega * t)

    yaw = np.arctan2(vel[:, 1], vel[:, 0])

    return {"t": t, "pos": pos, "vel": vel, "acc": acc, "yaw": yaw}


def generate_waypoint_trajectory(
    waypoints: NDArray[np.float64],
    speed: float,
    dt: float,
) -> Dict[str, NDArray[np.float64]]:
    """Generate a piecewise-linear trajectory through waypoints.

    Parameters
    ----------
    waypoints : (M, 3) array of 3D waypoints
    speed : desired cruise speed [m/s]
    dt : timestep [s]
    """
    segments = np.diff(waypoints, axis=0)
    lengths = np.linalg.norm(segments, axis=1)

    positions = []
    velocities = []
    times = []

    t_current = 0.0
    for i, (seg, length) in enumerate(zip(segments, lengths)):
        seg_time = length / speed
        direction = seg / (length + 1e-12)
        n_steps = max(1, int(seg_time / dt))

        for j in range(n_steps):
            frac = j / n_steps
            pos = waypoints[i] + frac * seg
            vel = direction * speed
            positions.append(pos)
            velocities.append(vel)
            times.append(t_current + frac * seg_time)

        t_current += seg_time

    positions.append(waypoints[-1])
    velocities.append(np.zeros(3))
    times.append(t_current)

    pos_arr = np.array(positions)
    vel_arr = np.array(velocities)
    t_arr = np.array(times)
    acc_arr = np.zeros_like(pos_arr)
    yaw_arr = np.arctan2(vel_arr[:, 1], vel_arr[:, 0])

    return {"t": t_arr, "pos": pos_arr, "vel": vel_arr, "acc": acc_arr, "yaw": yaw_arr}


# ======================================================================
#  Simulation Utilities
# ======================================================================

def simulate_trajectory(
    trajectory: Dict[str, NDArray[np.float64]],
    controller,
    params: QuadrotorParams,
    dt: float = 0.005,
    wind_func=None,
) -> Dict[str, NDArray[np.float64]]:
    r"""Run closed-loop simulation of a quadrotor tracking a trajectory.

    Returns dictionary with time, actual positions, errors, thrusts.
    """
    traj_t = trajectory["t"]
    traj_pos = trajectory["pos"]
    traj_vel = trajectory["vel"]
    traj_acc = trajectory["acc"]
    traj_yaw = trajectory["yaw"]

    state = QuadrotorState()
    state.position = traj_pos[0].copy()

    N = len(traj_t)
    actual_pos = np.zeros((N, 3))
    actual_vel = np.zeros((N, 3))
    actual_euler = np.zeros((N, 3))
    actual_omega = np.zeros((N, 3))
    rotations = np.zeros((N, 3, 3))
    thrust_log = np.zeros((N, 4))
    tracking_error = np.zeros(N)

    for i in range(N):
        actual_pos[i] = state.position
        actual_vel[i] = state.velocity
        actual_euler[i] = state.euler_angles
        actual_omega[i] = state.omega
        rotations[i] = state.rotation
        tracking_error[i] = np.linalg.norm(state.position - traj_pos[i])

        wind = wind_func(traj_t[i]) if wind_func else None

        if hasattr(controller, 'compute'):
            if isinstance(controller, SE3Controller):
                thrusts = controller.compute(
                    state, traj_pos[i], traj_vel[i], traj_acc[i],
                    traj_yaw[i], params)
            else:
                thrusts = controller.compute(
                    state, traj_pos[i], traj_vel[i], traj_acc[i],
                    traj_yaw[i], params, dt)
        else:
            thrusts = controller(state, traj_pos[i], traj_vel[i],
                                 traj_acc[i], traj_yaw[i], params, dt)

        thrust_log[i] = thrusts
        state = simulate_step(state, thrusts, params, dt, wind, integrator="rk4")

    return {
        "t": traj_t,
        "actual_pos": actual_pos,
        "actual_vel": actual_vel,
        "actual_euler": actual_euler,
        "actual_omega": actual_omega,
        "rotations": rotations,
        "desired_pos": traj_pos,
        "thrusts": thrust_log,
        "tracking_error": tracking_error,
    }
