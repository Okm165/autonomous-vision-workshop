"""Drone dynamics and VIO EKF tests mirroring the notebook usage patterns."""

import numpy as np

from src.drone import (
    GRAVITY,
    QuadrotorParams,
    QuadrotorState,
    quadrotor_dynamics,
    simulate_step,
    wrench_to_thrusts,
)
from src.ekf import VIO_EKF


def _hover_state(z=1.0):
    s = QuadrotorState()
    s.position = np.array([0.0, 0.0, z])
    return s


class TestQuadrotor:
    def test_hover_mixer(self):
        """Equal thrusts support the weight; dynamics give zero net force."""
        params = QuadrotorParams()
        f_hover = params.mass * GRAVITY / 4.0
        _, vdot, omegadot, _ = quadrotor_dynamics(
            _hover_state(), np.full(4, f_hover), params
        )
        assert np.isclose(vdot[2], 0.0, atol=1e-9)
        assert np.allclose(omegadot, 0.0, atol=1e-12)

    def test_free_fall_at_zero_thrust(self):
        params = QuadrotorParams()
        pdot, vdot, _, _ = quadrotor_dynamics(_hover_state(), np.zeros(4), params)
        assert np.allclose(pdot, 0.0, atol=1e-12)  # starts at rest
        assert np.isclose(vdot[2], -GRAVITY, atol=1e-9)  # free fall

    def test_simulate_step_preserves_hover(self):
        params = QuadrotorParams()
        state = _hover_state()
        thrusts = np.full(4, params.mass * GRAVITY / 4.0)
        nxt = simulate_step(state, thrusts, params, dt=0.01)
        assert np.allclose(nxt.position, state.position, atol=1e-6)
        assert np.allclose(nxt.velocity, state.velocity, atol=1e-6)
        assert np.allclose(nxt.rotation, np.eye(3), atol=1e-8)

    def test_wrench_to_thrusts_hover(self):
        params = QuadrotorParams()
        thrusts = wrench_to_thrusts(params.mass * GRAVITY, np.zeros(3), params)
        assert (thrusts >= 0).all()
        assert np.allclose(thrusts, params.mass * GRAVITY / 4.0, rtol=1e-6)


class TestVIOEKF:
    """Error-state EKF on a simulated circle trajectory, as in NB13."""

    def _circle(self, n=200, dt=0.05):
        t = np.arange(n) * dt
        pos = np.column_stack([2.0 * np.cos(t), 2.0 * np.sin(t), np.ones_like(t)])
        vel = np.column_stack([-2.0 * np.sin(t), 2.0 * np.cos(t), np.zeros_like(t)])
        acc_world = np.gradient(vel, t, axis=0)
        return t, pos, vel, acc_world

    def test_position_error_stays_bounded(self):
        rng = np.random.default_rng(0)
        t, pos, vel, acc_world = self._circle()
        ekf = VIO_EKF()
        ekf.position = pos[0].copy()
        ekf.velocity = vel[0].copy()
        # identity attitude: gravity compensation happens in the body frame
        R_meas = np.diag([1e-4] * 3 + [1e-6] * 3)
        for k in range(1, len(t)):
            dt = t[k] - t[k - 1]
            accel_body = acc_world[k - 1] + np.array([0, 0, GRAVITY])
            ekf.predict(
                accel_body + rng.normal(0, 1e-3, 3),
                np.array([0.0, 0.0, 1.0]) + rng.normal(0, 1e-4, 3),
                dt,
            )
            if k % 5 == 0:  # periodic VO updates
                ekf.update(
                    pos[k] + rng.normal(0, 1e-2, 3),
                    ekf.quaternion.copy(),
                    R_meas,
                )
        err = np.linalg.norm(ekf.get_state()["position"] - pos[-1])
        assert err < 0.5, f"final position error {err:.3f} m"
