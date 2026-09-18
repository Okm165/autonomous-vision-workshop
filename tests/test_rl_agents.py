"""Tests for src/rl_agents.py — buffers, GAE, policies, PPO, SAC, drone envs.

The GAE recursion is checked against hand-computed values and against the
Monte-Carlo / one-step-TD limits (lambda = 1 and lambda = 0).
"""

from typing import Any

import numpy as np
import pytest
from scipy.stats import norm

from src.drone import QuadrotorState
from src.rl_agents import (
    PPO,
    SAC,
    DroneHoverEnv,
    DroneNavigationEnv,
    MLPPolicy,
    MLPValueFunction,
    ReplayBuffer,
    RolloutBuffer,
)

# Tiny networks keep the ES-style numerical updates fast.


class TestReplayBuffer:
    def test_roundtrip(self):
        buf = ReplayBuffer(capacity=5, obs_dim=3, act_dim=2)
        rng = np.random.default_rng(0)
        obs = rng.normal(size=(5, 3))
        act = rng.normal(size=(5, 2))
        rew = np.arange(5.0)
        nxt = rng.normal(size=(5, 3))
        dones = np.array([False, False, True, False, False])
        for i in range(5):
            buf.add(obs[i], act[i], rew[i], nxt[i], dones[i])

        assert buf.size == 5
        assert buf.ptr == 0
        assert np.allclose(buf.obs, obs)
        assert np.allclose(buf.rewards, rew)
        assert list(buf.dones) == list(dones)

    def test_wraparound_keeps_latest(self):
        buf = ReplayBuffer(capacity=3, obs_dim=1, act_dim=1)
        for i in range(5):  # overflow by 2
            buf.add(
                np.array([float(i)]),
                np.array([0.0]),
                float(i),
                np.array([float(i) + 1]),
                False,
            )
        assert buf.size == 3
        # oldest entries (i=0,1) must have been overwritten by i=3,4
        assert sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0]

    def test_sample_shapes_and_values(self):
        buf = ReplayBuffer(capacity=10, obs_dim=3, act_dim=2)
        rng = np.random.default_rng(1)
        for i in range(10):
            buf.add(
                rng.normal(size=3),
                rng.normal(size=2),
                float(i),
                rng.normal(size=3),
                i == 4,
            )
        batch = buf.sample(16, np.random.RandomState(7))
        assert batch["obs"].shape == (16, 3)
        assert batch["actions"].shape == (16, 2)
        assert batch["rewards"].shape == (16,)
        assert batch["next_obs"].shape == (16, 3)
        assert batch["dones"].shape == (16,)
        assert batch["dones"].dtype == bool
        # every sampled row must exist in the buffer
        for i in range(16):
            d = np.linalg.norm(buf.obs - batch["obs"][i], axis=1)
            assert d.min() < 1e-12

    def test_sample_deterministic_with_seed(self):
        buf = ReplayBuffer(capacity=8, obs_dim=2, act_dim=1)
        rng = np.random.default_rng(2)
        for _ in range(8):
            buf.add(
                rng.normal(size=2), rng.normal(size=1), 0.0, rng.normal(size=2), False
            )
        b1 = buf.sample(4, np.random.RandomState(123))
        b2 = buf.sample(4, np.random.RandomState(123))
        assert np.array_equal(b1["obs"], b2["obs"])
        assert np.array_equal(b1["rewards"], b2["rewards"])


class TestRolloutBufferGAE:
    def _fill(self, buf, rewards, dones, values):
        for o, a, r, d, lp, v in zip(
            [np.zeros(2)] * len(rewards),
            [np.zeros(1)] * len(rewards),
            rewards,
            dones,
            [0.0] * len(rewards),
            values,
            strict=False,
        ):
            buf.add(o, a, r, d, lp, v)

    def test_gae_hand_computed(self):
        """gamma=0.9, lam=0.8, terminal step at t=2 (bootstrap ignored)."""
        buf = RolloutBuffer()
        self._fill(
            buf, rewards=[1.0, 2.0, 3.0], dones=[0, 0, 1], values=[0.5, 0.5, 0.5]
        )
        buf.compute_returns_and_advantages(last_value=100.0, gamma=0.9, lam=0.8)

        # deltas: d2 = 3 - 0.5 (mask kills bootstrap) = 2.5
        #         d1 = 2 + 0.9*0.5 - 0.5 = 1.95
        #         d0 = 1 + 0.9*0.5 - 0.5 = 0.95
        # A2 = 2.5; A1 = 1.95 + 0.72*2.5 = 3.75; A0 = 0.95 + 0.72*3.75 = 3.65
        assert np.allclose(buf.advantages, [3.65, 3.75, 2.5], atol=1e-12)
        assert np.allclose(buf.returns, [4.15, 4.25, 3.0], atol=1e-12)

    def test_gae_lambda1_is_monte_carlo(self):
        """With lam=1 advantages are discounted MC returns minus V(s_t)."""
        buf = RolloutBuffer()
        rewards, values = [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]
        self._fill(buf, rewards, [0, 0, 1], values)
        buf.compute_returns_and_advantages(10.0, gamma=0.9, lam=1.0)

        mc = np.zeros(3)
        g = 0.0
        for t in reversed(range(3)):  # episode ends at t=2
            g = rewards[t] + 0.9 * g
            mc[t] = g
        assert np.allclose(buf.returns, mc, atol=1e-12)
        assert np.allclose(buf.advantages, mc - np.array(values), atol=1e-12)

    def test_gae_lambda0_is_one_step_td(self):
        buf = RolloutBuffer()
        self._fill(buf, [1.0, 2.0, 3.0], [0, 0, 0], [0.5, 0.5, 0.5])
        buf.compute_returns_and_advantages(last_value=10.0, gamma=0.9, lam=0.0)
        # A_t = r_t + gamma * V_{t+1} - V_t  (V_3 -> last_value = 10)
        expected_a = [
            1.0 + 0.9 * 0.5 - 0.5,
            2.0 + 0.9 * 0.5 - 0.5,
            3.0 + 0.9 * 10.0 - 0.5,
        ]
        assert np.allclose(buf.advantages, expected_a, atol=1e-12)

    def test_mid_episode_done_resets_accumulation(self):
        """A done in the middle must cut the advantage bootstrapping."""
        buf = RolloutBuffer()
        #        t0        t1(done)  t2
        self._fill(buf, [1.0, 1.0, 1.0], [0, 1, 0], [0.0, 0.0, 0.0])
        buf.compute_returns_and_advantages(last_value=5.0, gamma=1.0, lam=1.0)
        # A1 = 1 (done -> no bootstrap), A0 = 1 + 1*1 = 2, A2 = 1 + 5 = 6
        assert np.allclose(buf.advantages, [2.0, 1.0, 6.0], atol=1e-12)

    def test_get_batches_covers_all_indices_once_per_epoch(self):
        buf = RolloutBuffer()
        self._fill(buf, [0.0] * 10, [0] * 10, [0.0] * 10)
        buf.compute_returns_and_advantages(0.0)
        seen = []
        for batch in buf.get_batches(4, np.random.RandomState(0)):
            assert batch["obs"].shape[1] == 2
            seen.extend(batch["returns"].tolist())
        assert sorted(seen) == [0.0] * 10  # permutation covers everything once

    def test_clear(self):
        buf = RolloutBuffer()
        self._fill(buf, [1.0], [0], [0.5])
        buf.compute_returns_and_advantages(0.0)
        buf.clear()
        assert len(buf.obs) == 0
        assert len(buf.returns) == 0


class TestMLPPolicy:
    def test_forward_shapes(self):
        pol = MLPPolicy(obs_dim=4, act_dim=3, hidden_sizes=(8, 8), seed=0)
        mean, log_std = pol.forward(np.zeros((5, 4)))
        assert mean.shape == (5, 3)
        assert log_std.shape == (3,)  # state-independent std, broadcasts

    def test_sample_action_deterministic_for_seed(self):
        obs = np.linspace(-1, 1, 4)
        p1 = MLPPolicy(4, 2, hidden_sizes=(8,), seed=123)
        p2 = MLPPolicy(4, 2, hidden_sizes=(8,), seed=123)
        a1, lp1 = p1.sample_action(obs)
        a2, lp2 = p2.sample_action(obs)
        assert np.array_equal(a1, a2)
        assert lp1 == lp2

    def test_log_prob_matches_scipy_gaussian(self):
        pol = MLPPolicy(obs_dim=3, act_dim=2, hidden_sizes=(8,), seed=5)
        obs = np.array([0.1, -0.2, 0.3])
        action, log_prob = pol.sample_action(obs)
        mean, log_std = pol.forward(obs)
        expected = norm.logpdf(action, loc=mean, scale=np.exp(log_std)).sum()
        assert np.isclose(log_prob, expected, atol=1e-10)

    def test_params_roundtrip(self):
        pol = MLPPolicy(3, 2, hidden_sizes=(6, 6), seed=9)
        params = [p.copy() for p in pol.get_params()]
        obs = np.ones(3)
        mean_before = pol.forward(obs)[0]
        # scramble weights, then restore (forward() is noise-free)
        scrambled = [p + 1.0 for p in params]
        pol.set_params(scrambled)
        assert not np.allclose(pol.forward(obs)[0], mean_before)
        pol.set_params(params)
        assert np.allclose(pol.forward(obs)[0], mean_before)

    def test_value_function_shapes_and_determinism(self):
        v1 = MLPValueFunction(input_dim=5, hidden_sizes=(8,), seed=3)
        v2 = MLPValueFunction(input_dim=5, hidden_sizes=(8,), seed=3)
        x = np.random.default_rng(3).normal(size=(7, 5))
        out = v1.forward(x)
        assert out.shape == (7,)
        assert np.allclose(out, v2.forward(x))


class TestPPO:
    def _collect(self, agent, n=8):
        rng = np.random.default_rng(11)
        obs = np.zeros(3)
        for _ in range(n):
            obs = rng.normal(size=3)
            action, log_prob, value = agent.select_action(obs)
            agent.store_transition(obs, action, 0.5, False, log_prob, value)
        return obs

    def test_select_action_types(self):
        agent = PPO(obs_dim=3, act_dim=2, seed=0)
        action, log_prob, value = agent.select_action(np.zeros(3))
        assert action.shape == (2,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_update_runs_and_clears_buffer(self):
        agent = PPO(obs_dim=3, act_dim=2, n_epochs=1, batch_size=8, seed=0)
        last_obs = self._collect(agent, n=8)
        stats = agent.update(last_obs)
        assert set(stats) == {"policy_loss", "value_loss", "entropy"}
        assert all(np.isfinite(v) for v in stats.values())
        assert len(agent.buffer.obs) == 0  # buffer cleared
        assert len(agent.training_stats) == 1

    def test_update_deterministic(self):
        def run():
            agent = PPO(obs_dim=3, act_dim=2, n_epochs=1, batch_size=8, seed=7)
            last = self._collect(agent)
            return agent.update(last)

        assert run() == run()

    def test_update_changes_parameters(self):
        """The ES-style update must modify policy and value parameters."""
        agent = PPO(obs_dim=3, act_dim=2, n_epochs=1, batch_size=8, seed=1)
        last_obs = self._collect(agent, n=8)
        policy_before = [p.copy() for p in agent.policy.get_params()]
        value_before = [w.copy() for w in agent.value_fn.weights] + [
            b.copy() for b in agent.value_fn.biases
        ]
        agent.update(last_obs)
        policy_after = agent.policy.get_params()
        value_after = agent.value_fn.weights + agent.value_fn.biases
        assert any(
            not np.array_equal(b, a)
            for b, a in zip(policy_before, policy_after, strict=False)
        )
        assert any(
            not np.array_equal(b, a)
            for b, a in zip(value_before, value_after, strict=False)
        )


class TestSAC:
    def _fill(self, agent, n):
        rng = np.random.default_rng(31)
        obs = rng.normal(size=agent.obs_dim)
        for _ in range(n):
            nxt = rng.normal(size=agent.obs_dim)
            agent.store_transition(obs, agent.select_action(obs), 0.3, nxt, False)
            obs = nxt

    def test_select_action_bounded_by_tanh(self):
        agent = SAC(obs_dim=3, act_dim=2, seed=0)
        obs = np.random.default_rng(5).normal(scale=50, size=3)
        a_stoch = agent.select_action(obs)
        a_det = agent.select_action(obs, deterministic=True)
        assert a_stoch.shape == (2,)
        assert np.all(a_stoch > -1)
        assert np.all(a_stoch < 1)
        assert np.all(a_det > -1)
        assert np.all(a_det < 1)
        assert np.allclose(a_det, np.tanh(agent.policy.forward(obs)[0]))

    def test_update_returns_empty_when_buffer_small(self):
        agent = SAC(obs_dim=3, act_dim=2, batch_size=32, seed=0)
        self._fill(agent, n=10)
        assert agent.update() == {}

    def test_update_returns_stats(self):
        agent = SAC(obs_dim=3, act_dim=2, batch_size=16, seed=0)
        self._fill(agent, n=20)
        stats = agent.update()
        assert set(stats) == {"q_loss", "q1_mean", "alpha"}
        assert np.isfinite(stats["q_loss"])
        assert stats["alpha"] == agent.alpha

    def test_soft_update_polyak(self):
        """Target must move toward online: tau=1 -> exact copy."""
        agent = SAC(obs_dim=3, act_dim=2, tau=1.0, seed=0)
        rng = np.random.default_rng(8)
        for w in agent.q1.weights:
            w[...] = rng.normal(size=w.shape)
        agent._soft_update()
        for w, wt in zip(agent.q1.weights, agent.q1_target.weights, strict=False):
            assert np.allclose(wt, w)
        assert np.allclose(agent.q1_target.w_out, agent.q1.w_out)

    def test_soft_update_interpolates(self):
        agent = SAC(obs_dim=3, act_dim=2, tau=0.25, seed=0)
        before = [wt.copy() for wt in agent.q2_target.weights]
        for w in agent.q2.weights:
            w[...] = 10.0
        agent._soft_update()
        for w0, wt in zip(before, agent.q2_target.weights, strict=False):
            assert np.allclose(wt, 0.25 * 10.0 + 0.75 * w0)

    def test_target_tracks_online_over_updates(self):
        """Polyak update shrinks the target-vs-online gap by exactly tau."""
        agent = SAC(obs_dim=3, act_dim=2, batch_size=16, seed=2)
        self._fill(agent, n=20)
        for w in agent.q1.weights:
            w[...] += 1.0  # make online differ from target
        d_before = sum(
            np.linalg.norm(w - wt)
            for w, wt in zip(agent.q1.weights, agent.q1_target.weights, strict=False)
        )
        agent.update()
        d_after = sum(
            np.linalg.norm(w - wt)
            for w, wt in zip(agent.q1.weights, agent.q1_target.weights, strict=False)
        )
        assert np.isclose(d_after, (1 - agent.tau) * d_before, atol=1e-10)


def _rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class TestDroneHoverEnv:
    def test_reset_obs_shape_and_determinism(self):
        e1 = DroneHoverEnv(seed=42)
        e2 = DroneHoverEnv(seed=42)
        o1, o2 = e1.reset(), e2.reset()
        assert o1.shape == (12,)
        assert np.array_equal(o1, o2)
        assert e1.step_count == 0

    def test_reset_positions_near_target(self):
        env = DroneHoverEnv(seed=0)
        for _ in range(20):
            env.reset()
            assert env.state is not None
            assert np.linalg.norm(env.state.position - env.target_pos) < 1.5

    def test_step_contract(self):
        env = DroneHoverEnv(seed=1)
        env.reset()
        obs, reward, done, info = env.step(np.zeros(4))
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert info == {}
        assert env.step_count == 1

    def test_reward_formula(self):
        """Reward must equal -||dp||^2 - 0.1||v||^2 - 0.01||w||^2
        - 0.001||a||^2 + 0.1 (+1 if ||dp|| < 0.1)."""
        env = DroneHoverEnv(seed=2)
        env.reset()
        st = QuadrotorState()
        st.position = np.array([0.3, -0.4, 0.5])
        st.velocity = np.array([0.1, 0.2, -0.1])
        st.omega = np.array([0.05, -0.05, 0.1])
        st.rotation = np.eye(3)
        env.state = st
        action = np.array([0.5, -0.5, 0.2, 0.0])
        expected = (
            -(np.linalg.norm(st.position - env.target_pos) ** 2)
            - 0.1 * np.linalg.norm(st.velocity) ** 2
            - 0.01 * np.linalg.norm(st.omega) ** 2
            - 0.001 * np.sum(action**2)
            + 0.1
        )
        assert np.isclose(env._compute_reward(action), expected, atol=1e-12)
        # inside the 0.1 m success ball the bonus applies
        st.position = env.target_pos + np.array([0.05, 0.0, 0.0])
        st.velocity = np.zeros(3)
        st.omega = np.zeros(3)
        # 1.0 bonus + 0.1 alive - 0.05^2 position cost
        assert env._compute_reward(np.zeros(4)) == pytest.approx(1.0975)

    def test_done_conditions(self):
        env = DroneHoverEnv(seed=3, max_steps=10)
        env.reset()
        assert env.state is not None

        env.state.position = env.target_pos + np.array([3.5, 0, 0])  # too far
        assert env._is_done()

        env.reset()
        env.state.position = env.target_pos.copy()
        env.state.rotation = _rot_x(np.radians(80))  # tilt > 75 deg
        assert env._is_done()

        env.reset()
        env.state.position = env.target_pos.copy()
        env.state.rotation = _rot_x(np.radians(10))  # mild tilt -> alive
        assert not env._is_done()

        env.reset()
        env.state.position = env.target_pos.copy()
        env.step_count = env.max_steps  # timeout
        assert env._is_done()

    def test_hover_action_keeps_drone_close(self):
        """Exact hover thrust should neither climb nor fall."""
        env = DroneHoverEnv(seed=4)
        env.reset()
        assert env.state is not None
        m, g = env.params.mass, 9.81
        f_hover = m * g / 4.0
        a_hover = 2.0 * f_hover / env.params.max_thrust_per_motor - 1.0
        start = env.state.position.copy()
        for _ in range(20):
            _obs, _r, done, _ = env.step(np.full(4, a_hover))
            assert not done
        assert np.linalg.norm(env.state.position - start) < 0.25

    def test_termination_on_full_throttle(self):
        env = DroneHoverEnv(seed=5)
        env.reset()
        done = False
        for _ in range(200):
            _, _, done, _ = env.step(np.ones(4))  # max thrust -> flies away/up
            if done:
                break
        assert done  # must eventually terminate (3 m limit or timeout)


class TestDroneNavigationEnv:
    GRID: tuple[int, int, int] = (20, 20, 10)
    RES: float = 0.5

    def _env(self, **kw):
        defaults: dict[str, Any] = {
            "grid_shape": self.GRID,
            "resolution": self.RES,
            "n_obstacles": 4,
            "n_depth_rays": 8,
            "max_steps": 200,
            "seed": 7,
        }
        defaults.update(kw)
        return DroneNavigationEnv(**defaults)

    def test_reset_obs_shape_and_free_start_goal(self):
        env = self._env()
        obs = env.reset()
        assert obs.shape == (12 + env.n_depth_rays,)
        assert env.state is not None
        assert env.goal is not None
        assert env.grid.is_free_world(env.state.position)
        assert env.grid.is_free_world(env.goal)
        assert np.linalg.norm(env.goal - env.state.position) >= 2.0

    def test_step_contract_and_info(self):
        env = self._env()
        env.reset()
        obs, reward, done, info = env.step(np.zeros(4))
        assert obs.shape == (12 + env.n_depth_rays,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        # NOTE: env returns np.bool_ here (minor type wart, behaves like bool)
        assert isinstance(info["goal_reached"], (bool, np.bool_))
        assert env.step_count == 1

    def test_depth_rays_in_valid_range(self):
        env = self._env()
        env.reset()
        depths = env._cast_depth_rays()
        assert depths.shape == (env.n_depth_rays,)
        assert np.all(depths >= 0.1)
        assert np.all(depths <= 5.0)

    def test_depth_rays_detect_wall_ahead(self):
        """A ray pointing at an adjacent obstacle must return a short hit."""
        env = self._env()
        env.reset()
        assert env.state is not None
        occ = np.argwhere(env.grid.grid)
        # sit one voxel in -x of an obstacle, with the neighbour free
        spot = None
        for i, j, k in occ:
            if i > 0 and env.grid.is_free((i - 1, j, k)):
                spot = (i - 1, j, k)
                break
        if spot is None:
            pytest.skip("no obstacle with a free -x neighbour")
        env.state.position = env.grid.grid_to_world(spot)
        env.state.rotation = np.eye(3)  # body +x aligned with world +x
        d = env._cast_depth_rays()
        assert d.shape == (env.n_depth_rays,)
        # ray 0 points along +x, straight into the obstacle voxel
        assert d[0] <= self.RES + 0.15
        assert np.all(d >= 0.1)
        assert np.all(d <= 5.0)

    def test_reward_progress_and_collision(self):
        env = self._env()
        env.reset()
        assert env.state is not None
        assert env.goal is not None
        # simulate having closed 0.2 m of distance -> positive progress reward
        dist = float(np.linalg.norm(env.state.position - env.goal))
        env.prev_dist = dist + 0.2
        r_progress = env._compute_reward(np.zeros(4))
        assert r_progress > 0.0
        assert np.isclose(r_progress, 10.0 * 0.2 - 0.01, atol=1e-9)

        # inside an obstacle -> big negative collision penalty
        occ = np.argwhere(env.grid.grid)
        env.state.position = env.grid.grid_to_world(tuple(occ[0]))
        env.prev_dist = float(
            np.linalg.norm(env.state.position - env.goal)
        )  # no progress
        r_coll = env._compute_reward(np.zeros(4))
        assert np.isclose(r_coll, -100.0 - 0.01, atol=1e-9)

    def test_done_on_collision_goal_and_timeout(self):
        env = self._env(max_steps=50)
        env.reset()
        assert env.state is not None
        assert env.goal is not None
        env.state.position = env.goal + np.array([0.1, 0, 0])
        assert env._is_done()  # goal reached

        env.reset()
        occ = np.argwhere(env.grid.grid)
        env.state.position = env.grid.grid_to_world(tuple(occ[0]))
        assert env._is_done()  # inside obstacle

        env.reset()
        env.step_count = 50
        assert env._is_done()  # timeout

    def test_deterministic_episode(self):
        def rollout():
            env = self._env()
            env.reset()
            rng = np.random.default_rng(99)
            out = []
            for _ in range(10):
                a = rng.uniform(-1, 1, 4)
                obs, r, done, _ = env.step(a)
                out.append((r, done, obs.copy()))
                if done:
                    break
            return out

        ep1, ep2 = rollout(), rollout()
        assert len(ep1) == len(ep2)
        for (r1, d1, o1), (r2, d2, o2) in zip(ep1, ep2, strict=False):
            assert r1 == r2
            assert d1 == d2
            assert np.array_equal(o1, o2)
