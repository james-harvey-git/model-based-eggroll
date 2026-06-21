"""Trivial-predictor baselines, NMSE normalization, and rollout-inspection figures."""

import jax.numpy as jnp
from matplotlib.figure import Figure
import numpy as np
import pytest

from mbrl.data import EpisodeBatch, TrajectoryWindows
from mbrl.world_models import rollout_figures as rf
from mbrl.world_models.ensemble_mlp import (
    _episode_obs_reward_mean,
    _masked_feature_stats,
    _normalized_curve_dict,
    _traj_baseline_curves,
)


def _windows(w=4, t=3, obs_dim=2, seed=0) -> TrajectoryWindows:
    rng = np.random.default_rng(seed)
    return TrajectoryWindows(
        start_obs=jnp.asarray(rng.standard_normal((w, obs_dim)), jnp.float32),
        actions=jnp.asarray(rng.standard_normal((w, t, 1)), jnp.float32),
        target_obs=jnp.asarray(rng.standard_normal((w, t, obs_dim)), jnp.float32),
        target_reward=jnp.asarray(rng.standard_normal((w, t)), jnp.float32),
        mask=jnp.ones((w, t), jnp.float32),
    )


class TestBaselines:
    def test_shapes_and_keys(self):
        win = _windows()
        obs_mean = win.target_obs.mean(axis=(0, 1))
        reward_mean = win.target_reward.mean()
        curves = _traj_baseline_curves(win, obs_mean, reward_mean)
        assert set(curves) == {"mean_state", "persistence"}
        assert len(curves["mean_state"]) == 3 and len(curves["persistence"]) == 3
        assert all(v >= 0 for v in curves["mean_state"] + curves["persistence"])

    def test_persistence_omitted_when_disabled(self):
        win = _windows()
        curves = _traj_baseline_curves(
            win, win.target_obs.mean(axis=(0, 1)), win.target_reward.mean(),
            include_persistence=False,
        )
        assert set(curves) == {"mean_state"}

    def test_mean_state_matches_variance(self):
        """The mean-predictor's per-step error equals the per-step target variance — the
        defining property of the 'blind average state' baseline."""
        win = _windows(w=64, seed=1)
        obs_mean = win.target_obs.mean(axis=(0, 1))
        reward_mean = win.target_reward.mean()
        curves = _traj_baseline_curves(win, obs_mean, reward_mean)
        target = jnp.concatenate(
            [win.target_obs, win.target_reward[..., None]], axis=-1
        )  # (W,T,D)
        const = jnp.concatenate([obs_mean, jnp.atleast_1d(reward_mean)])
        expected = jnp.mean((const[None, None] - target) ** 2, axis=-1).mean(axis=0)  # (T,)
        np.testing.assert_allclose(curves["mean_state"], np.asarray(expected), rtol=1e-5)

    def test_persistence_grows_with_horizon(self):
        """Open-loop zero-delta error grows as the true state drifts from the start."""
        # Construct a steadily drifting trajectory: target_obs[:, t] moves away from start.
        w, t, d = 8, 5, 2
        start = jnp.zeros((w, d))
        steps = jnp.arange(1, t + 1, dtype=jnp.float32)
        target_obs = jnp.broadcast_to(steps[None, :, None], (w, t, d))
        win = TrajectoryWindows(
            start_obs=start,
            actions=jnp.zeros((w, t, 1)),
            target_obs=target_obs,
            target_reward=jnp.zeros((w, t)),
            mask=jnp.ones((w, t)),
        )
        curves = _traj_baseline_curves(win, jnp.zeros((d,)), jnp.asarray(0.0))
        pers = curves["persistence"]
        assert all(pers[i] < pers[i + 1] for i in range(len(pers) - 1))


class TestNormalization:
    def test_mean_state_is_flat_one(self):
        baselines = {"mean_state": [4.0, 9.0, 16.0], "persistence": [1.0, 4.5, 8.0]}
        out = _normalized_curve_dict({"final": [2.0, 3.0, 4.0]}, baselines)
        assert out["mean_state"] == [1.0, 1.0, 1.0]
        np.testing.assert_allclose(out["final"], [0.5, 1 / 3, 0.25])
        np.testing.assert_allclose(out["persistence"], [0.25, 0.5, 0.5])

    def test_persistence_omitted_when_absent(self):
        out = _normalized_curve_dict({"final": [2.0, 4.0]}, {"mean_state": [4.0, 8.0]})
        assert set(out) == {"final", "mean_state"}

    def test_curve_equal_to_mean_state_normalizes_to_one(self):
        """A model whose rollout MSE equals the mean-state baseline scores exactly 1.0
        (no better than predicting the average state) — the grounding sanity check."""
        win = _windows(w=32, seed=2)
        obs_mean = win.target_obs.mean(axis=(0, 1))
        reward_mean = win.target_reward.mean()
        baselines = _traj_baseline_curves(win, obs_mean, reward_mean)
        out = _normalized_curve_dict({"final": baselines["mean_state"]}, baselines)
        np.testing.assert_allclose(out["final"], 1.0, rtol=1e-5)


class TestFeatureStats:
    def test_masked_feature_stats(self):
        win = _windows(w=16, seed=3)
        mean, var = _masked_feature_stats(win)
        assert mean.shape == (3,) and var.shape == (3,)  # obs_dim + reward
        assert jnp.all(var >= 0)

    def test_episode_means_ignore_padding(self):
        obs = jnp.ones((2, 4, 2))
        rew = jnp.concatenate([jnp.full((2, 2), 5.0), jnp.full((2, 1), 99.0)], axis=1)
        mask = jnp.concatenate([jnp.ones((2, 2)), jnp.zeros((2, 1))], axis=1)
        eps = EpisodeBatch(obs, jnp.zeros((2, 3, 1)), rew, mask, jnp.asarray([2, 2], jnp.int32))
        obs_mean, reward_mean = _episode_obs_reward_mean(eps)
        assert float(reward_mean) == pytest.approx(5.0)  # padded reward 99 excluded
        np.testing.assert_allclose(np.asarray(obs_mean), [1.0, 1.0])


class TestFigures:
    def test_timeseries_subplot_count(self):
        t, d = 5, 3
        true = np.zeros((t, d))
        fig = rf.plot_rollout_timeseries(true, true, true, true)
        assert isinstance(fig, Figure)
        drawn = [ax for ax in fig.axes if ax.has_data()]
        assert len(drawn) == d

    def test_heatmap_returns_figure(self):
        fig = rf.plot_error_heatmap(np.abs(np.random.randn(5, 4)))
        assert isinstance(fig, Figure)

    def test_phase_portraits_skipped_for_unregistered_env(self):
        traj = np.zeros((5, 18))
        assert rf.plot_joint_phase_portraits(traj, traj, "mujoco/ant/medium-v0") is None

    def test_phase_portraits_on_halfcheetah(self):
        traj = np.random.randn(5, 18)
        fig = rf.plot_joint_phase_portraits(traj, traj, "mujoco/halfcheetah/medium-v0")
        assert isinstance(fig, Figure)
        drawn = [ax for ax in fig.axes if ax.has_data()]
        assert len(drawn) == len(rf._PHASE_PORTRAIT_LAYOUTS["halfcheetah"])

    def test_phase_portraits_on_hopper(self):
        traj = np.random.randn(5, 12)  # 11 obs + reward, as build_rollout_figures passes
        fig = rf.plot_joint_phase_portraits(traj, traj, "mujoco/hopper/medium-v0")
        assert isinstance(fig, Figure)
        drawn = [ax for ax in fig.axes if ax.has_data()]
        assert len(drawn) == len(rf._PHASE_PORTRAIT_LAYOUTS["hopper"])

    def test_select_window_indices(self):
        sel = rf.select_window_indices(np.array([0.5, 0.1, 0.9, 0.3]))
        assert sel["best"] == 1 and sel["worst"] == 2
        assert set(sel) == {"best", "median", "worst"}
