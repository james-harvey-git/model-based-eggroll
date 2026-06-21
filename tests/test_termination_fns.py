"""Tests for environment-specific episode termination functions."""

import jax.numpy as jnp
import pytest

from mbrl.world_models.termination_fns import (
    get_termination_fn,
    termination_fn_halfcheetah,
    termination_fn_hopper,
    termination_fn_pen,
    termination_fn_walker2d,
)

OBS_DIM = 4
ACT_DIM = 2


class TestTerminationFns:
    def test_halfcheetah_normal(self):
        obs = jnp.zeros(OBS_DIM)
        act = jnp.zeros(ACT_DIM)
        next_obs = jnp.ones(OBS_DIM) * 5.0
        assert not termination_fn_halfcheetah(obs, act, next_obs)

    def test_halfcheetah_out_of_bounds(self):
        obs = jnp.zeros(OBS_DIM)
        act = jnp.zeros(ACT_DIM)
        next_obs = jnp.ones(OBS_DIM) * 200.0
        assert termination_fn_halfcheetah(obs, act, next_obs)

    def test_hopper_out_of_bounds(self):
        obs = jnp.zeros(OBS_DIM)
        act = jnp.zeros(ACT_DIM)
        # next_obs[1:] values outside ±100 should trigger termination
        next_obs = jnp.zeros(OBS_DIM).at[1].set(-200.0)
        assert termination_fn_hopper(obs, act, next_obs)

    def test_pen_held(self):
        # Pen still held (z >= 0.075) -> not done.
        obs = jnp.zeros(45)
        act = jnp.zeros(24)
        next_obs = jnp.zeros(45).at[26].set(0.21)  # typical held-pen height
        assert not termination_fn_pen(obs, act, next_obs)

    def test_pen_dropped(self):
        # Pen dropped below z=0.075 -> done.
        obs = jnp.zeros(45)
        act = jnp.zeros(24)
        next_obs = jnp.zeros(45).at[26].set(0.0)
        assert termination_fn_pen(obs, act, next_obs)

    def test_dispatcher(self):
        assert get_termination_fn("mujoco/halfcheetah/medium-v0") is termination_fn_halfcheetah
        assert get_termination_fn("mujoco/hopper/medium-v0") is termination_fn_hopper
        assert get_termination_fn("mujoco/walker2d/medium-v0") is termination_fn_walker2d
        assert get_termination_fn("D4RL/pen/cloned-v2") is termination_fn_pen

    def test_dispatcher_unknown(self):
        with pytest.raises(ValueError):
            get_termination_fn("mujoco/ant/medium-v0")

    def test_dispatcher_pen_not_pendulum(self):
        # "pen/" guard must not misfire on invertedpendulum dataset IDs.
        with pytest.raises(ValueError):
            get_termination_fn("mujoco/invertedpendulum/medium-v0")
