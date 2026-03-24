"""Tests for world model training methods."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from mbrl.data import Transition
from mbrl.world_models.mle import EnsembleDynamicsModel, MLEEnsemble
from mbrl.world_models.termination_fns import (
    get_termination_fn,
    termination_fn_halfcheetah,
    termination_fn_hopper,
    termination_fn_walker2d,
)

# Small dims for fast tests
OBS_DIM = 4
ACT_DIM = 2
NUM_ENSEMBLE = 3
NUM_ELITES = 2
N = 200  # transitions

FAST_CFG = OmegaConf.create({
    "num_ensemble": NUM_ENSEMBLE,
    "num_elites": NUM_ELITES,
    "n_layers": 2,
    "layer_size": 32,
    "num_epochs": 5,
    "lr": 1e-3,
    "batch_size": 32,
    "weight_decay": 2.5e-5,
    "logvar_diff_coef": 0.01,
    "validation_split": 0.2,
})


@pytest.fixture(scope="module")
def synthetic_dataset():
    rng = np.random.default_rng(0)
    obs = jnp.array(rng.standard_normal((N, OBS_DIM)), dtype=jnp.float32)
    action = jnp.array(rng.standard_normal((N, ACT_DIM)), dtype=jnp.float32)
    reward = jnp.array(rng.standard_normal((N,)), dtype=jnp.float32)
    next_obs = jnp.array(rng.standard_normal((N, OBS_DIM)), dtype=jnp.float32)
    done = jnp.zeros((N,), dtype=jnp.float32)
    return Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)


@pytest.fixture(scope="module")
def trained_ensemble(synthetic_dataset):
    model = MLEEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", FAST_CFG)
    model.train(synthetic_dataset, FAST_CFG, jax.random.key(0))
    return model


class TestEnsembleDynamicsModel:
    def test_forward_pass_shape(self):
        model = EnsembleDynamicsModel(
            obs_dim=OBS_DIM, action_dim=ACT_DIM, num_ensemble=NUM_ENSEMBLE,
            n_layers=2, layer_size=32,
        )
        dummy_input = jnp.zeros(OBS_DIM + ACT_DIM)
        params = model.init(jax.random.key(0), dummy_input)
        mean, logvar = model.apply(params, dummy_input)
        assert mean.shape == (NUM_ENSEMBLE, OBS_DIM + 1)
        assert logvar.shape == (NUM_ENSEMBLE, OBS_DIM + 1)

    def test_logvar_clamped(self):
        model = EnsembleDynamicsModel(
            obs_dim=OBS_DIM, action_dim=ACT_DIM, num_ensemble=NUM_ENSEMBLE,
            n_layers=2, layer_size=32,
            max_logvar_init=0.5, min_logvar_init=-10.0,
        )
        dummy_input = jnp.zeros(OBS_DIM + ACT_DIM)
        params = model.init(jax.random.key(0), dummy_input)
        _, logvar = model.apply(params, dummy_input)
        # Soft clamp means values should be within a small margin of the init bounds
        assert jnp.all(logvar <= 0.5 + 1.0)
        assert jnp.all(logvar >= -10.0 - 1.0)


class TestMLEEnsembleTraining:
    def test_train_completes(self, synthetic_dataset):
        model = MLEEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", FAST_CFG)
        model.train(synthetic_dataset, FAST_CFG, jax.random.key(42))
        assert model.params is not None
        assert model.num_elites == NUM_ELITES

    def test_elite_selection(self, trained_ensemble):
        assert trained_ensemble.num_elites == NUM_ELITES
        # After pruning, ensemble axis 0 should equal num_elites
        ensemble_params = trained_ensemble.params["params"]["ensemble"]
        first_param = jax.tree.leaves(ensemble_params)[0]
        assert first_param.shape[0] == NUM_ELITES


class TestMLEEnsembleStep:
    def test_step_shapes(self, trained_ensemble):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        next_obs, reward, done = trained_ensemble.step(obs, action, jax.random.key(0))
        assert next_obs.shape == (OBS_DIM,)
        assert reward.shape == ()
        assert done.shape == ()

    def test_step_determinism(self, trained_ensemble):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        out1 = trained_ensemble.step(obs, action, jax.random.key(7))
        out2 = trained_ensemble.step(obs, action, jax.random.key(7))
        out3 = trained_ensemble.step(obs, action, jax.random.key(8))
        assert jnp.array_equal(out1[0], out2[0])
        assert not jnp.array_equal(out1[0], out3[0])


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

    def test_dispatcher(self):
        assert get_termination_fn("mujoco/halfcheetah/medium-v0") is termination_fn_halfcheetah
        assert get_termination_fn("mujoco/hopper/medium-v0") is termination_fn_hopper
        assert get_termination_fn("mujoco/walker2d/medium-v0") is termination_fn_walker2d

    def test_dispatcher_unknown(self):
        with pytest.raises(ValueError):
            get_termination_fn("mujoco/ant/medium-v0")
