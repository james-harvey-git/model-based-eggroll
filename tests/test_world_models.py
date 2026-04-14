"""Tests for world model training methods."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from mbrl.data import Transition
from mbrl.world_models.eggroll import EGGROLLEnsemble
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

FAST_CFG = OmegaConf.create(
    {
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
    }
)


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
            obs_dim=OBS_DIM,
            action_dim=ACT_DIM,
            num_ensemble=NUM_ENSEMBLE,
            n_layers=2,
            layer_size=32,
        )
        dummy_input = jnp.zeros(OBS_DIM + ACT_DIM)
        params = model.init(jax.random.key(0), dummy_input)
        mean, logvar = model.apply(params, dummy_input)
        assert mean.shape == (NUM_ENSEMBLE, OBS_DIM + 1)
        assert logvar.shape == (NUM_ENSEMBLE, OBS_DIM + 1)

    def test_logvar_clamped(self):
        model = EnsembleDynamicsModel(
            obs_dim=OBS_DIM,
            action_dim=ACT_DIM,
            num_ensemble=NUM_ENSEMBLE,
            n_layers=2,
            layer_size=32,
            max_logvar_init=0.5,
            min_logvar_init=-10.0,
        )
        dummy_input = jnp.zeros(OBS_DIM + ACT_DIM)
        params = model.init(jax.random.key(0), dummy_input)
        _, logvar = model.apply(params, dummy_input)
        # Soft clamp means values should be within a small margin of the init bounds
        assert jnp.all(logvar <= 0.5 + 1.0)
        assert jnp.all(logvar >= -10.0 - 1.0)


class TestMLEEnsembleTraining:
    def test_train_calls_log_fn(self, synthetic_dataset):
        model = MLEEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", FAST_CFG)
        log_calls: list[dict] = []

        def log_fn(epoch, train_loss, val_mse):
            log_calls.append(
                {"epoch": int(epoch), "train_loss": float(train_loss), "val_mse": float(val_mse)}
            )

        model.train(synthetic_dataset, FAST_CFG, jax.random.key(42), log_fn=log_fn)
        jax.effects_barrier()  # flush async callbacks before asserting

        assert len(log_calls) == FAST_CFG.num_epochs
        assert all(c["epoch"] == i for i, c in enumerate(log_calls))
        assert all("train_loss" in c for c in log_calls)
        assert all("val_mse" in c for c in log_calls)

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


# ---------------------------------------------------------------------------
# EGGROLLEnsemble tests
# ---------------------------------------------------------------------------

NUM_EGGROLL_MEMBERS = 3

EGGROLL_FAST_CFG = OmegaConf.create(
    {
        "num_members": NUM_EGGROLL_MEMBERS,
        "hidden_dims": [8, 8],
        "activation": "relu",
        "num_epochs": 20,
        "validation_split": 0.2,
        "log_interval": 5,
        "eggroll": {
            "population_size": 8,
            "group_size": 2,
            "noise_reuse": 1,
            "sigma": 0.01,
            "sigma_decay_rate": 0.997,
            "lr": 1e-3,
        },
    }
)

EGGROLL_SLOW_CFG = OmegaConf.create(
    {
        "num_members": NUM_EGGROLL_MEMBERS,
        "hidden_dims": [8, 8],
        "activation": "relu",
        "num_epochs": 200,
        "validation_split": 0.2,
        "log_interval": 10,
        "eggroll": {
            "population_size": 8,
            "group_size": 2,
            "noise_reuse": 1,
            "sigma": 0.01,
            "sigma_decay_rate": 0.997,
            "lr": 1e-3,
        },
    }
)


@pytest.fixture(scope="module")
def eggroll_trained_fast(synthetic_dataset):
    model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", EGGROLL_FAST_CFG)
    model.train(synthetic_dataset, EGGROLL_FAST_CFG, jax.random.key(40))
    return model


@pytest.fixture(scope="module")
def eggroll_trained_slow(synthetic_dataset):
    model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", EGGROLL_SLOW_CFG)
    log_data: list[dict] = []

    def log_fn(epoch, train_nll, val_rmse):
        log_data.append({"epoch": int(epoch), "val_rmse": float(val_rmse)})

    model.train(synthetic_dataset, EGGROLL_SLOW_CFG, jax.random.key(41), log_fn=log_fn)
    jax.effects_barrier()
    return model, log_data


class TestEGGROLLEnsembleInit:
    def test_initial_state(self):
        model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", EGGROLL_FAST_CFG)
        assert model._state is None
        assert model._last_train_epoch == 0

    def test_termination_fn_callable(self):
        model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", EGGROLL_FAST_CFG)
        assert callable(model.termination_fn)

    def test_rejects_num_members_above_positive_population_half(self):
        bad_cfg = OmegaConf.create(
            {
                **OmegaConf.to_container(EGGROLL_FAST_CFG),  # type: ignore[arg-type]
                "num_members": 5,
            }
        )
        with pytest.raises(AssertionError, match="num_members must be <="):
            EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", bad_cfg)


class TestEGGROLLEnsemblePredict:
    def test_predict_ensemble_shape(self, eggroll_trained_fast):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        means, stds = eggroll_trained_fast.predict_ensemble(obs, action)
        assert means.shape == (NUM_EGGROLL_MEMBERS, OBS_DIM + 1)
        assert stds.shape == (NUM_EGGROLL_MEMBERS, OBS_DIM + 1)

    def test_predict_ensemble_finite(self, eggroll_trained_fast):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        means, stds = eggroll_trained_fast.predict_ensemble(obs, action)
        assert jnp.all(jnp.isfinite(means))
        assert jnp.all(jnp.isfinite(stds))

    def test_predict_ensemble_members_distinct(self, eggroll_trained_fast):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        means, _ = eggroll_trained_fast.predict_ensemble(obs, action)
        # Different thread_ids (0, 2, 4) produce distinct perturbations
        assert not jnp.allclose(means[0], means[1])


class TestEGGROLLEnsembleStep:
    def test_step_shapes(self, eggroll_trained_fast):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        next_obs, reward, done = eggroll_trained_fast.step(obs, action, jax.random.key(0))
        assert next_obs.shape == (OBS_DIM,)
        assert reward.shape == ()
        assert done.shape == ()

    def test_step_finite(self, eggroll_trained_fast):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        next_obs, reward, done = eggroll_trained_fast.step(obs, action, jax.random.key(0))
        assert jnp.all(jnp.isfinite(next_obs))
        assert jnp.isfinite(reward)

    def test_step_aleatoric_noise(self, eggroll_trained_fast):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        next_obs_1, _, _ = eggroll_trained_fast.step(obs, action, jax.random.key(1))
        next_obs_2, _, _ = eggroll_trained_fast.step(obs, action, jax.random.key(2))
        # Different rng keys should produce different samples (aleatoric noise)
        assert not jnp.array_equal(next_obs_1, next_obs_2)


@pytest.mark.slow
class TestEGGROLLEnsembleTrain:
    def test_last_train_epoch(self, eggroll_trained_slow):
        model, _ = eggroll_trained_slow
        assert model._last_train_epoch == 199

    def test_state_populated(self, eggroll_trained_slow):
        model, _ = eggroll_trained_slow
        assert model._state is not None

    def test_val_rmse_decreases(self, eggroll_trained_slow):
        _, log_data = eggroll_trained_slow
        assert len(log_data) > 1
        assert log_data[-1]["val_rmse"] < log_data[0]["val_rmse"]


class TestEGGROLLEnsembleCheckpoint:
    def test_roundtrip_pickle_and_load(self, eggroll_trained_fast, tmp_path):
        checkpoint = {
            "eggroll_state": eggroll_trained_fast.checkpoint_state(),
            "last_train_epoch": eggroll_trained_fast._last_train_epoch,
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
            "dataset_id": "mujoco/halfcheetah/medium-v0",
            "world_model_cfg": OmegaConf.to_container(EGGROLL_FAST_CFG),
            "wm_group": "test-group",
        }
        checkpoint_path = tmp_path / "world_model.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        reloaded = EGGROLLEnsemble.load_from_checkpoint(checkpoint_path)
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        means_a, stds_a = eggroll_trained_fast.predict_ensemble(obs, action)
        means_b, stds_b = reloaded.predict_ensemble(obs, action)
        assert jnp.allclose(means_a, means_b)
        assert jnp.allclose(stds_a, stds_b)
