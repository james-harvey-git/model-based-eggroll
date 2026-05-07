"""Tests for world model training methods."""

import pickle
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from mbrl.data import DatasetInfo, Transition, train_val_split
from mbrl.experiments import world_model as world_model_exp
from mbrl.logger import Logger
from mbrl.world_models.eggroll import EGGROLLEnsemble, _eggroll_work_counters
from mbrl.world_models.mle import EnsembleDynamicsModel, MLEEnsemble
from mbrl.world_models.mle_dynamicsnet import MLEDynamicsNet
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

        def log_fn(step, train_loss, val_mse, transitions_seen, forward_evals, epoch=None):
            log_calls.append(
                {
                    "step": int(step),
                    "epoch": None if epoch is None else int(epoch),
                    "train_loss": float(train_loss),
                    "val_mse": float(val_mse),
                    "transitions_seen": int(transitions_seen),
                    "forward_evals": int(forward_evals),
                }
            )

        model.train(synthetic_dataset, FAST_CFG, jax.random.key(42), log_fn=log_fn)
        jax.effects_barrier()  # flush async callbacks before asserting

        n_train = int((1 - FAST_CFG.validation_split) * N)
        batches_per_epoch = n_train // FAST_CFG.batch_size
        train_examples_per_epoch = batches_per_epoch * FAST_CFG.batch_size
        val_examples_per_epoch = (N - n_train) // FAST_CFG.batch_size
        val_examples_per_epoch *= FAST_CFG.batch_size
        forward_evals_per_epoch = (
            train_examples_per_epoch + val_examples_per_epoch
        ) * FAST_CFG.num_ensemble

        assert len(log_calls) == FAST_CFG.num_epochs + 1
        # epoch is the logical epoch counter; step is cumulative update steps
        assert [c["epoch"] for c in log_calls] == list(range(FAST_CFG.num_epochs + 1))
        assert [c["step"] for c in log_calls] == [
            epoch * batches_per_epoch for epoch in range(FAST_CFG.num_epochs + 1)
        ]
        assert np.isnan(log_calls[0]["train_loss"])
        assert all("train_loss" in c for c in log_calls)
        assert all("val_mse" in c for c in log_calls)
        assert [c["transitions_seen"] for c in log_calls] == [
            0,
            *[train_examples_per_epoch * i for i in range(1, FAST_CFG.num_epochs + 1)],
        ]
        assert [c["forward_evals"] for c in log_calls] == [
            val_examples_per_epoch * FAST_CFG.num_ensemble,
            *[
                val_examples_per_epoch * FAST_CFG.num_ensemble + forward_evals_per_epoch * i
                for i in range(1, FAST_CFG.num_epochs + 1)
            ],
        ]

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
        "logvar_diff_coef": 0.01,
        "log_interval": 5,
        "full_validation_interval": 10,
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
        "logvar_diff_coef": 0.01,
        "log_interval": 10,
        "full_validation_interval": 20,
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

EGGROLL_UNGROUPED_CFG = OmegaConf.create(
    {
        "num_members": NUM_EGGROLL_MEMBERS,
        "hidden_dims": [8, 8],
        "activation": "relu",
        "num_epochs": 10,
        "validation_split": 0.2,
        "logvar_diff_coef": 0.01,
        "log_interval": 5,
        "full_validation_interval": 10,
        "eggroll": {
            "population_size": 8,
            "group_size": 0,
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

    def log_fn(epoch, train_nll, val_mse, transitions_seen, forward_evals):
        val_mse_f = float(val_mse)
        if np.isfinite(val_mse_f):
            log_data.append(
                {
                    "epoch": int(epoch),
                    "val_mse": val_mse_f,
                    "transitions_seen": int(transitions_seen),
                    "forward_evals": int(forward_evals),
                }
            )

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
    def test_work_counter_helper_uses_python_ints_for_large_values(self):
        step = 600_000
        transitions_seen, forward_evals = _eggroll_work_counters(
            step=step,
            n_prompts=4_096,
            population_size=4_096,
            n_val=200_000,
            full_validation_interval=5_000,
        )

        assert isinstance(transitions_seen, int)
        assert isinstance(forward_evals, int)
        assert transitions_seen == 2_457_600_000
        assert forward_evals == 2_482_000_000

    def test_last_train_epoch(self, eggroll_trained_slow):
        model, _ = eggroll_trained_slow
        assert model._last_train_epoch == 199

    def test_state_populated(self, eggroll_trained_slow):
        model, _ = eggroll_trained_slow
        assert model._state is not None

    def test_val_mse_decreases(self, eggroll_trained_slow):
        _, log_data = eggroll_trained_slow
        assert len(log_data) > 1
        assert log_data[0]["epoch"] == 0
        assert log_data[-1]["val_mse"] < log_data[0]["val_mse"]

    def test_work_counters_accumulate(self, eggroll_trained_slow):
        _, log_data = eggroll_trained_slow
        prompts_per_epoch = (
            EGGROLL_SLOW_CFG.eggroll.population_size // EGGROLL_SLOW_CFG.eggroll.group_size
        )
        expected_epochs = [0, 1] + list(
            range(
                EGGROLL_SLOW_CFG.full_validation_interval,
                EGGROLL_SLOW_CFG.num_epochs + 1,
                EGGROLL_SLOW_CFG.full_validation_interval,
            )
        )
        expected_transitions = [0] + [prompts_per_epoch * epoch for epoch in expected_epochs[1:]]
        n_val = N - int((1 - EGGROLL_SLOW_CFG.validation_split) * N)
        expected_forward_evals = [
            n_val,
            *[
                epoch * EGGROLL_SLOW_CFG.eggroll.population_size
                + (
                    1
                    + (
                        epoch // EGGROLL_SLOW_CFG.full_validation_interval
                        + (0 if EGGROLL_SLOW_CFG.full_validation_interval == 1 else 1)
                    )
                )
                * n_val
                for epoch in expected_epochs[1:]
            ],
        ]

        assert [entry["epoch"] for entry in log_data] == expected_epochs
        assert [entry["transitions_seen"] for entry in log_data] == expected_transitions
        assert [entry["forward_evals"] for entry in log_data] == expected_forward_evals

    def test_group_size_zero_trains(self, synthetic_dataset):
        model = EGGROLLEnsemble(
            OBS_DIM,
            ACT_DIM,
            "mujoco/halfcheetah/medium-v0",
            EGGROLL_UNGROUPED_CFG,
        )
        log_calls: list[dict] = []

        def log_fn(epoch, train_loss, val_mse, transitions_seen, forward_evals):
            log_calls.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "transitions_seen": int(transitions_seen),
                    "forward_evals": int(forward_evals),
                    "val_mse": float(val_mse),
                }
            )

        model.train(synthetic_dataset, EGGROLL_UNGROUPED_CFG, jax.random.key(52), log_fn=log_fn)
        jax.effects_barrier()

        assert model._state is not None
        assert log_calls[-1]["transitions_seen"] == (
            EGGROLL_UNGROUPED_CFG.eggroll.population_size * EGGROLL_UNGROUPED_CFG.num_epochs
        )
        assert np.isfinite(log_calls[-1]["train_loss"])


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


# ---------------------------------------------------------------------------
# MLEDynamicsNet tests
# ---------------------------------------------------------------------------

NUM_MLE_DYN_MEMBERS = 1

MLE_DYN_FAST_CFG = OmegaConf.create(
    {
        "hidden_dims": [8, 8],
        "activation": "relu",
        "init_scheme": "eggroll",
        "num_epochs": 5,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "validation_split": 0.2,
        "logvar_diff_coef": 0.01,
        "max_logvar_init": 0.5,
        "min_logvar_init": -10.0,
        "num_members": NUM_MLE_DYN_MEMBERS,
        "seed": 0,
    }
)


@pytest.fixture(scope="module")
def mle_dyn_trained(synthetic_dataset):
    model = MLEDynamicsNet(
        OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", MLE_DYN_FAST_CFG
    )
    model.train(synthetic_dataset, MLE_DYN_FAST_CFG, jax.random.key(0))
    return model


class TestMLEDynamicsNet:
    def test_train_completes(self, synthetic_dataset):
        model = MLEDynamicsNet(
            OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", MLE_DYN_FAST_CFG
        )
        model.train(synthetic_dataset, MLE_DYN_FAST_CFG, jax.random.key(42))
        assert model._params is not None
        assert model._opt_state is not None
        assert model._update_steps_completed > 0
        assert model._validation_split == MLE_DYN_FAST_CFG.validation_split
        assert model._seed == MLE_DYN_FAST_CFG.seed

    def test_train_calls_log_fn(self, synthetic_dataset):
        model = MLEDynamicsNet(
            OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", MLE_DYN_FAST_CFG
        )
        log_calls: list[dict] = []

        def log_fn(step, train_loss, val_mse, transitions_seen, forward_evals, epoch=None):
            log_calls.append(
                {
                    "step": int(step),
                    "epoch": None if epoch is None else int(epoch),
                    "train_loss": float(train_loss),
                    "val_mse": float(val_mse),
                    "transitions_seen": int(transitions_seen),
                    "forward_evals": int(forward_evals),
                }
            )

        model.train(synthetic_dataset, MLE_DYN_FAST_CFG, jax.random.key(42), log_fn=log_fn)
        jax.effects_barrier()

        n_train = int((1 - MLE_DYN_FAST_CFG.validation_split) * N)
        batches_per_epoch = n_train // MLE_DYN_FAST_CFG.batch_size

        assert len(log_calls) == MLE_DYN_FAST_CFG.num_epochs + 1
        assert [c["epoch"] for c in log_calls] == list(range(MLE_DYN_FAST_CFG.num_epochs + 1))
        assert [c["step"] for c in log_calls] == [
            epoch * batches_per_epoch for epoch in range(MLE_DYN_FAST_CFG.num_epochs + 1)
        ]
        assert np.isnan(log_calls[0]["train_loss"])
        assert all(np.isfinite(c["val_mse"]) for c in log_calls)

    def test_loss_decreases(self, synthetic_dataset):
        # Use a longer-than-fast config so any noise in the smoke run averages out.
        cfg = OmegaConf.create(
            {**OmegaConf.to_container(MLE_DYN_FAST_CFG), "num_epochs": 30}  # type: ignore[arg-type]
        )
        model = MLEDynamicsNet(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", cfg)
        train_losses: list[float] = []

        def log_fn(step, train_loss, val_mse, transitions_seen, forward_evals, epoch=None):
            tl = float(train_loss)
            if np.isfinite(tl):
                train_losses.append(tl)

        model.train(synthetic_dataset, cfg, jax.random.key(0), log_fn=log_fn)
        jax.effects_barrier()

        assert len(train_losses) >= 4
        first_two = sum(train_losses[:2]) / 2
        last_two = sum(train_losses[-2:]) / 2
        assert last_two < first_two

    def test_step_shapes(self, mle_dyn_trained):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        next_obs, reward, done = mle_dyn_trained.step(obs, action, jax.random.key(0))
        assert next_obs.shape == (OBS_DIM,)
        assert reward.shape == ()
        assert done.shape == ()

    def test_predict_ensemble_shape(self, mle_dyn_trained):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        means, stds = mle_dyn_trained.predict_ensemble(obs, action)
        assert means.shape == (NUM_MLE_DYN_MEMBERS, OBS_DIM + 1)
        assert stds.shape == (NUM_MLE_DYN_MEMBERS, OBS_DIM + 1)
        assert jnp.all(jnp.isfinite(means))
        assert jnp.all(jnp.isfinite(stds))

    def test_determinism(self, synthetic_dataset):
        model_a = MLEDynamicsNet(
            OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", MLE_DYN_FAST_CFG
        )
        model_b = MLEDynamicsNet(
            OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", MLE_DYN_FAST_CFG
        )
        model_a.train(synthetic_dataset, MLE_DYN_FAST_CFG, jax.random.key(123))
        model_b.train(synthetic_dataset, MLE_DYN_FAST_CFG, jax.random.key(123))
        leaves_a = jax.tree.leaves(model_a._params)
        leaves_b = jax.tree.leaves(model_b._params)
        assert len(leaves_a) == len(leaves_b)
        for a, b in zip(leaves_a, leaves_b):
            assert jnp.array_equal(a, b)

    def test_checkpoint_roundtrip(self, mle_dyn_trained, tmp_path):
        checkpoint = {
            **mle_dyn_trained.checkpoint_state(),
            "obs_dim": OBS_DIM,
            "act_dim": ACT_DIM,
            "dataset_id": "mujoco/halfcheetah/medium-v0",
            "world_model_cfg": OmegaConf.to_container(MLE_DYN_FAST_CFG),
            "wm_group": "test-group",
        }
        checkpoint_path = tmp_path / "world_model.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        reloaded = MLEDynamicsNet.load_from_checkpoint(checkpoint_path)
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        means_a, stds_a = mle_dyn_trained.predict_ensemble(obs, action)
        means_b, stds_b = reloaded.predict_ensemble(obs, action)
        assert jnp.allclose(means_a, means_b)
        assert jnp.allclose(stds_a, stds_b)
        assert reloaded._update_steps_completed == mle_dyn_trained._update_steps_completed
        assert reloaded._validation_split == mle_dyn_trained._validation_split
        assert reloaded._seed == mle_dyn_trained._seed


# ---------------------------------------------------------------------------
# Hybrid MLE→EGGROLL handoff tests (issue #30)
# ---------------------------------------------------------------------------

# Stage-2 EGGROLL config that matches MLE_DYN_FAST_CFG's network architecture
# so the param pytrees line up. num_epochs=0 lets the splice tests inspect the
# EGGROLL state immediately after init+splice without any training updates.
HYBRID_EGGROLL_CFG = OmegaConf.create(
    {
        "num_members": NUM_EGGROLL_MEMBERS,
        "hidden_dims": [8, 8],
        "activation": "relu",
        "init_scheme": "eggroll",
        "num_epochs": 0,
        "validation_split": 0.2,
        "logvar_diff_coef": 0.01,
        "log_interval": 1,
        "full_validation_interval": 1,
        "init_checkpoint": None,
        "reset_optax_state": False,
        "eggroll": {
            "population_size": 8,
            "group_size": 2,
            "noise_reuse": 1,
            "sigma": 0.01,
            "sigma_decay_rate": 0.997,
            "lr": 1e-3,
            "solver": "adamw",
            "solver_kwargs": {"weight_decay": 1e-5, "b1": 0.9},
            "use_batched_update": True,
        },
    }
)


@pytest.fixture(scope="module")
def stage1_checkpoint(synthetic_dataset, tmp_path_factory):
    """Train a stage-1 MLEDynamicsNet and write a checkpoint to disk.

    Returns ``(checkpoint_path, ckpt_dict, mle_model)``.
    """
    model = MLEDynamicsNet(
        OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", MLE_DYN_FAST_CFG
    )
    model.train(synthetic_dataset, MLE_DYN_FAST_CFG, jax.random.key(0))

    ckpt = {
        **model.checkpoint_state(),
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "dataset_id": "mujoco/halfcheetah/medium-v0",
        "world_model_cfg": OmegaConf.to_container(MLE_DYN_FAST_CFG),
        "wm_group": "test-stage1",
    }
    path = tmp_path_factory.mktemp("stage1") / "world_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)

    return path, ckpt, model


def _stage2_cfg(init_checkpoint, reset_optax_state=False):
    return OmegaConf.create(
        {
            **OmegaConf.to_container(HYBRID_EGGROLL_CFG),  # type: ignore[arg-type]
            "init_checkpoint": str(init_checkpoint),
            "reset_optax_state": reset_optax_state,
        }
    )


class TestHybridHandoff:
    def test_param_handoff(self, synthetic_dataset, stage1_checkpoint):
        ckpt_path, ckpt, _ = stage1_checkpoint
        cfg = _stage2_cfg(ckpt_path)
        model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", cfg)
        model.train(synthetic_dataset, cfg, jax.random.key(1))

        # With num_epochs=0 the fori_loop is a no-op; state.params == ckpt["params"].
        assert model._state is not None
        spliced = jax.tree.leaves(model._state.params)
        loaded = jax.tree.leaves(ckpt["params"])
        assert len(spliced) == len(loaded)
        for s, ld in zip(spliced, loaded):
            assert jnp.array_equal(s, ld)

    def test_optax_state_carryover(self, synthetic_dataset, stage1_checkpoint):
        ckpt_path, ckpt, _ = stage1_checkpoint
        cfg = _stage2_cfg(ckpt_path, reset_optax_state=False)
        model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", cfg)
        model.train(synthetic_dataset, cfg, jax.random.key(2))

        assert model._state is not None
        carried = jax.tree.leaves(model._state.noiser_params["opt_state"])
        loaded = jax.tree.leaves(ckpt["opt_state"])
        assert len(carried) == len(loaded)
        for c, ld in zip(carried, loaded):
            assert jnp.array_equal(c, ld)

    def test_optax_state_reset(self, synthetic_dataset, stage1_checkpoint):
        ckpt_path, ckpt, _ = stage1_checkpoint
        cfg = _stage2_cfg(ckpt_path, reset_optax_state=True)
        model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", cfg)
        model.train(synthetic_dataset, cfg, jax.random.key(3))

        assert model._state is not None
        # Loaded state has count > 0 (stage 1 ran several update steps); the
        # reset path leaves the freshly-initialised count at 0.
        loaded_counts = [
            int(c) for c in jax.tree.leaves(ckpt["opt_state"]) if jnp.asarray(c).shape == ()
        ]
        fresh_counts = [
            int(c)
            for c in jax.tree.leaves(model._state.noiser_params["opt_state"])
            if jnp.asarray(c).shape == ()
        ]
        assert any(c > 0 for c in loaded_counts), "stage-1 ckpt should have a non-zero count"
        assert all(c == 0 for c in fresh_counts), "reset path should re-init to count=0"

    def test_logvar_bounds_preserved(self, synthetic_dataset, stage1_checkpoint):
        ckpt_path, ckpt, _ = stage1_checkpoint
        cfg = _stage2_cfg(ckpt_path)
        model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", cfg)
        model.train(synthetic_dataset, cfg, jax.random.key(4))

        assert model._state is not None
        assert jnp.array_equal(
            model._state.params["max_logvar"], ckpt["params"]["max_logvar"]
        )
        assert jnp.array_equal(
            model._state.params["min_logvar"], ckpt["params"]["min_logvar"]
        )

    def test_val_set_parity(self, synthetic_dataset, stage1_checkpoint):
        """Stage 2's val partition must be bit-equal to stage 1's."""
        _, ckpt, _ = stage1_checkpoint

        # Recompute stage 1's val partition from the checkpoint's seed +
        # validation_split. This is what stage 2 must reproduce.
        stage1_root = jax.random.key(int(ckpt["seed"]))
        _, split_rng_stage1, _ = jax.random.split(stage1_root, 3)
        _, val_stage1 = train_val_split(
            synthetic_dataset, ckpt["validation_split"], split_rng_stage1
        )

        # Capture the val partition stage-2 actually uses.
        captured: dict = {}
        original = train_val_split

        def capturing(dataset, val_fraction, rng):
            tr, va = original(dataset, val_fraction, rng)
            captured["val"] = va
            return tr, va

        ckpt_path, _, _ = stage1_checkpoint
        cfg = _stage2_cfg(ckpt_path)
        model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", cfg)
        with patch("mbrl.world_models.eggroll.train_val_split", capturing):
            model.train(synthetic_dataset, cfg, jax.random.key(5))

        assert "val" in captured
        assert jnp.array_equal(captured["val"].obs, val_stage1.obs)
        assert jnp.array_equal(captured["val"].action, val_stage1.action)
        assert jnp.array_equal(captured["val"].next_obs, val_stage1.next_obs)
        assert jnp.array_equal(captured["val"].reward, val_stage1.reward)

    def test_dataset_id_mismatch_errors(self, synthetic_dataset, stage1_checkpoint, tmp_path):
        """`experiments.world_model.run` rejects a checkpoint from a different dataset."""
        _, ckpt, _ = stage1_checkpoint
        bad_ckpt = {**ckpt, "dataset_id": "mujoco/hopper/medium-v0"}
        bad_path = tmp_path / "bad_dataset.pkl"
        with open(bad_path, "wb") as f:
            pickle.dump(bad_ckpt, f)

        run_cfg = OmegaConf.create(
            {
                "seed": 0,
                "checkpoint_dir": str(tmp_path),
                "wandb": {"enabled": False},
                "dataset": {"name": "mujoco/halfcheetah/medium-v0"},
                "world_model": _stage2_cfg(bad_path),
            }
        )
        info = DatasetInfo(
            obs_mean=jnp.zeros(OBS_DIM),
            obs_std=jnp.ones(OBS_DIM),
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            dataset_id="mujoco/halfcheetah/medium-v0",
        )
        logger = Logger(run_cfg)
        with patch(
            "mbrl.experiments.world_model.load_dataset",
            return_value=(synthetic_dataset, info),
        ):
            with pytest.raises(ValueError, match="dataset_id mismatch"):
                world_model_exp.run(run_cfg, logger)

    def test_obs_act_dim_mismatch_errors(self, synthetic_dataset, stage1_checkpoint, tmp_path):
        """`experiments.world_model.run` rejects a checkpoint with wrong obs/act dims."""
        _, ckpt, _ = stage1_checkpoint
        bad_ckpt = {**ckpt, "obs_dim": OBS_DIM + 1}
        bad_path = tmp_path / "bad_dims.pkl"
        with open(bad_path, "wb") as f:
            pickle.dump(bad_ckpt, f)

        run_cfg = OmegaConf.create(
            {
                "seed": 0,
                "checkpoint_dir": str(tmp_path),
                "wandb": {"enabled": False},
                "dataset": {"name": "mujoco/halfcheetah/medium-v0"},
                "world_model": _stage2_cfg(bad_path),
            }
        )
        info = DatasetInfo(
            obs_mean=jnp.zeros(OBS_DIM),
            obs_std=jnp.ones(OBS_DIM),
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            dataset_id="mujoco/halfcheetah/medium-v0",
        )
        logger = Logger(run_cfg)
        with patch(
            "mbrl.experiments.world_model.load_dataset",
            return_value=(synthetic_dataset, info),
        ):
            with pytest.raises(ValueError, match="shape mismatch"):
                world_model_exp.run(run_cfg, logger)
