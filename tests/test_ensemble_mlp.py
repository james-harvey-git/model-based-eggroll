"""Tests for the unified EnsembleMLP world model (backprop + eggroll trainers)."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytest

from mbrl.data import Transition
from mbrl.world_models.ensemble_mlp import EnsembleMLP

OBS_DIM = 4
ACT_DIM = 2
N = 200
NUM_ENSEMBLE = 3
NUM_ELITES = 2
DATASET_ID = "mujoco/halfcheetah/medium-v0"


@pytest.fixture(scope="module")
def synthetic_dataset():
    rng = np.random.default_rng(0)
    return Transition(
        obs=jnp.array(rng.standard_normal((N, OBS_DIM)), jnp.float32),
        action=jnp.array(rng.standard_normal((N, ACT_DIM)), jnp.float32),
        reward=jnp.array(rng.standard_normal((N,)), jnp.float32),
        next_obs=jnp.array(rng.standard_normal((N, OBS_DIM)), jnp.float32),
        done=jnp.zeros((N,), jnp.float32),
    )


def _backprop_cfg(**overrides) -> DictConfig:
    base = {
        "trainer": "backprop", "num_ensemble": NUM_ENSEMBLE, "num_elites": NUM_ELITES,
        "hidden_dims": [8, 8], "activation": "relu", "init_scheme": "eggroll", "backbone": "mlp",
        "num_epochs": 4, "batch_size": 32, "lr": 1e-3, "optimizer": "adamw",
        "optimizer_kwargs": {"eps": 1e-5, "weight_decay": 1e-5},
        "validation_split": 0.2, "logvar_diff_coef": 0.01,
        "log_interval": 2, "full_validation_interval": 4, "seed": 0,
    }
    return OmegaConf.create({**base, **overrides})


def _eggroll_cfg(**overrides) -> DictConfig:
    base = {
        "trainer": "eggroll", "num_ensemble": NUM_ENSEMBLE, "num_elites": NUM_ELITES,
        "hidden_dims": [8, 8], "activation": "relu", "init_scheme": "eggroll", "backbone": "mlp",
        "num_epochs": 15, "validation_split": 0.2, "logvar_diff_coef": 0.01,
        "log_interval": 5, "full_validation_interval": 10, "use_shared_perturbations": False,
        "seed": 0,
        "eggroll": {
            "population_size": 8, "group_size": 2, "noise_reuse": 1, "sigma": 0.02,
            "sigma_decay_rate": 0.997, "lr": 1e-3, "solver": "adamw",
            "solver_kwargs": {"weight_decay": 1e-5}, "use_batched_update": True,
        },
    }
    merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))
    assert isinstance(merged, DictConfig)
    return merged


def _train(cfg, dataset, key, log_fn=None):
    model = EnsembleMLP(OBS_DIM, ACT_DIM, DATASET_ID, cfg)
    model.train(dataset, cfg, jax.random.key(key), log_fn=log_fn)
    jax.effects_barrier()
    return model


def _assert_inference_ok(model, num_elites=NUM_ELITES):
    obs, action = jnp.zeros(OBS_DIM), jnp.zeros(ACT_DIM)
    means, stds = model.predict_ensemble(obs, action)
    assert means.shape == (num_elites, OBS_DIM + 1)
    assert stds.shape == (num_elites, OBS_DIM + 1)
    assert jnp.all(jnp.isfinite(means)) and jnp.all(jnp.isfinite(stds))
    next_obs, reward, done = model.step(obs, action, jax.random.key(7))
    assert next_obs.shape == (OBS_DIM,) and reward.shape == () and done.shape == ()
    assert jnp.all(jnp.isfinite(next_obs)) and jnp.isfinite(reward)


@pytest.fixture(scope="module")
def backprop_model(synthetic_dataset):
    return _train(_backprop_cfg(), synthetic_dataset, 0)


@pytest.fixture(scope="module")
def eggroll_model(synthetic_dataset):
    return _train(_eggroll_cfg(), synthetic_dataset, 1)


class TestEnsembleMLPBackprop:
    def test_train_completes(self, backprop_model):
        assert backprop_model._params is not None
        assert backprop_model._elite_idxs is not None
        assert backprop_model._elite_idxs.shape == (NUM_ELITES,)
        assert backprop_model._update_steps_completed > 0

    def test_inference(self, backprop_model):
        _assert_inference_ok(backprop_model)

    def test_members_distinct(self, backprop_model):
        means, _ = backprop_model.predict_ensemble(jnp.zeros(OBS_DIM), jnp.zeros(ACT_DIM))
        assert not jnp.allclose(means[0], means[1])

    def test_sub_epoch_val_logging(self, synthetic_dataset):
        cfg = _backprop_cfg()
        rows: list[dict] = []

        def log_fn(step, train_loss, val_mse, transitions_seen, forward_evals, **kw):
            rows.append({"step": int(step), "val_mse": float(val_mse)})

        _train(cfg, synthetic_dataset, 0, log_fn=log_fn)
        n_train = int((1 - cfg.validation_split) * N)
        batches_per_epoch = n_train // cfg.batch_size
        val_steps = [r["step"] for r in rows if np.isfinite(r["val_mse"]) and r["step"] > 0]
        # At least one validation lands mid-epoch (proves sub-epoch logging).
        assert any(s % batches_per_epoch != 0 for s in val_steps)
        # transitions_seen is per-update (step * batch_size), not per-epoch.

    def test_checkpoint_roundtrip(self, backprop_model, tmp_path):
        _roundtrip(backprop_model, _backprop_cfg(), tmp_path)


class TestEnsembleMLPEggroll:
    def test_train_completes(self, eggroll_model):
        assert eggroll_model._params is not None
        assert eggroll_model._elite_idxs.shape == (NUM_ELITES,)

    def test_inference(self, eggroll_model):
        _assert_inference_ok(eggroll_model)

    def test_members_distinct(self, eggroll_model):
        means, _ = eggroll_model.predict_ensemble(jnp.zeros(OBS_DIM), jnp.zeros(ACT_DIM))
        assert not jnp.allclose(means[0], means[1])

    def test_transitions_seen_not_scaled_by_ensemble(self, synthetic_dataset):
        cfg = _eggroll_cfg()
        rows: list[dict] = []

        def log_fn(step, train_loss, val_mse, transitions_seen, forward_evals, **kw):
            rows.append({"ts": int(transitions_seen), "fe": int(forward_evals)})

        _train(cfg, synthetic_dataset, 1, log_fn=log_fn)
        n_prompts = cfg.eggroll.population_size // cfg.eggroll.group_size
        # Shared batch across members => transitions = n_prompts * step, NOT x num_ensemble.
        assert rows[-1]["ts"] == n_prompts * cfg.num_epochs
        # forward_evals (compute) DOES scale ~ num_ensemble.
        assert rows[-1]["fe"] >= cfg.num_epochs * cfg.eggroll.population_size * cfg.num_ensemble

    def test_num_ensemble_one_single_net(self, synthetic_dataset):
        model = _train(_eggroll_cfg(num_ensemble=1, num_elites=1), synthetic_dataset, 2)
        _assert_inference_ok(model, num_elites=1)

    def test_shared_perturbations_trains(self, synthetic_dataset):
        model = _train(_eggroll_cfg(use_shared_perturbations=True), synthetic_dataset, 1)
        _assert_inference_ok(model)

    def test_checkpoint_roundtrip(self, eggroll_model, tmp_path):
        _roundtrip(eggroll_model, _eggroll_cfg(), tmp_path)


class TestEnsembleMLPCrossTrainerHandoff:
    def test_backprop_to_eggroll(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt_path = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        cfg = _eggroll_cfg(init_checkpoint=str(ckpt_path), reset_optax_state=True, num_epochs=5)
        model = _train(cfg, synthetic_dataset, 3)
        _assert_inference_ok(model)

    def test_eggroll_to_backprop(self, eggroll_model, synthetic_dataset, tmp_path):
        ckpt_path = _write_ckpt(eggroll_model, _eggroll_cfg(), tmp_path)
        cfg = _backprop_cfg(init_checkpoint=str(ckpt_path), reset_optax_state=True, num_epochs=3)
        model = _train(cfg, synthetic_dataset, 4)
        _assert_inference_ok(model)


def _write_ckpt(model, cfg, tmp_path):
    ckpt = {
        **model.checkpoint_state(),
        "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "dataset_id": DATASET_ID,
        "world_model_cfg": OmegaConf.to_container(cfg), "wm_group": "test-group",
    }
    path = tmp_path / "world_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    return path


def _roundtrip(model, cfg, tmp_path):
    path = _write_ckpt(model, cfg, tmp_path)
    reloaded = EnsembleMLP.load_from_checkpoint(path)
    obs, action = jnp.zeros(OBS_DIM), jnp.zeros(ACT_DIM)
    ma, sa = model.predict_ensemble(obs, action)
    mb, sb = reloaded.predict_ensemble(obs, action)
    assert jnp.allclose(ma, mb) and jnp.allclose(sa, sb)
    assert jnp.array_equal(reloaded._elite_idxs, model._elite_idxs)
