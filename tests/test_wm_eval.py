"""Tests for the wm_eval stage and compute_val_mse on each world model class.

Round-trip strategy: train a small model capturing the final-epoch logged
``val_mse`` (or ``val_mse_elite``), pickle to the same checkpoint shape the
``world_model`` stage produces, reload, and assert
``compute_val_mse(val_split)`` matches.

Parity requires no tail-drop in training's batched val pass. Fixtures choose
``N`` and ``validation_split`` so ``n_val`` is divisible by ``batch_size`` —
once that holds, the SSE/count accumulator and training's mean-of-batch-means
are mathematically equal up to float ordering.
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytest

from mbrl.data import Transition, train_val_split
from mbrl.experiments import wm_eval
from mbrl.logger import Logger
from mbrl.world_models.eggroll import EGGROLLEnsemble
from mbrl.world_models.mle import MLEEnsemble
from mbrl.world_models.mle_dynamicsnet import MLEDynamicsNet

OBS_DIM = 4
ACT_DIM = 2
N = 64  # n_val=32 with validation_split=0.5
BATCH_SIZE = 32  # divides n_val=32 cleanly → no tail-drop in training-time val pass

MLE_CFG = OmegaConf.create(
    {
        "_target_": "mbrl.world_models.mle.MLEEnsemble",
        "num_ensemble": 3,
        "num_elites": 2,
        "n_layers": 2,
        "layer_size": 32,
        "num_epochs": 4,
        "lr": 1e-3,
        "batch_size": BATCH_SIZE,
        "weight_decay": 2.5e-5,
        "logvar_diff_coef": 0.01,
        "validation_split": 0.5,
    }
)

MLE_DYN_CFG = OmegaConf.create(
    {
        "_target_": "mbrl.world_models.mle_dynamicsnet.MLEDynamicsNet",
        "hidden_dims": [8, 8],
        "activation": "relu",
        "init_scheme": "eggroll",
        "num_epochs": 4,
        "batch_size": BATCH_SIZE,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "validation_split": 0.5,
        "logvar_diff_coef": 0.01,
        "max_logvar_init": 0.5,
        "min_logvar_init": -10.0,
        "num_members": 1,
        "seed": 0,
    }
)

EGGROLL_CFG = OmegaConf.create(
    {
        "_target_": "mbrl.world_models.eggroll.EGGROLLEnsemble",
        "num_members": 2,
        "hidden_dims": [8, 8],
        "activation": "relu",
        "num_epochs": 8,
        "validation_split": 0.5,
        "logvar_diff_coef": 0.01,
        "log_interval": 1,
        "full_validation_interval": 1,
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

DATASET_ID = "mujoco/halfcheetah/medium-v0"


@pytest.fixture(scope="module")
def synthetic_dataset() -> Transition:
    rng = np.random.default_rng(0)
    obs = jnp.array(rng.standard_normal((N, OBS_DIM)), dtype=jnp.float32)
    action = jnp.array(rng.standard_normal((N, ACT_DIM)), dtype=jnp.float32)
    reward = jnp.array(rng.standard_normal((N,)), dtype=jnp.float32)
    next_obs = jnp.array(rng.standard_normal((N, OBS_DIM)), dtype=jnp.float32)
    done = jnp.zeros((N,), dtype=jnp.float32)
    return Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)


def _disabled_logger() -> Logger:
    cfg = OmegaConf.create(
        {"wandb": {"enabled": False}, "seed": 0, "dataset": {"name": DATASET_ID}}
    )
    return Logger(cfg, wm_group="test-group", timestamp="20260101-000000")


def _write_mle_checkpoint(model: MLEEnsemble, path) -> None:
    ckpt = {
        "params": model.params,
        "num_elites": model.num_elites,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "dataset_id": DATASET_ID,
        "world_model_cfg": OmegaConf.to_container(MLE_CFG),
        "wm_group": "test-group",
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)


def _write_mle_dyn_checkpoint(model: MLEDynamicsNet, path) -> None:
    ckpt = {
        **model.checkpoint_state(),
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "dataset_id": DATASET_ID,
        "world_model_cfg": OmegaConf.to_container(MLE_DYN_CFG),
        "wm_group": "test-group",
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)


def _write_eggroll_checkpoint(model: EGGROLLEnsemble, path) -> None:
    ckpt = {
        "eggroll_state": model.checkpoint_state(),
        "last_train_epoch": model._last_train_epoch,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "dataset_id": DATASET_ID,
        "world_model_cfg": OmegaConf.to_container(EGGROLL_CFG),
        "wm_group": "test-group",
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)


class TestMLEEnsembleRoundtrip:
    def test_compute_val_mse_matches_training_val_mse_elite(
        self, synthetic_dataset, tmp_path
    ):
        captured: list[float] = []

        def log_fn(step, train_loss, val_mse, transitions_seen, forward_evals, **kwargs):
            elite = kwargs.get("val_mse_elite")
            if elite is not None and np.isfinite(elite):
                captured.append(float(elite))

        rng_in = jax.random.key(7)
        model = MLEEnsemble(OBS_DIM, ACT_DIM, DATASET_ID, MLE_CFG)
        model.train(synthetic_dataset, MLE_CFG, rng_in, log_fn=log_fn)

        # Mirror train()'s rng chain: rng, split_rng, init_rng = jax.random.split(rng, 3).
        _, val_data = train_val_split(
            synthetic_dataset,
            float(MLE_CFG.validation_split),
            jax.random.split(rng_in, 3)[1],
        )

        ckpt_path = tmp_path / "world_model.pkl"
        _write_mle_checkpoint(model, ckpt_path)
        reloaded = MLEEnsemble.load_from_checkpoint(ckpt_path)
        reloaded_val_mse = float(reloaded.compute_val_mse(val_data))

        assert captured, "training never logged a finite val_mse_elite"
        assert np.isclose(captured[-1], reloaded_val_mse, atol=1e-5), (
            f"final training val_mse_elite={captured[-1]} vs "
            f"compute_val_mse from checkpoint={reloaded_val_mse}"
        )


class TestMLEDynamicsNetRoundtrip:
    def test_compute_val_mse_matches_training_val_mse(self, synthetic_dataset, tmp_path):
        captured: list[float] = []

        def log_fn(step, train_loss, val_mse, transitions_seen, forward_evals, **kwargs):
            if np.isfinite(val_mse):
                captured.append(float(val_mse))

        rng_in = jax.random.key(11)
        model = MLEDynamicsNet(OBS_DIM, ACT_DIM, DATASET_ID, MLE_DYN_CFG)
        model.train(synthetic_dataset, MLE_DYN_CFG, rng_in, log_fn=log_fn)

        _, val_data = train_val_split(
            synthetic_dataset,
            float(MLE_DYN_CFG.validation_split),
            jax.random.split(rng_in, 3)[1],
        )

        ckpt_path = tmp_path / "world_model.pkl"
        _write_mle_dyn_checkpoint(model, ckpt_path)
        reloaded = MLEDynamicsNet.load_from_checkpoint(ckpt_path)
        reloaded_val_mse = float(reloaded.compute_val_mse(val_data))

        # captured contains the init log + per-epoch logs; the last is the final epoch.
        assert len(captured) >= 2
        assert np.isclose(captured[-1], reloaded_val_mse, atol=1e-5)


class TestEGGROLLEnsembleRoundtrip:
    def test_compute_val_mse_matches_training_val_mse(self, synthetic_dataset, tmp_path):
        captured: list[float] = []

        def log_fn(step, train_loss, val_mse, transitions_seen, forward_evals, **kwargs):
            if np.isfinite(val_mse):
                captured.append(float(val_mse))

        rng_in = jax.random.key(13)
        model = EGGROLLEnsemble(OBS_DIM, ACT_DIM, DATASET_ID, EGGROLL_CFG)
        model.train(synthetic_dataset, EGGROLL_CFG, rng_in, log_fn=log_fn)
        # Flush async jax.debug.callback emissions before asserting on `captured`.
        jax.effects_barrier()

        # Mirror train()'s rng chain (non-warm-start branch at eggroll.py:318):
        # rng, split_rng, init_rng, es_rng = jax.random.split(rng, 4).
        _, val_data = train_val_split(
            synthetic_dataset,
            float(EGGROLL_CFG.validation_split),
            jax.random.split(rng_in, 4)[1],
        )

        ckpt_path = tmp_path / "world_model.pkl"
        _write_eggroll_checkpoint(model, ckpt_path)
        reloaded = EGGROLLEnsemble.load_from_checkpoint(ckpt_path)
        reloaded_val_mse = float(reloaded.compute_val_mse(val_data))

        assert captured, "training never logged a finite val_mse"
        assert np.isclose(captured[-1], reloaded_val_mse, atol=1e-5), (
            f"final training val_mse={captured[-1]} vs "
            f"compute_val_mse from checkpoint={reloaded_val_mse}"
        )


class TestShapeMismatch:
    def test_wm_eval_raises_on_obs_dim_mismatch(self, synthetic_dataset, tmp_path):
        # Train a model with OBS_DIM=OBS_DIM, then ask wm_eval to evaluate it on a
        # dataset whose shape pretends to be (OBS_DIM-1, ACT_DIM). Easiest way to
        # provoke the shape check is to write a checkpoint whose recorded obs_dim
        # differs from the eval dataset's obs_dim.
        rng_in = jax.random.key(3)
        model = MLEEnsemble(OBS_DIM, ACT_DIM, DATASET_ID, MLE_CFG)
        model.train(synthetic_dataset, MLE_CFG, rng_in)
        ckpt_path = tmp_path / "world_model.pkl"
        ckpt = {
            "params": model.params,
            "num_elites": model.num_elites,
            "obs_dim": OBS_DIM + 5,  # deliberately wrong
            "act_dim": ACT_DIM,
            "dataset_id": DATASET_ID,
            "world_model_cfg": OmegaConf.to_container(MLE_CFG),
            "wm_group": "test-group",
        }
        with open(ckpt_path, "wb") as f:
            pickle.dump(ckpt, f)

        # Mock load_dataset by monkey-patching: wm_eval.run calls load_dataset(name).
        cfg: DictConfig = OmegaConf.create(  # type: ignore[assignment]
            {
                "checkpoint_dir": str(tmp_path),
                "dataset": {"name": DATASET_ID},
                "wandb": {"enabled": False},
                "seed": 0,
            }
        )

        from mbrl.data import DatasetInfo

        fake_info = DatasetInfo(
            obs_mean=jnp.zeros(OBS_DIM),
            obs_std=jnp.ones(OBS_DIM),
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            dataset_id=DATASET_ID,
        )
        from unittest.mock import patch

        with patch(
            "mbrl.experiments.wm_eval.load_dataset",
            return_value=(synthetic_dataset, fake_info),
        ):
            with pytest.raises(ValueError, match="Shape mismatch"):
                wm_eval.run(cfg, _disabled_logger())
