"""Tests for experiment runners."""

import pickle
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from mbrl.data import DatasetInfo, Transition
from mbrl.experiments import world_model as world_model_exp
from mbrl.logger import Logger

DATASET_ID = "mujoco/halfcheetah/medium-v0"
OBS_DIM = 4
ACT_DIM = 2
N = 200


@pytest.fixture
def synthetic_dataset():
    rng = np.random.default_rng(0)
    obs = jnp.array(rng.standard_normal((N, OBS_DIM)), dtype=jnp.float32)
    action = jnp.array(rng.standard_normal((N, ACT_DIM)), dtype=jnp.float32)
    reward = jnp.array(rng.standard_normal((N,)), dtype=jnp.float32)
    next_obs = jnp.array(rng.standard_normal((N, OBS_DIM)), dtype=jnp.float32)
    done = jnp.zeros((N,), dtype=jnp.float32)
    dataset = Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
    info = DatasetInfo(
        obs_mean=jnp.zeros(OBS_DIM),
        obs_std=jnp.ones(OBS_DIM),
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        dataset_id=DATASET_ID,
    )
    return dataset, info


@pytest.fixture
def run_cfg(tmp_path):
    return OmegaConf.create(
        {
            "seed": 0,
            "checkpoint_dir": str(tmp_path),
            "wandb": {"enabled": False},
            "dataset": {"name": DATASET_ID},
            "world_model": {
                "num_ensemble": 3,
                "num_elites": 2,
                "n_layers": 2,
                "layer_size": 32,
                "num_epochs": 3,
                "lr": 1e-3,
                "batch_size": 32,
                "weight_decay": 2.5e-5,
                "logvar_diff_coef": 0.01,
                "validation_split": 0.2,
            },
        }
    )


@pytest.fixture
def trained_checkpoint(run_cfg, synthetic_dataset, tmp_path):
    """Run the experiment once and return the checkpoint path."""
    dataset, info = synthetic_dataset
    logger = Logger(run_cfg)

    with patch("mbrl.experiments.world_model.load_dataset", return_value=(dataset, info)):
        world_model_exp.run(run_cfg, logger)

    return tmp_path / "world_model.pkl"


class TestWorldModelRun:
    def test_checkpoint_saved(self, trained_checkpoint):
        assert trained_checkpoint.exists()

    def test_checkpoint_contents(self, trained_checkpoint, run_cfg):
        with open(trained_checkpoint, "rb") as f:
            ckpt = pickle.load(f)

        assert "params" in ckpt
        assert "num_elites" in ckpt
        assert ckpt["num_elites"] == run_cfg.world_model.num_elites
        assert ckpt["obs_dim"] == OBS_DIM
        assert ckpt["act_dim"] == ACT_DIM
        assert ckpt["dataset_id"] == DATASET_ID
        assert ckpt["params"] is not None

    def test_log_fn_called_each_epoch(self, run_cfg, synthetic_dataset):
        dataset, info = synthetic_dataset
        log_calls: list[dict] = []

        class SpyLogger(Logger):
            def log_world_model_step(self, epoch: int, **metrics: float) -> None:
                log_calls.append({"epoch": epoch, **metrics})

        logger = SpyLogger(run_cfg)

        with patch("mbrl.experiments.world_model.load_dataset", return_value=(dataset, info)):
            world_model_exp.run(run_cfg, logger)
        jax.effects_barrier()  # flush async callbacks before asserting

        assert len(log_calls) == run_cfg.world_model.num_epochs
        assert all("train_loss" in c for c in log_calls)
        assert all("val_mse" in c for c in log_calls)
