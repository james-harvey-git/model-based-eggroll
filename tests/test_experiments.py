"""Tests for experiment runners."""

import importlib
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from mbrl.data import DatasetInfo, Transition
from mbrl.experiments import policy as policy_exp
from mbrl.experiments import world_model as world_model_exp
from mbrl.logger import Logger
from mbrl.main import (
    _find_latest_policy_run,
    _find_latest_wm_for_dataset,
    _resolve_eval_inputs,
    _update_latest_symlink,
)

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
                "_target_": "mbrl.world_models.mle.MLEEnsemble",
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


@pytest.fixture
def policy_run_cfg(tmp_path):
    wm_dir = tmp_path / "wm-run"
    policy_dir = wm_dir / "policies" / "mopo-s0-20260413-120000"
    return OmegaConf.create(
        {
            "seed": 0,
            "stage": "policy",
            "debug": False,
            "checkpoint_dir": str(wm_dir),
            "policy_checkpoint_dir": str(policy_dir),
            "wandb": {"enabled": False},
            "dataset": {"name": DATASET_ID},
            "world_model": {"_target_": "mbrl.world_models.mle.MLEEnsemble"},
            "policy_optimizer": {
                "_target_": "mbrl.policy_optimizers.mopo.train",
                "eval_interval": 1,
                "num_policy_updates": 0,
            },
            "eval": {"num_episodes": 1, "eval_workers": 1},
        }
    )


def _dump_pickle(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


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
        assert "wm_group" in ckpt
        assert isinstance(ckpt["wm_group"], str)

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
        assert all("transitions_seen" in c for c in log_calls)
        assert all("forward_evals" in c for c in log_calls)
        assert all("wall_time_sec" in c for c in log_calls)
        assert [c["transitions_seen"] for c in log_calls] == sorted(
            c["transitions_seen"] for c in log_calls
        )
        assert [c["forward_evals"] for c in log_calls] == sorted(
            c["forward_evals"] for c in log_calls
        )
        assert [c["wall_time_sec"] for c in log_calls] == sorted(
            c["wall_time_sec"] for c in log_calls
        )


class TestPolicyRun:
    def test_policy_checkpoint_contains_provenance(self, policy_run_cfg, synthetic_dataset):
        dataset, info = synthetic_dataset
        wm_checkpoint_path = Path(policy_run_cfg.checkpoint_dir) / "world_model.pkl"
        _dump_pickle(
            wm_checkpoint_path,
            {
                "dataset_id": DATASET_ID,
                "wm_group": "mle-halfcheetah-medium-s0-20260413-120000",
            },
        )

        fake_pol_module = SimpleNamespace(
            make_train_step=lambda world_model, dataset, cfg, rng: (lambda state, _: (state, {}), {
                "actor_params": {"params": {"w": jnp.ones((ACT_DIM,))}},
                "step": 7,
            }),
            extract_actor=lambda runner_state: (runner_state["actor_params"], runner_state["step"]),
        )

        logger = Logger(
            policy_run_cfg,
            wm_group="mle-halfcheetah-medium-s0-20260413-120000",
            timestamp="20260413-120000",
        )
        original_import_module = importlib.import_module

        with patch("mbrl.experiments.policy.load_dataset", return_value=(dataset, info)):
            with patch(
                "mbrl.experiments.policy.MLEEnsemble.load_from_checkpoint",
                return_value=object(),
            ):
                with patch(
                    "mbrl.experiments.policy.importlib.import_module",
                    side_effect=lambda name: (
                        fake_pol_module
                        if name == "mbrl.policy_optimizers.mopo"
                        else original_import_module(name)
                    ),
                ):
                    policy_exp.run(policy_run_cfg, logger)

        with open(Path(policy_run_cfg.policy_checkpoint_dir) / "policy.pkl", "rb") as f:
            ckpt = pickle.load(f)

        assert ckpt["dataset_id"] == DATASET_ID
        assert ckpt["world_model_dataset_id"] == DATASET_ID
        assert ckpt["wm_group"] == "mle-halfcheetah-medium-s0-20260413-120000"
        assert ckpt["seed"] == 0
        assert ckpt["stage"] == "policy"
        assert ckpt["algorithm"] == "mopo"
        assert ckpt["step"] == 7
        assert ckpt["world_model_checkpoint_path"] == str(wm_checkpoint_path.resolve())
        assert ckpt["world_model_checkpoint_parent"] == str(wm_checkpoint_path.parent.resolve())
        assert "created_at" in ckpt


class TestEvalResolution:
    def test_prefers_policy_checkpoint_wm_group(self, tmp_path):
        run_dir = tmp_path / "run-a"
        policy_dir = run_dir / "policies" / "mopo-s0-a"
        _dump_pickle(
            run_dir / "world_model.pkl",
            {"dataset_id": DATASET_ID, "wm_group": "wm-group-a"},
        )
        _dump_pickle(
            policy_dir / "policy.pkl",
            {"dataset_id": DATASET_ID, "wm_group": "wm-group-a"},
        )

        wm_ckpt, resolved_policy_dir, policy_ckpt, wm_group = _resolve_eval_inputs(
            tmp_path, DATASET_ID, str(policy_dir)
        )

        assert resolved_policy_dir == policy_dir
        assert policy_ckpt["wm_group"] == "wm-group-a"
        assert wm_group == "wm-group-a"
        assert wm_ckpt == run_dir / "world_model.pkl"

    def test_lineage_mismatch_raises(self, tmp_path):
        run_a = tmp_path / "run-a"
        run_b = tmp_path / "run-b"
        policy_dir = run_b / "policies" / "mopo-s0-b"
        _dump_pickle(
            run_a / "world_model.pkl",
            {"dataset_id": DATASET_ID, "wm_group": "wm-group-a"},
        )
        _dump_pickle(
            run_b / "world_model.pkl",
            {"dataset_id": DATASET_ID, "wm_group": "wm-group-b"},
        )
        _dump_pickle(
            policy_dir / "policy.pkl",
            {"dataset_id": DATASET_ID, "wm_group": "wm-group-b"},
        )

        with pytest.raises(ValueError, match="Lineage mismatch"):
            _resolve_eval_inputs(run_a, DATASET_ID, str(policy_dir))

    def test_lineage_mismatch_can_be_overridden(self, tmp_path):
        run_a = tmp_path / "run-a"
        run_b = tmp_path / "run-b"
        policy_dir = run_b / "policies" / "mopo-s0-b"
        _dump_pickle(
            run_a / "world_model.pkl",
            {"dataset_id": DATASET_ID, "wm_group": "wm-group-a"},
        )
        _dump_pickle(
            run_b / "world_model.pkl",
            {"dataset_id": DATASET_ID, "wm_group": "wm-group-b"},
        )
        _dump_pickle(
            policy_dir / "policy.pkl",
            {"dataset_id": DATASET_ID, "wm_group": "wm-group-b"},
        )

        _, _, _, wm_group = _resolve_eval_inputs(
            run_a, DATASET_ID, str(policy_dir), allow_lineage_mismatch=True
        )
        assert wm_group == "wm-group-b"


class TestUpdateLatestSymlink:
    def test_creates_symlink(self, tmp_path):
        (tmp_path / "run-1").mkdir()
        _update_latest_symlink(tmp_path, "run-1")
        link = tmp_path / "latest"
        assert link.is_symlink()
        assert link.resolve() == (tmp_path / "run-1").resolve()

    def test_updates_existing_symlink(self, tmp_path):
        (tmp_path / "run-1").mkdir()
        (tmp_path / "run-2").mkdir()
        _update_latest_symlink(tmp_path, "run-1")
        _update_latest_symlink(tmp_path, "run-2")
        assert (tmp_path / "latest").resolve() == (tmp_path / "run-2").resolve()


class TestLatestResolution:
    def test_policy_resolution_prefers_explicit_latest_pointer(self, tmp_path):
        policies_dir = tmp_path / "policies"
        older = policies_dir / "mopo-s0-old"
        newer = policies_dir / "mopo-s0-new"
        _dump_pickle(older / "policy.pkl", {"dataset_id": DATASET_ID, "wm_group": "old"})
        _dump_pickle(newer / "policy.pkl", {"dataset_id": DATASET_ID, "wm_group": "new"})
        _update_latest_symlink(policies_dir, older.name)

        resolved = _find_latest_policy_run(policies_dir)
        assert resolved == older.resolve()

    def test_world_model_resolution_validates_dataset_after_latest_pointer(self, tmp_path):
        halfcheetah_run = tmp_path / "mle-halfcheetah-medium-s0-1"
        hopper_run = tmp_path / "mle-hopper-medium-s0-1"
        _dump_pickle(
            halfcheetah_run / "world_model.pkl",
            {"dataset_id": DATASET_ID, "wm_group": "wm-halfcheetah"},
        )
        _dump_pickle(
            hopper_run / "world_model.pkl",
            {"dataset_id": "mujoco/hopper/medium-v0", "wm_group": "wm-hopper"},
        )
        _update_latest_symlink(tmp_path, hopper_run.name)

        resolved = _find_latest_wm_for_dataset(tmp_path, DATASET_ID)
        assert resolved == halfcheetah_run / "world_model.pkl"
