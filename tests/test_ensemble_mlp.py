"""Tests for the unified EnsembleMLP world model (backprop + eggroll trainers)."""

import pickle
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytest

from mbrl.data import (
    DatasetInfo,
    EpisodeBatch,
    TrajectoryWindows,
    Transition,
    derive_train_val_split,
)
from mbrl.experiments import world_model as world_model_exp
from mbrl.logger import Logger
from mbrl.world_models.base import load_world_model_from_checkpoint
import mbrl.world_models.ensemble_mlp as ensemble_mlp_mod
from mbrl.world_models.ensemble_mlp import EnsembleMLP
from mbrl.world_models.term_stats import compute_model_discrepancy

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
        "max_logvar_init": 0.5, "min_logvar_init": -10.0,
        "log_interval": 2, "full_validation_interval": 4, "seed": 0,
    }
    return OmegaConf.create({**base, **overrides})


def _eggroll_cfg(**overrides) -> DictConfig:
    base = {
        "trainer": "eggroll", "num_ensemble": NUM_ENSEMBLE, "num_elites": NUM_ELITES,
        "hidden_dims": [8, 8], "activation": "relu", "init_scheme": "eggroll", "backbone": "mlp",
        "num_epochs": 15, "validation_split": 0.2, "logvar_diff_coef": 0.01,
        "max_logvar_init": 0.5, "min_logvar_init": -10.0,
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
    assert stds is not None and stds.shape == (num_elites, OBS_DIM + 1)
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

    def test_logs_lr(self, synthetic_dataset):
        cfg = _backprop_cfg()
        lrs: list[float] = []

        def log_fn(step, train_loss, val_mse, transitions_seen, forward_evals, **kw):
            if kw.get("lr") is not None:
                lrs.append(float(kw["lr"]))

        _train(cfg, synthetic_dataset, 0, log_fn=log_fn)
        # Both trainers must log lr; backprop's is the constant cfg.lr.
        assert lrs, "backprop logging never emitted lr"
        assert all(np.isclose(v, float(cfg.lr)) for v in lrs)


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

    def test_eggroll_to_eggroll(self, eggroll_model, synthetic_dataset, tmp_path):
        ckpt_path = _write_ckpt(eggroll_model, _eggroll_cfg(), tmp_path)
        cfg = _eggroll_cfg(init_checkpoint=str(ckpt_path), reset_optax_state=True, num_epochs=5)
        model = _train(cfg, synthetic_dataset, 6)
        _assert_inference_ok(model)

    def test_logvar_bounds_preserved(self, backprop_model, synthetic_dataset, tmp_path):
        """Warm-start loads the checkpoint params wholesale, so the learned
        max/min_logvar bounds survive the handoff."""
        ckpt_path = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        ckpt_max = backprop_model._params["max_logvar"]
        ckpt_min = backprop_model._params["min_logvar"]
        # num_epochs=0: the train loop is a no-op, so params == loaded checkpoint params.
        cfg = _eggroll_cfg(init_checkpoint=str(ckpt_path), reset_optax_state=True, num_epochs=0)
        model = _train(cfg, synthetic_dataset, 7)
        assert jnp.array_equal(model._params["max_logvar"], ckpt_max)
        assert jnp.array_equal(model._params["min_logvar"], ckpt_min)

    def test_val_split_parity_on_finetune(self, backprop_model, synthetic_dataset, tmp_path):
        """A fine-tune run reproduces the checkpoint's train/val split — keyed on the
        checkpoint's (validation_split, seed), not the fine-tune cfg's."""
        ckpt_path = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        # Checkpoint was trained with validation_split=0.2, seed=0.
        _, expected_val = derive_train_val_split(synthetic_dataset, 0.2, 0)

        captured: dict = {}
        real = ensemble_mlp_mod.derive_train_val_split

        def capturing(dataset, val_fraction, seed):
            tr, va = real(dataset, val_fraction, seed)
            captured["val"] = va
            return tr, va

        # Deliberately mismatched split knobs on the fine-tune cfg: the trainer must
        # ignore these and use the checkpoint's (0.2, 0).
        cfg = _backprop_cfg(
            init_checkpoint=str(ckpt_path), reset_optax_state=True,
            num_epochs=2, validation_split=0.4, seed=5,
        )
        with patch.object(ensemble_mlp_mod, "derive_train_val_split", capturing):
            _train(cfg, synthetic_dataset, 8)

        assert "val" in captured
        assert jnp.array_equal(captured["val"].obs, expected_val.obs)
        assert jnp.array_equal(captured["val"].next_obs, expected_val.next_obs)
        assert jnp.array_equal(captured["val"].reward, expected_val.reward)

    def test_finetune_records_source_split_seed(self, backprop_model, synthetic_dataset, tmp_path):
        """A fine-tune checkpoint records the source checkpoint's split seed/fraction
        (not the fine-tune cfg's), so a 2nd-gen fine-tune reproduces the same val split."""
        ckpt_path = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)  # seed=0, split=0.2
        cfg = _backprop_cfg(
            init_checkpoint=str(ckpt_path), reset_optax_state=True,
            num_epochs=2, seed=7, validation_split=0.4,
        )
        model = _train(cfg, synthetic_dataset, 8)
        state = model.checkpoint_state()
        assert state["seed"] == 0            # source split seed, not the fine-tune cfg seed 7
        assert state["validation_split"] == 0.2  # source split fraction, not 0.4


def _opt_count(model) -> int:
    """Largest scalar leaf in the optimiser state — the optax step counter."""
    leaves = [jnp.asarray(c) for c in jax.tree.leaves(model._opt_state)]
    scalars = [int(c) for c in leaves if c.ndim == 0]
    return max(scalars) if scalars else 0


class TestEnsembleMLPOptStateHandoff:
    def test_same_solver_carries_optimiser_step(self, backprop_model, synthetic_dataset, tmp_path):
        """Same-solver fine-tune (reset_optax_state=false) carries the optimiser step
        counter forward; resetting starts it from zero."""
        ckpt_path = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        carried = _train(
            _backprop_cfg(init_checkpoint=str(ckpt_path), reset_optax_state=False, num_epochs=2),
            synthetic_dataset, 5,
        )
        reset = _train(
            _backprop_cfg(init_checkpoint=str(ckpt_path), reset_optax_state=True, num_epochs=2),
            synthetic_dataset, 5,
        )
        assert _opt_count(carried) > _opt_count(reset)

    def test_solver_mismatch_requires_reset(self, backprop_model, synthetic_dataset, tmp_path):
        """An adamw checkpoint cannot carry into an sgd optimiser without a reset."""
        ckpt_path = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        cfg = _backprop_cfg(
            init_checkpoint=str(ckpt_path), reset_optax_state=False,
            optimizer="sgd", optimizer_kwargs={}, num_epochs=1,
        )
        with pytest.raises(AssertionError, match="opt_state structure mismatch"):
            _train(cfg, synthetic_dataset, 5)

    def test_solver_mismatch_ok_with_reset(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt_path = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        cfg = _backprop_cfg(
            init_checkpoint=str(ckpt_path), reset_optax_state=True,
            optimizer="sgd", optimizer_kwargs={}, num_epochs=2,
        )
        _assert_inference_ok(_train(cfg, synthetic_dataset, 5))

    def test_num_ensemble_mismatch_raises(self, backprop_model, synthetic_dataset, tmp_path):
        """init_checkpoint with a different num_ensemble is rejected — the param pytree
        structure is identical across num_ensemble, so only the shape guard catches it."""
        ckpt_path = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)  # num_ensemble=3
        cfg = _backprop_cfg(
            init_checkpoint=str(ckpt_path), reset_optax_state=True,
            num_ensemble=5, num_elites=3, num_epochs=1,
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            _train(cfg, synthetic_dataset, 5)


class TestEnsembleMLPEggrollConfig:
    def test_invalid_group_size_raises(self, synthetic_dataset):
        # group_size=3 is odd and does not divide population_size=8.
        cfg = _eggroll_cfg(eggroll={"group_size": 3})
        with pytest.raises(AssertionError):
            _train(cfg, synthetic_dataset, 0)

    def test_ungrouped_trains(self, synthetic_dataset):
        # group_size=0 is the ungrouped path (one perturbation per population member).
        _assert_inference_ok(_train(_eggroll_cfg(eggroll={"group_size": 0}), synthetic_dataset, 1))


class TestEnsembleMLPWarmStartGuards:
    """experiments.world_model.run rejects a fine-tune checkpoint from a different
    dataset or with mismatched obs/act dims (guards are world-model-class agnostic)."""

    def _run_with_ckpt(self, ckpt, synthetic_dataset, tmp_path):
        path = tmp_path / "stage1.pkl"
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
        run_cfg = OmegaConf.create({
            "seed": 0,
            "checkpoint_dir": str(tmp_path),
            "wandb": {"enabled": False},
            "dataset": {"name": DATASET_ID},
            "world_model": _backprop_cfg(init_checkpoint=str(path)),
        })
        info = DatasetInfo(
            obs_mean=jnp.zeros(OBS_DIM), obs_std=jnp.ones(OBS_DIM),
            obs_dim=OBS_DIM, act_dim=ACT_DIM, dataset_id=DATASET_ID,
        )
        logger = Logger(run_cfg)
        with patch(
            "mbrl.experiments.world_model.load_dataset",
            return_value=(synthetic_dataset, info),
        ):
            world_model_exp.run(run_cfg, logger)

    def test_dataset_id_mismatch_errors(self, synthetic_dataset, tmp_path):
        ckpt = {"dataset_id": "mujoco/hopper/medium-v0", "obs_dim": OBS_DIM, "act_dim": ACT_DIM}
        with pytest.raises(ValueError, match="dataset_id mismatch"):
            self._run_with_ckpt(ckpt, synthetic_dataset, tmp_path)

    def test_dims_mismatch_errors(self, synthetic_dataset, tmp_path):
        ckpt = {"dataset_id": DATASET_ID, "obs_dim": OBS_DIM + 1, "act_dim": ACT_DIM}
        with pytest.raises(ValueError, match="shape mismatch"):
            self._run_with_ckpt(ckpt, synthetic_dataset, tmp_path)


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


class TestTermStats:
    """MoReL halt-penalty stats are precomputed and survive a checkpoint round-trip.

    Stats work identically for both trainers (shared base implementation); the
    eggroll path is the same code, so backprop coverage here is sufficient.
    """

    def test_precompute_and_checkpoint(self, synthetic_dataset, tmp_path):
        model = _train(_backprop_cfg(), synthetic_dataset, 0)
        assert model.discrepancy is None and model.min_r is None

        model.precompute_term_stats(synthetic_dataset, jax.random.key(3))
        assert model.discrepancy is not None and model.discrepancy > 0
        assert jnp.isclose(model.min_r, float(jnp.min(synthetic_dataset.reward)))

        # Deterministic for a fixed key.
        d2 = compute_model_discrepancy(model, synthetic_dataset, jax.random.key(3))
        assert jnp.isclose(model.discrepancy, d2)

        # Checkpoint round-trip carries the stats.
        path = _write_ckpt(model, _backprop_cfg(), tmp_path)
        reloaded = EnsembleMLP.load_from_checkpoint(path)
        assert reloaded.discrepancy is not None and reloaded.min_r is not None
        assert jnp.isclose(reloaded.discrepancy, model.discrepancy)
        assert jnp.isclose(reloaded.min_r, model.min_r)

    def test_old_checkpoint_loads_none(self, backprop_model, tmp_path):
        """A checkpoint written before term stats existed lacks the keys → None."""
        ckpt = {
            **backprop_model.checkpoint_state(),
            "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "dataset_id": DATASET_ID,
            "world_model_cfg": OmegaConf.to_container(_backprop_cfg()), "wm_group": "g",
        }
        ckpt.pop("discrepancy", None)
        ckpt.pop("min_r", None)
        path = tmp_path / "old.pkl"
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
        reloaded = EnsembleMLP.load_from_checkpoint(path)
        assert reloaded.discrepancy is None and reloaded.min_r is None


# ── Trajectory (Phase-2) trainer ────────────────────────────────────────────────


def _toy_episodes(n_ep=8, length=5, seed=0) -> EpisodeBatch:
    rng = np.random.default_rng(seed)
    obs = jnp.asarray(rng.standard_normal((n_ep, length + 1, OBS_DIM)), jnp.float32)
    act = jnp.asarray(rng.standard_normal((n_ep, length, ACT_DIM)), jnp.float32)
    rew = jnp.asarray(rng.standard_normal((n_ep, length)), jnp.float32)
    mask = jnp.ones((n_ep, length), jnp.float32)
    return EpisodeBatch(obs, act, rew, mask, jnp.asarray([length] * n_ep, jnp.int32))


def _traj_cfg(**overrides) -> DictConfig:
    base = {
        "trainer": "eggroll_trajectory", "num_ensemble": NUM_ENSEMBLE, "num_elites": NUM_ELITES,
        "hidden_dims": [8, 8], "activation": "relu", "init_scheme": "eggroll", "backbone": "mlp",
        "max_logvar_init": 0.5, "min_logvar_init": -10.0,
        "init_checkpoint": None, "reset_optax_state": True,
        "num_epochs": 3, "horizon": 2, "curriculum": None, "batch_size": 2,
        "logvar_diff_coef": 0.01, "freeze_logvar_clamp": False,
        "trajectory_validation_split": 0.34, "val_transition_mse": True,
        "log_ensemble_disagreement": False,
        "log_interval": 1, "full_validation_interval": 2, "seed": 0,
        "eggroll": {
            "population_size": 4, "solver": "sgd", "solver_kwargs": {}, "noise_reuse": 1,
            "sigma": 0.02, "sigma_decay_rate": 0.99, "sigma_schedule": "exponential",
            "sigma_schedule_kwargs": {}, "lr": 1e-2, "lr_schedule": "constant",
            "lr_schedule_kwargs": {}, "use_batched_update": False,
        },
    }
    merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))
    assert isinstance(merged, DictConfig)
    return merged


def _train_traj(cfg, dataset, episodes, key, log_fn=None):
    model = EnsembleMLP(OBS_DIM, ACT_DIM, DATASET_ID, cfg)
    model.train(dataset, cfg, jax.random.key(key), log_fn=log_fn, episodes=episodes)
    jax.effects_barrier()
    return model


class TestEnsembleMLPTrajectory:
    def test_train_and_infer(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        model = _train_traj(
            _traj_cfg(init_checkpoint=str(ckpt)), synthetic_dataset, _toy_episodes(), 1
        )
        assert model._params is not None
        assert model._elite_idxs.shape == (NUM_ELITES,)
        assert model._update_steps_completed > 0
        _assert_inference_ok(model)

    def test_requires_init_checkpoint(self, synthetic_dataset):
        with pytest.raises(AssertionError, match="requires init_checkpoint"):
            _train_traj(_traj_cfg(init_checkpoint=None), synthetic_dataset, _toy_episodes(), 0)

    def test_population_must_be_even(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        cfg = _traj_cfg(init_checkpoint=str(ckpt), eggroll={"population_size": 3})
        with pytest.raises(AssertionError, match="even"):
            _train_traj(cfg, synthetic_dataset, _toy_episodes(), 0)

    def test_determinism(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        cfg = _traj_cfg(init_checkpoint=str(ckpt))
        eps = _toy_episodes()
        a = _train_traj(cfg, synthetic_dataset, eps, 3)
        b = _train_traj(cfg, synthetic_dataset, eps, 3)
        c = _train_traj(cfg, synthetic_dataset, eps, 9)
        # Same RNG key -> bit-identical evolved params across every leaf.
        assert all(
            jnp.array_equal(x, y)
            for x, y in zip(jax.tree.leaves(a._params), jax.tree.leaves(b._params))
        )
        # Different RNG key -> at least one leaf differs.
        assert any(
            not jnp.array_equal(x, y)
            for x, y in zip(jax.tree.leaves(a._params), jax.tree.leaves(c._params))
        )

    def test_records_both_splits(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)  # split 0.2, seed 0
        model = _train_traj(
            _traj_cfg(init_checkpoint=str(ckpt), trajectory_validation_split=0.34),
            synthetic_dataset, _toy_episodes(), 0,
        )
        state = model.checkpoint_state()
        assert state["validation_split"] == 0.2  # carried from the Phase-1 checkpoint
        assert state["trajectory_validation_split"] == 0.34  # episode split from cfg

    def test_transition_val_uses_phase1_set(self, backprop_model, synthetic_dataset, tmp_path):
        """The transition-level metric is computed on Phase-1's exact val set (reproduced
        from the checkpoint's validation_split/seed), not the flattened episode-val set."""
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)  # split 0.2, seed 0
        _, expected_val = derive_train_val_split(synthetic_dataset, 0.2, 0)

        captured: dict = {}
        real = ensemble_mlp_mod.derive_train_val_split

        def capturing(dataset, val_fraction, seed):
            tr, va = real(dataset, val_fraction, seed)
            captured["val"] = va
            return tr, va

        with patch.object(ensemble_mlp_mod, "derive_train_val_split", capturing):
            _train_traj(_traj_cfg(init_checkpoint=str(ckpt)), synthetic_dataset, _toy_episodes(), 0)
        assert jnp.array_equal(captured["val"].obs, expected_val.obs)
        assert jnp.array_equal(captured["val"].next_obs, expected_val.next_obs)

    def test_validation_is_mse_only(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        rows: list[dict] = []
        _train_traj(
            _traj_cfg(init_checkpoint=str(ckpt)), synthetic_dataset, _toy_episodes(), 0,
            log_fn=lambda gen, **kw: rows.append(kw),
        )
        keys = set().union(*(r.keys() for r in rows))
        assert "val_traj_mse" in keys and "val_traj_mse_elite" in keys
        assert "val_transition_mse" in keys and "val_transition_mse_elite" in keys
        # Training loss is logged as train_loss (Phase-1 naming); validation is MSE-only.
        assert "train_loss" in keys
        assert not any("nll" in k for k in keys)
        # The per-step curve is one reserved key (single line panel), not T h-scalars.
        assert not any(k.startswith("val_traj_mse_h") for k in keys)

    def test_logs_train_and_val_traj_curves(self, backprop_model, synthetic_dataset, tmp_path):
        """Both splits' compounding-error curves are logged — at the pre-finetune baseline
        (``*_init``, the Phase-1 model) and at the end of the fine-tune — as length-T lists."""
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        rows: list[dict] = []
        _train_traj(
            _traj_cfg(init_checkpoint=str(ckpt)), synthetic_dataset, _toy_episodes(), 0,
            log_fn=lambda gen, **kw: rows.append(kw),
        )
        first, last = rows[0], rows[-1]
        for k in ("val_traj_mse_curve_init", "train_traj_mse_curve_init"):
            assert len(first[k]) == 2  # horizon=2 -> one entry per rollout step
        for k in ("val_traj_mse_curve", "train_traj_mse_curve"):
            assert len(last[k]) == 2
            assert all(np.isfinite(v) for v in last[k])

    def test_val_transition_toggle_off(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        rows: list[dict] = []
        _train_traj(
            _traj_cfg(init_checkpoint=str(ckpt), val_transition_mse=False),
            synthetic_dataset, _toy_episodes(), 0, log_fn=lambda gen, **kw: rows.append(kw),
        )
        keys = set().union(*(r.keys() for r in rows))
        assert "val_traj_mse" in keys
        assert "val_transition_mse" not in keys

    def test_freeze_logvar_clamp(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        ckpt_max = backprop_model._params["max_logvar"]
        ckpt_min = backprop_model._params["min_logvar"]
        frozen = _train_traj(
            _traj_cfg(init_checkpoint=str(ckpt), freeze_logvar_clamp=True, num_epochs=5),
            synthetic_dataset, _toy_episodes(), 0,
        )
        assert jnp.array_equal(frozen._params["max_logvar"], ckpt_max)
        assert jnp.array_equal(frozen._params["min_logvar"], ckpt_min)
        # Unfrozen (default) DOES evolve the clamp bounds.
        unfrozen = _train_traj(
            _traj_cfg(init_checkpoint=str(ckpt), freeze_logvar_clamp=False, num_epochs=5),
            synthetic_dataset, _toy_episodes(), 0,
        )
        assert not jnp.array_equal(unfrozen._params["max_logvar"], ckpt_max)

    def test_use_mse_fitness_changes_optimisation(
        self, backprop_model, synthetic_dataset, tmp_path
    ):
        """The MSE-fitness diagnostic trains and infers, and with the same seed evolves
        params different from the NLL fitness — i.e. the fitness path actually changed."""
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        eps = _toy_episodes()
        nll = _train_traj(
            _traj_cfg(init_checkpoint=str(ckpt), use_mse_fitness=False, num_epochs=5),
            synthetic_dataset, eps, 0,
        )
        mse = _train_traj(
            _traj_cfg(init_checkpoint=str(ckpt), use_mse_fitness=True, num_epochs=5),
            synthetic_dataset, eps, 0,
        )
        _assert_inference_ok(mse)
        assert any(
            not jnp.array_equal(x, y)
            for x, y in zip(jax.tree.leaves(nll._params), jax.tree.leaves(mse._params))
        )

    def test_curriculum_runs_and_counts(self, backprop_model, synthetic_dataset, tmp_path):
        ckpt = _write_ckpt(backprop_model, _backprop_cfg(), tmp_path)
        step_offset = backprop_model._update_steps_completed
        cfg = _traj_cfg(
            init_checkpoint=str(ckpt),
            curriculum=[{"horizon": 2, "num_epochs": 2}, {"horizon": 3, "num_epochs": 3}],
        )
        model = _train_traj(cfg, synthetic_dataset, _toy_episodes(), 0)
        _assert_inference_ok(model)
        assert model._update_steps_completed == step_offset + 5

    def test_masking_invariant(self, backprop_model):
        """A padded (masked) trailing step does not change the rollout MSE — the same
        NaN-safe masking used by the fitness."""
        rng = np.random.default_rng(0)
        start = jnp.asarray(rng.standard_normal((1, OBS_DIM)), jnp.float32)
        acts = jnp.asarray(rng.standard_normal((1, 3, ACT_DIM)), jnp.float32)
        tobs = jnp.asarray(rng.standard_normal((1, 3, OBS_DIM)), jnp.float32)
        trew = jnp.asarray(rng.standard_normal((1, 3)), jnp.float32)
        w_full = TrajectoryWindows(start, acts, tobs, trew, jnp.asarray([[1.0, 1.0, 0.0]]))
        w_trunc = TrajectoryWindows(
            start, acts[:, :2], tobs[:, :2], trew[:, :2], jnp.asarray([[1.0, 1.0]])
        )
        s_full, _ = backprop_model._per_member_traj_mse(backprop_model._params, w_full)
        s_trunc, _ = backprop_model._per_member_traj_mse(backprop_model._params, w_trunc)
        assert jnp.allclose(s_full, s_trunc)

    def test_freeze_clamp_bounds_unperturbed_in_forward(self):
        """With freeze_clamp_bounds=True the noisy forward returns the clamp bounds at
        their base (Phase-1) values — not perturbed — even with a non-None iterinfo; the
        default perturbs them."""
        import optax

        from mbrl.eggroll.networks import DynamicsNet
        from mbrl.eggroll.primitives import EggRoll, simple_es_tree_key

        kw = dict(
            obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dims=[8, 8], activation="relu",
            init_scheme="eggroll", backbone="mlp", max_logvar_init=0.5, min_logvar_init=-10.0,
        )
        s = DynamicsNet.rand_init(jax.random.key(0), **kw)
        fnp, nps = EggRoll.init_noiser(s.params, 0.5, 1e-2, solver=optax.sgd, group_size=0)
        es_key = simple_es_tree_key(s.params, jax.random.key(1), s.scan_map)
        it = (jnp.int32(0), jnp.int32(1))  # non-None -> default path perturbs
        o, a = jnp.zeros(OBS_DIM), jnp.zeros(ACT_DIM)
        args = (EggRoll, fnp, nps, s.frozen_params, s.params, es_key, it, o, a)
        *_, max_frozen, min_frozen = DynamicsNet._forward_noisy_with_bounds(
            *args, freeze_clamp_bounds=True
        )
        *_, max_pert, min_pert = DynamicsNet._forward_noisy_with_bounds(
            *args, freeze_clamp_bounds=False
        )
        assert jnp.array_equal(max_frozen, s.params["max_logvar"])
        assert jnp.array_equal(min_frozen, s.params["min_logvar"])
        # The default (unfrozen) path actually perturbs the bounds.
        assert not jnp.array_equal(max_pert, s.params["max_logvar"])
        assert not jnp.array_equal(min_pert, s.params["min_logvar"])


class TestTrajectoryCombinedPipeline:
    def test_finetune_true_runs_two_phases(self, synthetic_dataset, tmp_path):
        """world_model.finetune=true runs backprop then trajectory in one invocation,
        auto-handing off the Phase-1 checkpoint and forcing Phase-2 arch to match it."""
        info = DatasetInfo(
            obs_mean=jnp.zeros(OBS_DIM), obs_std=jnp.ones(OBS_DIM),
            obs_dim=OBS_DIM, act_dim=ACT_DIM, dataset_id=DATASET_ID,
        )
        target = "mbrl.world_models.ensemble_mlp.EnsembleMLP"
        bp = _backprop_cfg()
        OmegaConf.update(bp, "_target_", target, force_add=True)
        bp.finetune = True
        # Phase-2 arch deliberately mismatched; the experiment must force it to Phase-1's.
        ft = _traj_cfg(num_ensemble=5, num_elites=4)
        OmegaConf.update(ft, "_target_", target, force_add=True)
        run_cfg = OmegaConf.create({
            "seed": 0, "checkpoint_dir": str(tmp_path), "wandb": {"enabled": False},
            "dataset": {"name": DATASET_ID}, "world_model": bp, "finetune_world_model": ft,
        })
        logger = Logger(run_cfg)
        with (
            patch("mbrl.experiments.world_model.load_dataset",
                  return_value=(synthetic_dataset, info)),
            patch("mbrl.experiments.world_model.load_episodes",
                  return_value=(_toy_episodes(), info)),
        ):
            world_model_exp.run(run_cfg, logger)

        assert (tmp_path / "world_model_phase1.pkl").exists()
        final = load_world_model_from_checkpoint(str(tmp_path / "world_model.pkl"))
        assert final.num_ensemble == NUM_ENSEMBLE  # forced from Phase-1, not the ft cfg's 5
        _assert_inference_ok(final, num_elites=NUM_ELITES)


@pytest.mark.slow
def test_trajectory_finetune_reduces_compounding_error(tmp_path):
    """On an over-fittable toy (one simple deterministic system), Phase-2 reduces the
    multi-step rollout MSE versus the Phase-1 checkpoint."""
    obs_dim, act_dim, n_ep, length = 2, 1, 12, 4
    rng = np.random.default_rng(0)
    a_mat = np.array([[0.9, 0.1], [-0.1, 0.8]], np.float32)
    b_mat = np.array([[0.2], [0.3]], np.float32)
    obs = np.zeros((n_ep, length + 1, obs_dim), np.float32)
    act = rng.standard_normal((n_ep, length, act_dim)).astype(np.float32)
    rew = np.zeros((n_ep, length), np.float32)
    obs[:, 0] = rng.standard_normal((n_ep, obs_dim))
    for k in range(length):
        delta = obs[:, k] @ a_mat.T * 0.1 + act[:, k] @ b_mat.T
        obs[:, k + 1] = obs[:, k] + delta
        rew[:, k] = obs[:, k + 1].sum(axis=1)
    episodes = EpisodeBatch(
        jnp.asarray(obs), jnp.asarray(act), jnp.asarray(rew),
        jnp.ones((n_ep, length), jnp.float32), jnp.asarray([length] * n_ep, jnp.int32),
    )
    flat = Transition(
        obs=jnp.asarray(obs[:, :-1].reshape(-1, obs_dim)),
        action=jnp.asarray(act.reshape(-1, act_dim)),
        reward=jnp.asarray(rew.reshape(-1)),
        next_obs=jnp.asarray(obs[:, 1:].reshape(-1, obs_dim)),
        done=jnp.zeros((n_ep * length,), jnp.float32),
    )

    arch = dict(
        num_ensemble=2, num_elites=1, hidden_dims=[32, 32], activation="relu",
        init_scheme="eggroll", backbone="mlp", max_logvar_init=0.5, min_logvar_init=-10.0,
    )
    bp_cfg = OmegaConf.create({
        "trainer": "backprop", **arch, "num_epochs": 30, "batch_size": 16, "lr": 1e-3,
        "optimizer": "adamw", "optimizer_kwargs": {"eps": 1e-5}, "validation_split": 0.2,
        "logvar_diff_coef": 0.01, "log_interval": 50, "full_validation_interval": 50, "seed": 0,
    })
    m1 = EnsembleMLP(obs_dim, act_dim, DATASET_ID, bp_cfg)
    m1.train(flat, bp_cfg, jax.random.key(0))
    jax.effects_barrier()
    ckpt = {
        **m1.checkpoint_state(), "obs_dim": obs_dim, "act_dim": act_dim,
        "dataset_id": DATASET_ID, "world_model_cfg": OmegaConf.to_container(bp_cfg),
    }
    ckpt_path = tmp_path / "phase1.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)

    ft_cfg = OmegaConf.create({
        "trainer": "eggroll_trajectory", **arch, "init_checkpoint": str(ckpt_path),
        "reset_optax_state": True, "num_epochs": 600, "horizon": 4, "curriculum": None,
        "batch_size": 8, "logvar_diff_coef": 0.01, "freeze_logvar_clamp": False,
        "trajectory_validation_split": 0.25, "val_transition_mse": False,
        "log_ensemble_disagreement": False, "log_interval": 100,
        "full_validation_interval": 100, "seed": 0,
        "eggroll": {
            "population_size": 64, "solver": "sgd", "solver_kwargs": {}, "noise_reuse": 1,
            "sigma": 0.03, "sigma_decay_rate": 0.995, "sigma_schedule": "exponential",
            "sigma_schedule_kwargs": {}, "lr": 0.02, "lr_schedule": "constant",
            "lr_schedule_kwargs": {}, "use_batched_update": False,
        },
    })
    # Initial (Phase-1) trajectory MSE on the held-out episodes, via a model loaded
    # straight from the Phase-1 checkpoint (inference structure + params set on load).
    from mbrl.data import derive_episode_train_val_split, tile_episodes_to_windows

    _, val_eps = derive_episode_train_val_split(episodes, 0.25, 0)
    val_windows = tile_episodes_to_windows(val_eps, 4)
    m_init = EnsembleMLP.load_from_checkpoint(ckpt_path)
    initial = float(m_init._per_member_traj_mse(m_init._params, val_windows)[0].min())

    m2 = EnsembleMLP(obs_dim, act_dim, DATASET_ID, ft_cfg)
    m2.train(flat, ft_cfg, jax.random.key(1), episodes=episodes)
    jax.effects_barrier()
    final = float(m2._per_member_traj_mse(m2._params, val_windows)[0].min())
    assert final < initial, f"trajectory MSE did not improve: {initial=} {final=}"


def _assert_deterministic_inference_ok(model, num_elites=NUM_ELITES):
    """Inference checks for a no-logvar model: predict_ensemble returns std=None."""
    obs, action = jnp.zeros(OBS_DIM), jnp.zeros(ACT_DIM)
    means, stds = model.predict_ensemble(obs, action)
    assert means.shape == (num_elites, OBS_DIM + 1)
    assert stds is None
    assert jnp.all(jnp.isfinite(means))
    next_obs, reward, done = model.step(obs, action, jax.random.key(7))
    assert next_obs.shape == (OBS_DIM,) and reward.shape == () and done.shape == ()
    assert jnp.all(jnp.isfinite(next_obs)) and jnp.isfinite(reward)


class TestDisableLogvar:
    def test_no_clamp_params_and_smaller_head(self, synthetic_dataset):
        """Deterministic head drops the logvar half and the clamp Parameters."""
        model = _train(_backprop_cfg(disable_logvar_predictions=True), synthetic_dataset, 0)
        assert model.predicts_logvar is False
        assert "max_logvar" not in model._params and "min_logvar" not in model._params
        # Mean head only: backbone output width is obs_dim+1, not 2*(obs_dim+1).
        means, _ = model.predict_ensemble(jnp.zeros(OBS_DIM), jnp.zeros(ACT_DIM))
        assert means.shape == (NUM_ELITES, OBS_DIM + 1)

    def test_backprop_trains_and_infers(self, synthetic_dataset):
        model = _train(_backprop_cfg(disable_logvar_predictions=True), synthetic_dataset, 0)
        _assert_deterministic_inference_ok(model)

    def test_eggroll_trains_and_infers(self, synthetic_dataset):
        model = _train(_eggroll_cfg(disable_logvar_predictions=True), synthetic_dataset, 0)
        _assert_deterministic_inference_ok(model)

    def test_step_is_member_resampling_only(self, synthetic_dataset):
        """No aleatoric noise: with a single elite the member choice is fixed too, so
        step is fully deterministic regardless of the rng (the noise term is dropped)."""
        obs, action = jnp.ones(OBS_DIM), jnp.ones(ACT_DIM) * 0.5
        single = _train(
            _backprop_cfg(disable_logvar_predictions=True, num_elites=1), synthetic_dataset, 0
        )
        a, _, _ = single.step(obs, action, jax.random.key(1))
        b, _, _ = single.step(obs, action, jax.random.key(99))
        assert jnp.array_equal(a, b)  # deterministic given one elite (no aleatoric noise)

    def test_conflict_with_use_mse_fitness(self):
        with pytest.raises(ValueError, match="incompatible"):
            EnsembleMLP(
                OBS_DIM, ACT_DIM, DATASET_ID,
                _traj_cfg(disable_logvar_predictions=True, use_mse_fitness=True),
            )

    def test_conflict_with_freeze_clamp(self):
        with pytest.raises(ValueError, match="incompatible"):
            EnsembleMLP(
                OBS_DIM, ACT_DIM, DATASET_ID,
                _traj_cfg(disable_logvar_predictions=True, freeze_logvar_clamp=True),
            )

    def test_checkpoint_roundtrip_preserves_flag(self, synthetic_dataset, tmp_path):
        cfg = _backprop_cfg(disable_logvar_predictions=True)
        model = _train(cfg, synthetic_dataset, 0)
        path = _write_ckpt(model, cfg, tmp_path)
        reloaded = EnsembleMLP.load_from_checkpoint(path)
        assert reloaded.predicts_logvar is False
        ma, sa = model.predict_ensemble(jnp.zeros(OBS_DIM), jnp.zeros(ACT_DIM))
        mb, sb = reloaded.predict_ensemble(jnp.zeros(OBS_DIM), jnp.zeros(ACT_DIM))
        assert jnp.allclose(ma, mb) and sa is None and sb is None

    def test_trajectory_finetune_no_logvar(self, synthetic_dataset, tmp_path):
        """A deterministic Phase-1 model fine-tunes at trajectory level (MSE forced)."""
        cfg1 = _backprop_cfg(disable_logvar_predictions=True)
        ckpt = _write_ckpt(_train(cfg1, synthetic_dataset, 0), cfg1, tmp_path)
        ft = _traj_cfg(init_checkpoint=str(ckpt), disable_logvar_predictions=True)
        model = _train_traj(ft, synthetic_dataset, _toy_episodes(), 1)
        assert model.predicts_logvar is False
        _assert_deterministic_inference_ok(model)
