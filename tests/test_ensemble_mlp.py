"""Tests for the unified EnsembleMLP world model (backprop + eggroll trainers)."""

import pickle
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytest

from mbrl.data import DatasetInfo, Transition, derive_train_val_split
from mbrl.experiments import world_model as world_model_exp
from mbrl.logger import Logger
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
