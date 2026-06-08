"""Tests for MoReL policy training (policy_optimizers/morel.py)."""

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from mbrl.data import Transition
from mbrl.policy_optimizers.morel import (
    extract_actor,
    make_morel_step,
    make_rollout_fn,
    make_train_step,
    train,
)
from mbrl.policy_optimizers.sac_n import AgentTrainState, create_agent_state, make_sac_update
from mbrl.world_models.unifloral_ensemble_mlp import UnifloralEnsembleMLP

OBS_DIM = 4
ACT_DIM = 2
N = 256

WM_CFG = OmegaConf.create({
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
})

FAST_CFG = OmegaConf.create({
    "lr": 1e-4,
    "batch_size": 32,
    "gamma": 0.99,
    "polyak_step_size": 0.005,
    "num_critics": 2,
    "rollout_length": 2,
    "num_policy_updates": 4,
    "rollout_interval": 2,
    "rollout_batch_size": 16,
    "model_retain_epochs": 2,
    "dataset_sample_ratio": 0.25,
    "eval_interval": 2,
    "threshold_coef": 1.0,
    "term_penalty_offset": -200.0,
})

# Max buffer size derived from fast config: 16 * 2 * 2 = 64
MAX_BUFFER_SIZE = (
    FAST_CFG.rollout_batch_size * FAST_CFG.rollout_length * FAST_CFG.model_retain_epochs
)


@pytest.fixture(scope="module")
def synthetic_dataset():
    rng = np.random.default_rng(0)
    obs = jnp.array(rng.standard_normal((N, OBS_DIM)), dtype=jnp.float32)
    action = jnp.array(rng.uniform(-1.0, 1.0, (N, ACT_DIM)), dtype=jnp.float32)
    reward = jnp.array(rng.standard_normal(N), dtype=jnp.float32)
    next_obs = jnp.array(rng.standard_normal((N, OBS_DIM)), dtype=jnp.float32)
    done = jnp.zeros(N, dtype=jnp.float32)
    return Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)


@pytest.fixture(scope="module")
def trained_world_model(synthetic_dataset):
    """World model with precomputed MoReL term stats — required by MoReL."""
    model = UnifloralEnsembleMLP(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", WM_CFG)
    model.train(synthetic_dataset, WM_CFG, jax.random.key(0))
    model.precompute_term_stats(synthetic_dataset, jax.random.key(1))
    return model


@pytest.fixture(scope="module")
def agent_state():
    return create_agent_state(OBS_DIM, ACT_DIM, FAST_CFG, jax.random.key(1))


@pytest.fixture(scope="module")
def zero_buffer(synthetic_dataset):
    return jax.tree.map(
        lambda x: jnp.zeros((MAX_BUFFER_SIZE, *x.shape[1:])),
        synthetic_dataset,
    )


class TestTermStatsRequired:
    def test_missing_stats_raises(self, synthetic_dataset, agent_state):
        """MoReL must refuse a world model without precomputed term stats."""
        model = UnifloralEnsembleMLP(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", WM_CFG)
        model.train(synthetic_dataset, WM_CFG, jax.random.key(0))
        assert model.discrepancy is None  # not precomputed
        with pytest.raises(AssertionError, match="precomputed term stats"):
            make_rollout_fn(
                model, agent_state.actor.apply_fn, synthetic_dataset.obs, FAST_CFG
            )


class TestRolloutFn:
    def test_fills_buffer(self, trained_world_model, synthetic_dataset, agent_state, zero_buffer):
        rollout_fn = make_rollout_fn(
            trained_world_model, agent_state.actor.apply_fn, synthetic_dataset.obs, FAST_CFG
        )
        new_buf = rollout_fn(jax.random.key(42), agent_state.actor.params, zero_buffer)
        assert jnp.any(new_buf.obs != 0)
        assert new_buf.obs.shape == zero_buffer.obs.shape
        assert new_buf.reward.shape == zero_buffer.reward.shape

    def test_halt_penalty_fires(
        self, trained_world_model, synthetic_dataset, agent_state, zero_buffer
    ):
        """threshold_coef=0 → every state-action is 'uncertain' → halt reward + done."""
        base = OmegaConf.to_container(FAST_CFG)
        cfg = OmegaConf.create({**base, "threshold_coef": 0.0})  # type: ignore[arg-type]
        rollout_fn = make_rollout_fn(
            trained_world_model, agent_state.actor.apply_fn, synthetic_dataset.obs, cfg
        )
        new_buf = rollout_fn(jax.random.key(7), agent_state.actor.params, zero_buffer)

        num_new = FAST_CFG.rollout_batch_size * FAST_CFG.rollout_length
        halt_reward = float(trained_world_model.min_r) + cfg.term_penalty_offset
        # All freshly written transitions are halted: reward floored, done set.
        assert jnp.allclose(new_buf.reward[-num_new:], halt_reward)
        assert jnp.all(new_buf.done[-num_new:] == 1.0)

    def test_halt_penalty_does_not_fire(
        self, trained_world_model, synthetic_dataset, agent_state, zero_buffer
    ):
        """Huge threshold → disagreement never exceeds it → no halt, model rewards kept."""
        base = OmegaConf.to_container(FAST_CFG)
        cfg = OmegaConf.create({**base, "threshold_coef": 1e6})  # type: ignore[arg-type]
        rollout_fn = make_rollout_fn(
            trained_world_model, agent_state.actor.apply_fn, synthetic_dataset.obs, cfg
        )
        new_buf = rollout_fn(jax.random.key(7), agent_state.actor.params, zero_buffer)

        num_new = FAST_CFG.rollout_batch_size * FAST_CFG.rollout_length
        halt_reward = float(trained_world_model.min_r) + cfg.term_penalty_offset
        # No transition was halted: rewards are model samples (not the constant floor),
        # and the halfcheetah termination_fn never terminates.
        assert not jnp.allclose(new_buf.reward[-num_new:], halt_reward)
        assert jnp.all(new_buf.done[-num_new:] == 0.0)


class TestMorelStep:
    def test_metrics_keys_and_finite(
        self, trained_world_model, synthetic_dataset, agent_state, zero_buffer
    ):
        sac_update = make_sac_update(
            agent_state.actor.apply_fn,
            agent_state.vec_q.apply_fn,
            agent_state.alpha.apply_fn,
            FAST_CFG,
        )
        rollout_fn = make_rollout_fn(
            trained_world_model, agent_state.actor.apply_fn, synthetic_dataset.obs, FAST_CFG
        )
        morel_step = make_morel_step(sac_update, rollout_fn, synthetic_dataset, FAST_CFG)

        runner_state = (jax.random.key(99), agent_state, zero_buffer)
        _, metrics = morel_step(runner_state, None)

        expected_keys = {
            "critic_loss", "actor_loss", "alpha_loss", "entropy", "alpha", "q_min", "q_std"
        }
        assert set(metrics.keys()) == expected_keys
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' is not finite: {v}"


class TestMorelTrainE2E:
    def test_returns_agent_state(self, trained_world_model, synthetic_dataset):
        result = train(trained_world_model, synthetic_dataset, FAST_CFG, jax.random.key(0))
        assert isinstance(result, AgentTrainState)

    def test_actor_step_incremented(self, trained_world_model, synthetic_dataset):
        result = train(trained_world_model, synthetic_dataset, FAST_CFG, jax.random.key(0))
        assert result.actor.step == FAST_CFG.num_policy_updates


class TestMakeTrainStep:
    def test_returns_callable_and_state(self, trained_world_model, synthetic_dataset):
        step_fn, runner_state = make_train_step(
            trained_world_model, synthetic_dataset, FAST_CFG, jax.random.key(0)
        )
        _, agent_state, rollout_buffer = runner_state
        assert callable(step_fn)
        assert isinstance(agent_state, AgentTrainState)
        assert rollout_buffer.obs.shape == (MAX_BUFFER_SIZE, OBS_DIM)

    def test_one_step_advances_actor(self, trained_world_model, synthetic_dataset):
        step_fn, runner_state = make_train_step(
            trained_world_model, synthetic_dataset, FAST_CFG, jax.random.key(0)
        )
        new_runner_state, _ = step_fn(runner_state, None)
        _, new_agent_state, _ = new_runner_state
        assert int(new_agent_state.actor.step) == 1


class TestExtractActor:
    def test_returns_params_and_step(self, trained_world_model, synthetic_dataset):
        _, runner_state = make_train_step(
            trained_world_model, synthetic_dataset, FAST_CFG, jax.random.key(0)
        )
        actor_params, step = extract_actor(runner_state)
        assert isinstance(actor_params, dict)
        assert step == 0


# Deterministic (no-logvar) EnsembleMLP: MoReL's penalty (mean-disagreement) and term
# stats are mean-only, so they must work with no aleatoric head.
DET_WM_CFG = OmegaConf.create({
    "trainer": "backprop", "num_ensemble": 3, "num_elites": 2,
    "hidden_dims": [32, 32], "activation": "relu", "init_scheme": "eggroll",
    "backbone": "mlp", "num_epochs": 3, "batch_size": 32, "lr": 1e-3,
    "optimizer": "adamw", "optimizer_kwargs": {}, "validation_split": 0.2,
    "log_interval": 2, "full_validation_interval": 3, "seed": 0,
    "disable_logvar_predictions": True,
})


@pytest.fixture(scope="module")
def deterministic_world_model(synthetic_dataset):
    from mbrl.world_models.ensemble_mlp import EnsembleMLP

    model = EnsembleMLP(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", DET_WM_CFG)
    model.train(synthetic_dataset, DET_WM_CFG, jax.random.key(0))
    model.precompute_term_stats(synthetic_dataset, jax.random.key(1))
    jax.effects_barrier()
    return model


class TestDeterministicWorldModel:
    def test_term_stats_precompute(self, deterministic_world_model, synthetic_dataset):
        """discrepancy/min_r are mean-only, so they compute fine with no logvar head."""
        assert deterministic_world_model.predicts_logvar is False
        assert deterministic_world_model.discrepancy is not None
        assert deterministic_world_model.discrepancy > 0  # members disagree
        assert jnp.isclose(
            deterministic_world_model.min_r, float(jnp.min(synthetic_dataset.reward))
        )

    def test_rollout_fills_buffer(
        self, deterministic_world_model, synthetic_dataset, agent_state, zero_buffer
    ):
        """MoReL rollout (std=None sampling) fills the buffer with finite rewards."""
        rollout_fn = make_rollout_fn(
            deterministic_world_model, agent_state.actor.apply_fn,
            synthetic_dataset.obs, FAST_CFG,
        )
        new_buf = rollout_fn(jax.random.key(42), agent_state.actor.params, zero_buffer)
        assert jnp.any(new_buf.obs != 0)
        assert jnp.all(jnp.isfinite(new_buf.reward))
