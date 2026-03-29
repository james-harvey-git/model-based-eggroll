"""Tests for MOPO policy training (policy_optimizers/mopo.py)."""

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from mbrl.data import Transition
from mbrl.policy_optimizers.mopo import make_mopo_step, make_rollout_fn, train
from mbrl.policy_optimizers.sac_n import AgentTrainState, create_agent_state, make_sac_update
from mbrl.world_models.mle import MLEEnsemble

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
    "penalty_coeff": 1.0,
    "rollout_length": 2,
    "num_policy_updates": 4,
    "rollout_interval": 2,
    "rollout_batch_size": 16,
    "model_retain_epochs": 2,
    "dataset_sample_ratio": 0.25,
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
    model = MLEEnsemble(OBS_DIM, ACT_DIM, "mujoco/halfcheetah/medium-v0", WM_CFG)
    model.train(synthetic_dataset, WM_CFG, jax.random.key(0))
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


class TestPredictEnsemble:
    def test_shape(self, trained_world_model):
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        mean, std = trained_world_model.predict_ensemble(obs, action)
        num_elites = trained_world_model.num_elites
        assert mean.shape == (num_elites, OBS_DIM + 1)
        assert std.shape == (num_elites, OBS_DIM + 1)

    def test_std_positive(self, trained_world_model):
        obs = jnp.ones(OBS_DIM)
        action = jnp.ones(ACT_DIM) * 0.5
        _, std = trained_world_model.predict_ensemble(obs, action)
        assert jnp.all(std > 0)


class TestRolloutBuffer:
    def test_allocation_shapes(self, synthetic_dataset):
        buf = jax.tree.map(
            lambda x: jnp.zeros((MAX_BUFFER_SIZE, *x.shape[1:])),
            synthetic_dataset,
        )
        assert buf.obs.shape == (MAX_BUFFER_SIZE, OBS_DIM)
        assert buf.action.shape == (MAX_BUFFER_SIZE, ACT_DIM)
        assert buf.reward.shape == (MAX_BUFFER_SIZE,)
        assert buf.next_obs.shape == (MAX_BUFFER_SIZE, OBS_DIM)
        assert buf.done.shape == (MAX_BUFFER_SIZE,)


class TestRolloutFn:
    def test_fills_buffer(self, trained_world_model, synthetic_dataset, agent_state, zero_buffer):
        rollout_fn = make_rollout_fn(
            trained_world_model, agent_state.actor.apply_fn, synthetic_dataset.obs, FAST_CFG
        )
        new_buf = rollout_fn(jax.random.key(42), agent_state.actor.params, zero_buffer)
        assert jnp.any(new_buf.obs != 0)
        assert jnp.any(new_buf.reward != 0)

    def test_shapes_preserved(
        self, trained_world_model, synthetic_dataset, agent_state, zero_buffer
    ):
        rollout_fn = make_rollout_fn(
            trained_world_model, agent_state.actor.apply_fn, synthetic_dataset.obs, FAST_CFG
        )
        new_buf = rollout_fn(jax.random.key(42), agent_state.actor.params, zero_buffer)
        assert new_buf.obs.shape == zero_buffer.obs.shape
        assert new_buf.reward.shape == zero_buffer.reward.shape
        assert new_buf.done.shape == zero_buffer.done.shape

    def test_fifo_insertion(self, trained_world_model, synthetic_dataset, agent_state, zero_buffer):
        """Second rollout replaces oldest entries; first rollout's data is partially retained."""
        rollout_fn = make_rollout_fn(
            trained_world_model, agent_state.actor.apply_fn, synthetic_dataset.obs, FAST_CFG
        )
        num_new = FAST_CFG.rollout_batch_size * FAST_CFG.rollout_length  # 32

        buf1 = rollout_fn(jax.random.key(1), agent_state.actor.params, zero_buffer)
        buf2 = rollout_fn(jax.random.key(2), agent_state.actor.params, buf1)

        # FIFO: buf2 = concat(buf1[:-num_new], new_rollouts).
        # The first (buffer_size - num_new) entries are preserved from buf1.
        assert jnp.array_equal(buf2.obs[:-num_new], buf1.obs[:-num_new])
        # The last num_new entries are the new rollouts (different from buf1's last entries).
        assert not jnp.array_equal(buf2.obs[-num_new:], buf1.obs[-num_new:])

    def test_penalty_reduces_reward(
        self, trained_world_model, synthetic_dataset, agent_state, zero_buffer
    ):
        """Higher penalty_coeff should produce lower rollout rewards."""
        base = OmegaConf.to_container(FAST_CFG)
        cfg_no_penalty = OmegaConf.create({**base, "penalty_coeff": 0.0})  # type: ignore[arg-type]
        cfg_hi_penalty = OmegaConf.create({**base, "penalty_coeff": 5.0})  # type: ignore[arg-type]

        rollout_no = make_rollout_fn(
            trained_world_model, agent_state.actor.apply_fn, synthetic_dataset.obs, cfg_no_penalty
        )
        rollout_hi = make_rollout_fn(
            trained_world_model, agent_state.actor.apply_fn, synthetic_dataset.obs, cfg_hi_penalty
        )
        rng = jax.random.key(7)
        buf_no = rollout_no(rng, agent_state.actor.params, zero_buffer)
        buf_hi = rollout_hi(rng, agent_state.actor.params, zero_buffer)

        # num_new entries were written (rest are zeros); compare only the new entries.
        num_new = FAST_CFG.rollout_batch_size * FAST_CFG.rollout_length
        mean_no = buf_no.reward[-num_new:].mean()
        mean_hi = buf_hi.reward[-num_new:].mean()
        assert mean_hi < mean_no


class TestMopoStep:
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
        mopo_step = make_mopo_step(sac_update, rollout_fn, synthetic_dataset, FAST_CFG)

        runner_state = (jax.random.key(99), agent_state, zero_buffer)
        _, metrics = mopo_step(runner_state, None)

        expected_keys = {
            "critic_loss", "actor_loss", "alpha_loss", "entropy", "alpha", "q_min", "q_std"
        }
        assert set(metrics.keys()) == expected_keys
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' is not finite: {v}"

    def test_updates_actor_params(
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
        mopo_step = make_mopo_step(sac_update, rollout_fn, synthetic_dataset, FAST_CFG)

        runner_state = (jax.random.key(99), agent_state, zero_buffer)
        (_, new_agent_state, _), _ = mopo_step(runner_state, None)

        changed = jax.tree.map(
            lambda a, b: ~jnp.array_equal(a, b),
            agent_state.actor.params,
            new_agent_state.actor.params,
        )
        assert any(jax.tree.leaves(changed))


class TestMopoTrainE2E:
    def test_returns_agent_state(self, trained_world_model, synthetic_dataset):
        result = train(trained_world_model, synthetic_dataset, FAST_CFG, jax.random.key(0))
        assert isinstance(result, AgentTrainState)

    def test_actor_step_incremented(self, trained_world_model, synthetic_dataset):
        result = train(trained_world_model, synthetic_dataset, FAST_CFG, jax.random.key(0))
        assert result.actor.step == FAST_CFG.num_policy_updates
