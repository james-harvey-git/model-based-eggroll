"""Tests for SAC-N shared building block (policy_optimizers/sac_n.py)."""

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

from mbrl.data import Transition
from mbrl.policy_optimizers.sac_n import (
    AgentTrainState,
    EntropyCoef,
    SoftQNetwork,
    TanhGaussianActor,
    VectorQ,
    create_agent_state,
    make_sac_update,
)

OBS_DIM = 4
ACT_DIM = 2
NUM_CRITICS = 2
BATCH_SIZE = 32

FAST_CFG = OmegaConf.create(
    {"lr": 1e-4, "gamma": 0.99, "polyak_step_size": 0.005, "num_critics": NUM_CRITICS}
)


@pytest.fixture(scope="module")
def synthetic_batch():
    rng = np.random.default_rng(0)
    n = BATCH_SIZE
    obs = jnp.array(rng.standard_normal((n, OBS_DIM)), dtype=jnp.float32)
    action = jnp.array(rng.uniform(-1.0, 1.0, (n, ACT_DIM)), dtype=jnp.float32)
    reward = jnp.array(rng.standard_normal(n), dtype=jnp.float32)
    next_obs = jnp.array(rng.standard_normal((n, OBS_DIM)), dtype=jnp.float32)
    done = jnp.zeros(n, dtype=jnp.float32)
    return Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)


@pytest.fixture(scope="module")
def agent_state():
    return create_agent_state(OBS_DIM, ACT_DIM, FAST_CFG, jax.random.key(0))


class TestNetworkShapes:
    def test_soft_q_output_shape(self):
        net = SoftQNetwork()
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        params = net.init(jax.random.key(0), obs, action)
        out = net.apply(params, obs, action)
        assert out.shape == () # type: ignore

    def test_vector_q_output_shape(self):
        net = VectorQ(num_critics=NUM_CRITICS)
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        params = net.init(jax.random.key(0), obs, action)
        out = net.apply(params, obs, action)
        assert out.shape == (NUM_CRITICS,) # type: ignore

    def test_actor_distribution_shape(self):
        net = TanhGaussianActor(ACT_DIM)
        obs = jnp.zeros(OBS_DIM)
        params = net.init(jax.random.key(0), obs)
        pi = net.apply(params, obs)
        sample, log_prob = pi.sample_and_log_prob(seed=jax.random.key(1)) # type: ignore
        assert sample.shape == (ACT_DIM,)
        assert log_prob.shape == (ACT_DIM,)

    def test_entropy_coef_init(self):
        net = EntropyCoef()
        params = net.init(jax.random.key(0))
        log_coef = net.apply(params)
        assert jnp.allclose(jnp.exp(log_coef), jnp.ones(()), atol=1e-6) # type: ignore


class TestAgentState:
    def test_create_agent_state_fields(self, agent_state):
        assert isinstance(agent_state, AgentTrainState)
        assert isinstance(agent_state.actor, TrainState)
        assert isinstance(agent_state.vec_q, TrainState)
        assert isinstance(agent_state.vec_q_target, TrainState)
        assert isinstance(agent_state.alpha, TrainState)

    def test_target_q_matches_online_q_at_init(self, agent_state):
        leaves_equal = jax.tree.map(
            lambda a, b: jnp.array_equal(a, b),
            agent_state.vec_q.params,
            agent_state.vec_q_target.params,
        )
        assert all(jax.tree.leaves(leaves_equal))


class TestSacUpdate:
    def test_sac_update_step(self, agent_state, synthetic_batch):
        sac_update = make_sac_update(
            agent_state.actor.apply_fn,
            agent_state.vec_q.apply_fn,
            agent_state.alpha.apply_fn,
            FAST_CFG,
        )
        rng = jax.random.key(42)
        (_rng, new_state), metrics = sac_update(rng, agent_state, synthetic_batch)

        # Metrics structure
        assert set(metrics.keys()) == {
            "critic_loss", "actor_loss", "alpha_loss",
            "entropy", "alpha", "q_min", "q_std",
        }

        # All metrics finite
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' is not finite: {v}"

        # Actor params changed after gradient step
        actor_leaves_changed = jax.tree.map(
            lambda a, b: ~jnp.array_equal(a, b),
            agent_state.actor.params,
            new_state.actor.params,
        )
        assert any(jax.tree.leaves(actor_leaves_changed))

        # Critic params changed after gradient step
        critic_leaves_changed = jax.tree.map(
            lambda a, b: ~jnp.array_equal(a, b),
            agent_state.vec_q.params,
            new_state.vec_q.params,
        )
        assert any(jax.tree.leaves(critic_leaves_changed))

        # Target Q params differ from online Q after Polyak update
        target_differs_from_online = jax.tree.map(
            lambda t, q: ~jnp.array_equal(t, q),
            new_state.vec_q_target.params,
            new_state.vec_q.params,
        )
        assert any(jax.tree.leaves(target_differs_from_online))

    def test_terminal_state_masking(self, agent_state, synthetic_batch):
        """done=1 should zero the bootstrap term so next_obs has no effect on critic loss.

        For terminal transitions: target = reward + gamma * (1 - 1) * next_v = reward.
        Changing next_obs should leave the critic loss unchanged.
        For non-terminal transitions: target depends on next_obs, so different next_obs
        should produce a different critic loss.
        """
        sac_update = make_sac_update(
            agent_state.actor.apply_fn,
            agent_state.vec_q.apply_fn,
            agent_state.alpha.apply_fn,
            FAST_CFG,
        )
        rng = jax.random.key(7)
        next_obs_a = synthetic_batch.next_obs
        next_obs_b = next_obs_a * 100.0  # very different next observations

        # --- Terminal batch (done=1): next_obs should not matter ---
        done_terminal = jnp.ones(BATCH_SIZE, dtype=jnp.float32)
        batch_t1 = synthetic_batch._replace(done=done_terminal, next_obs=next_obs_a)
        batch_t2 = synthetic_batch._replace(done=done_terminal, next_obs=next_obs_b)
        _, metrics_t1 = sac_update(rng, agent_state, batch_t1)
        _, metrics_t2 = sac_update(rng, agent_state, batch_t2)
        assert jnp.allclose(metrics_t1["critic_loss"], metrics_t2["critic_loss"]), (
            "critic_loss should be identical for terminal transitions regardless of next_obs"
        )

        # --- Non-terminal batch (done=0): next_obs should matter ---
        done_nonterminal = jnp.zeros(BATCH_SIZE, dtype=jnp.float32)
        batch_n1 = synthetic_batch._replace(done=done_nonterminal, next_obs=next_obs_a)
        batch_n2 = synthetic_batch._replace(done=done_nonterminal, next_obs=next_obs_b)
        _, metrics_n1 = sac_update(rng, agent_state, batch_n1)
        _, metrics_n2 = sac_update(rng, agent_state, batch_n2)
        assert not jnp.allclose(metrics_n1["critic_loss"], metrics_n2["critic_loss"]), (
            "critic_loss should differ for non-terminal transitions when next_obs changes"
        )
