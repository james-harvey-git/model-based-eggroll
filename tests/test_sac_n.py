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
        assert out.shape == ()

    def test_vector_q_output_shape(self):
        net = VectorQ(num_critics=NUM_CRITICS)
        obs = jnp.zeros(OBS_DIM)
        action = jnp.zeros(ACT_DIM)
        params = net.init(jax.random.key(0), obs, action)
        out = net.apply(params, obs, action)
        assert out.shape == (NUM_CRITICS,)

    def test_actor_distribution_shape(self):
        net = TanhGaussianActor(ACT_DIM)
        obs = jnp.zeros(OBS_DIM)
        params = net.init(jax.random.key(0), obs)
        pi = net.apply(params, obs)
        sample, log_prob = pi.sample_and_log_prob(seed=jax.random.key(1))
        assert sample.shape == (ACT_DIM,)
        assert log_prob.shape == (ACT_DIM,)

    def test_entropy_coef_init(self):
        net = EntropyCoef()
        params = net.init(jax.random.key(0))
        log_coef = net.apply(params)
        assert jnp.allclose(jnp.exp(log_coef), jnp.ones(()), atol=1e-6)


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
        (rng_out, new_state), metrics = sac_update(rng, agent_state, synthetic_batch)

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

        # Target Q params differ from online Q after Polyak update
        target_differs_from_online = jax.tree.map(
            lambda t, q: ~jnp.array_equal(t, q),
            new_state.vec_q_target.params,
            new_state.vec_q.params,
        )
        assert any(jax.tree.leaves(target_differs_from_online))
