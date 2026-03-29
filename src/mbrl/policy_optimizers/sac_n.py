"""SAC-N: Soft Actor-Critic with N critics.

Shared building block used by MOPO, MoRel, and MoBRAC. Not a standalone policy
optimiser — exports networks, state management, and a `make_sac_update` factory
that each model-based method wraps with its own batch construction and rollout logic.

Ported from Unifloral (algorithms/sac_n.py):
https://github.com/EmptyJackson/unifloral
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from omegaconf import DictConfig
import optax

from mbrl.data import Transition

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sym(scale: float) -> Callable:
    """Symmetric uniform initialiser in [-scale, +scale]."""

    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


# ---------------------------------------------------------------------------
# Networks (ported from Unifloral sac_n.py)
# ---------------------------------------------------------------------------


class SoftQNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
        q = nn.Dense(1, kernel_init=_sym(3e-3), bias_init=_sym(3e-3))(x)
        return q.squeeze(-1)


class VectorQ(nn.Module):
    num_critics: int

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,  # type: ignore[arg-type]
            out_axes=-1,
            axis_size=self.num_critics,
        )
        return vmap_critic()(obs, action)  # (num_critics,)


class TanhGaussianActor(nn.Module):
    num_actions: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x: jax.Array) -> distrax.Distribution:
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
        mean = nn.Dense(self.num_actions, kernel_init=_sym(1e-3), bias_init=_sym(1e-3))(x)
        log_std = nn.Dense(self.num_actions, kernel_init=_sym(1e-3), bias_init=_sym(1e-3))(x)
        std = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max))
        return distrax.Transformed(distrax.Normal(mean, std), distrax.Tanh())


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jax.Array:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return log_ent_coef


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentTrainState(NamedTuple):
    actor: TrainState
    vec_q: TrainState
    vec_q_target: TrainState
    alpha: TrainState


def create_agent_state(
    obs_dim: int,
    act_dim: int,
    cfg: DictConfig,
    rng: jax.Array,
) -> AgentTrainState:
    """Initialise all SAC-N networks and return an AgentTrainState.

    vec_q and vec_q_target are initialised with the same RNG key so their
    parameters are identical at the start of training.
    """
    actor_net = TanhGaussianActor(act_dim)
    q_net = VectorQ(cfg.num_critics)
    alpha_net = EntropyCoef()

    dummy_obs = jnp.zeros(obs_dim)
    dummy_action = jnp.zeros(act_dim)

    rng, rng_actor, rng_q, rng_alpha = jax.random.split(rng, 4)
    tx = optax.adam(cfg.lr, eps=1e-5)

    return AgentTrainState(
        actor=TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(rng_actor, dummy_obs),
            tx=tx,
        ),
        vec_q=TrainState.create(
            apply_fn=q_net.apply,
            params=q_net.init(rng_q, dummy_obs, dummy_action),
            tx=tx,
        ),
        # Same rng_q as vec_q — matched initialisation
        vec_q_target=TrainState.create(
            apply_fn=q_net.apply,
            params=q_net.init(rng_q, dummy_obs, dummy_action),
            tx=tx,
        ),
        alpha=TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(rng_alpha),
            tx=tx,
        ),
    )


# ---------------------------------------------------------------------------
# SAC update
# ---------------------------------------------------------------------------


def make_sac_update(
    actor_apply_fn: Callable,
    q_apply_fn: Callable,
    alpha_apply_fn: Callable,
    cfg: DictConfig,
) -> Callable:
    """Return a pure SAC-N update function.

    The returned function accepts a pre-built batch so that callers (MOPO,
    MoRel, MoBRAC) can own their own batch construction — e.g. mixing real
    and synthetic transitions — while sharing the identical gradient update.

    Usage:
        sac_update = make_sac_update(actor.apply, q.apply, alpha.apply, cfg)
        (rng, new_state), metrics = sac_update(rng, agent_state, batch)
    """
    # Extract Python scalars at factory time — DictConfig is not JAX-traceable.
    gamma = float(cfg.gamma)
    polyak_step_size = float(cfg.polyak_step_size)

    def sac_update(
        rng: jax.Array,
        agent_state: AgentTrainState,
        batch: Transition,
    ) -> tuple[tuple[jax.Array, AgentTrainState], dict[str, jax.Array]]:
        batch_size = batch.obs.shape[0]

        # --- Update alpha ---
        @jax.value_and_grad
        def _alpha_loss_fn(alpha_params: dict, rng: jax.Array) -> jax.Array:
            def _entropy(rng: jax.Array, transition: Transition) -> jax.Array:
                pi = actor_apply_fn(agent_state.actor.params, transition.obs)
                # sample_and_log_prob required for numerical stability under Tanh bijector
                # See https://github.com/deepmind/distrax/issues/7
                _, log_pi = pi.sample_and_log_prob(seed=rng)
                return -log_pi.sum()

            log_alpha = alpha_apply_fn(alpha_params)
            rngs = jax.random.split(rng, batch_size)
            entropy = jax.vmap(_entropy)(rngs, batch).mean()
            target_entropy = -batch.action.shape[-1]
            return log_alpha * (entropy - target_entropy)

        rng, rng_alpha = jax.random.split(rng)
        alpha_loss, alpha_grad = _alpha_loss_fn(agent_state.alpha.params, rng_alpha)
        updated_alpha = agent_state.alpha.apply_gradients(grads=alpha_grad)
        agent_state = agent_state._replace(alpha=updated_alpha)
        alpha = jnp.exp(alpha_apply_fn(agent_state.alpha.params))

        # --- Update actor ---
        @partial(jax.value_and_grad, has_aux=True)
        def _actor_loss_fn(
            actor_params: dict, rng: jax.Array
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
            def _loss(
                rng: jax.Array, transition: Transition
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                pi = actor_apply_fn(actor_params, transition.obs)
                sampled_action, log_pi = pi.sample_and_log_prob(seed=rng)
                log_pi = log_pi.sum()  # sum over action dims → scalar per sample
                q_values = q_apply_fn(
                    agent_state.vec_q.params, transition.obs, sampled_action
                )
                q_min = jnp.min(q_values)
                return -q_min + alpha * log_pi, -log_pi, q_min, q_values.std()

            rngs = jax.random.split(rng, batch_size)
            loss, entropy, q_min, q_std = jax.vmap(_loss)(rngs, batch)
            return loss.mean(), (entropy.mean(), q_min.mean(), q_std.mean())

        rng, rng_actor = jax.random.split(rng)
        (actor_loss, (entropy, q_min, q_std)), actor_grad = _actor_loss_fn(
            agent_state.actor.params, rng_actor
        )
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        # --- Polyak update target Q ---
        updated_q_target_params = optax.incremental_update(
            agent_state.vec_q.params,
            agent_state.vec_q_target.params,
            polyak_step_size,
        )
        updated_q_target = agent_state.vec_q_target.replace(
            step=agent_state.vec_q_target.step + 1,
            params=updated_q_target_params,
        )
        agent_state = agent_state._replace(vec_q_target=updated_q_target)

        # --- Compute Q targets (using updated actor + updated target Q) ---
        def _next_v(rng: jax.Array, transition: Transition) -> jax.Array:
            next_pi = actor_apply_fn(agent_state.actor.params, transition.next_obs)
            next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng)
            next_q = q_apply_fn(
                agent_state.vec_q_target.params, transition.next_obs, next_action
            )
            return next_q.min(-1) - alpha * log_next_pi.sum(-1)

        rng, rng_next_v = jax.random.split(rng)
        rngs_next_v = jax.random.split(rng_next_v, batch_size)
        next_v = jax.vmap(_next_v)(rngs_next_v, batch)
        target = batch.reward + gamma * (1 - batch.done) * next_v

        # --- Update critics ---
        @jax.value_and_grad
        def _critic_loss_fn(q_params: dict) -> jax.Array:
            q_pred = q_apply_fn(q_params, batch.obs, batch.action)  # (batch, num_critics)
            return jnp.square(q_pred - jnp.expand_dims(target, -1)).sum(-1).mean()

        critic_loss, critic_grad = _critic_loss_fn(agent_state.vec_q.params)
        updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad)
        agent_state = agent_state._replace(vec_q=updated_q)

        metrics: dict[str, jax.Array] = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "entropy": entropy,
            "alpha": alpha,
            "q_min": q_min,
            "q_std": q_std,
        }
        return (rng, agent_state), metrics

    return sac_update
