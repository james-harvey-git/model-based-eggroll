"""MOReL: Model-Based Offline Reinforcement Learning.

Reference: Kidambi et al., 2020 (https://arxiv.org/abs/2005.05951)

Ported from Unifloral (algorithms/morel.py and algorithms/dynamics.py):
https://github.com/EmptyJackson/unifloral

MoReL trains SAC-N on a mixture of real offline transitions and synthetic
transitions rolled out through a learned world model — the same scaffolding as
MOPO. It differs from MOPO only in the rollout penalty: instead of softly
penalising the reward by the ensemble's predicted std, MoReL builds a pessimistic
MDP with a *halt state*. When the elite ensemble disagrees more than a calibrated
threshold at a state-action (the Unknown State-Action Detector, USAD), the
synthetic reward is overwritten with ``min_r + term_penalty_offset`` and the
rollout terminates — teaching the policy to avoid model-uncertain regions.

The threshold (``discrepancy``) and the reward floor (``min_r``) are precomputed
when the world model is trained (``precompute_term_stats``) and read off the model
here. See ``world_models/term_stats.py``.

Usage:
    agent_state = train(world_model, dataset, cfg, rng)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from mbrl.data import Transition, sample_batch
from mbrl.policy_optimizers.sac_n import (
    AgentTrainState,
    create_agent_state,
    make_sac_update,
)
from mbrl.world_models.base import EnsembleDynamics


def make_rollout_fn(
    world_model: EnsembleDynamics,
    actor_apply_fn: Callable,
    dataset_obs: jnp.ndarray,
    cfg: DictConfig,
) -> Callable[..., Transition]:
    """Return a pure rollout function generating synthetic transitions with MoReL's penalty.

    Args:
        world_model: Trained ensemble dynamics model carrying precomputed term stats.
        actor_apply_fn: Actor network apply function (immutable, set at agent creation).
        dataset_obs: Offline dataset observations used to sample rollout starting states.
        cfg: Policy optimizer config — must have rollout_batch_size, rollout_length,
             threshold_coef, term_penalty_offset.

    Returns:
        rollout_fn(rng, actor_params, rollout_buffer) -> rollout_buffer
    """
    # Extract Python scalars at factory time — DictConfig is not JAX-traceable.
    rollout_batch_size = int(cfg.rollout_batch_size)
    rollout_length = int(cfg.rollout_length)
    threshold_coef = float(cfg.threshold_coef)
    term_penalty_offset = float(cfg.term_penalty_offset)
    # num_new is a Python int used as a static slice bound in the FIFO insertion.
    num_new = rollout_batch_size * rollout_length

    # MoReL requires the world model to carry precomputed halt-penalty stats.
    discrepancy = world_model.discrepancy
    min_r = world_model.min_r
    assert discrepancy is not None and min_r is not None, (
        "MoReL requires a world model with precomputed term stats. Train the world "
        "model with world_model.precompute_term_stats=true so discrepancy/min_r are "
        "stored in the checkpoint."
    )
    # USAD threshold (a global scale) and the halt-state reward, both static floats.
    threshold = float(discrepancy) * threshold_coef
    halt_reward = float(min_r) + term_penalty_offset

    # Capture bound methods from world model at factory time.
    predict_ensemble = world_model.predict_ensemble
    termination_fn = world_model.termination_fn

    def rollout_fn(
        rng: jax.Array,
        actor_params: dict,
        rollout_buffer: Transition,
    ) -> Transition:
        def _sample_transition(rng: jax.Array, obs: jax.Array) -> Transition:
            """Generate one transition from a single obs using the current policy.

            Operates on a single (obs,) — vmapped by caller over the batch dimension.
            """
            rng_action, rng_elite, rng_noise = jax.random.split(rng, 3)

            # Sample action from current policy
            pi = actor_apply_fn(actor_params, obs)
            action = pi.sample(seed=rng_action)

            # Get full elite ensemble predictions for the disagreement penalty
            ensemble_mean, ensemble_std = predict_ensemble(obs, action)

            # Sample from one randomly-selected elite member
            num_elites = ensemble_mean.shape[0]
            sample_idx = jax.random.randint(rng_elite, (), 0, num_elites)
            mean = ensemble_mean[sample_idx]
            std = ensemble_std[sample_idx]
            noise = jax.random.normal(rng_noise, shape=mean.shape)
            sample = mean + noise * std

            # Split into delta_obs and reward.
            # Use indexing (sample[-1]) not slicing (sample[..., -1:]) to get a
            # scalar reward consistent with our Transition.reward convention of (N,).
            delta_obs = sample[:-1]
            reward = sample[-1]
            next_obs = obs + delta_obs
            done = termination_fn(obs, action, next_obs)

            # MoReL halt-state penalty (USAD): if the max pairwise disagreement between
            # elite mean-predictions at this (obs, action) exceeds the calibrated
            # threshold, the state-action is treated as "unknown" — terminate and assign
            # the worst-case reward so the policy learns to avoid it.
            pairwise = ensemble_mean[:, None, :] - ensemble_mean[None, :, :]  # (E, E, D)
            disagreement = jnp.max(jnp.linalg.norm(pairwise, axis=-1))
            is_uncertain = disagreement > threshold
            reward = jnp.where(is_uncertain, halt_reward, reward)
            done = jnp.logical_or(done, is_uncertain)

            return Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

        def _rollout_step(
            carry: tuple[jax.Array, jax.Array], _: None
        ) -> tuple[tuple[jax.Array, jax.Array], Transition]:
            """One time step of the rollout. Vmapped over the batch at the scan call site."""
            obs, rng = carry
            rng, rng_step = jax.random.split(rng)
            transition = _sample_transition(rng_step, obs)
            return (transition.next_obs, rng), transition

        # Sample starting observations uniformly from the offline dataset.
        rng, rng_init, rng_rollout = jax.random.split(rng, 3)
        init_idxs = jax.random.choice(rng_init, dataset_obs.shape[0], (rollout_batch_size,))
        init_obs = dataset_obs[init_idxs]  # (rollout_batch_size, obs_dim)
        rng_rollouts = jax.random.split(rng_rollout, rollout_batch_size)

        # Generate rollouts: vmap is INNER (over batch), scan is OUTER (over time steps).
        # Each scan step processes all rollout_batch_size observations in parallel.
        _, rollouts = jax.lax.scan(
            jax.vmap(_rollout_step),
            (init_obs, rng_rollouts),
            None,
            length=rollout_length,
        )
        # rollouts fields: (rollout_length, rollout_batch_size, ...)

        # Flatten (rollout_length, rollout_batch_size, ...) -> (num_new, ...).
        rollouts = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), rollouts)

        # FIFO insertion: replace the oldest num_new entries with fresh rollouts.
        # Oldest data is at the front of the buffer, newest at the back.
        # num_new is a Python int — required for static slice bounds inside JIT.
        return jax.tree.map(
            lambda old, new: jnp.concatenate([old[:-num_new], new]),
            rollout_buffer,
            rollouts,
        )

    return rollout_fn


def make_morel_step(
    sac_update: Callable,
    rollout_fn: Callable,
    dataset: Transition,
    cfg: DictConfig,
) -> Callable:
    """Return a single MoReL training step suitable for jax.lax.scan.

    Identical scaffolding to MOPO — mixes real and synthetic transitions and runs
    the shared SAC-N update; the MoReL-specific behaviour lives entirely in
    ``rollout_fn``'s penalty.

    Args:
        sac_update: Pure SAC-N update function from make_sac_update.
        rollout_fn: Rollout function from make_rollout_fn.
        dataset: Offline dataset (real transitions).
        cfg: Policy optimizer config — must have batch_size, rollout_interval,
             dataset_sample_ratio.

    Returns:
        morel_step(runner_state, _) -> (runner_state, metrics)
        where runner_state = (rng, agent_state, rollout_buffer).
    """
    # Extract Python scalars at factory time — DictConfig is not JAX-traceable.
    batch_size = int(cfg.batch_size)
    rollout_interval = int(cfg.rollout_interval)
    real_size = int(batch_size * float(cfg.dataset_sample_ratio))
    synthetic_size = batch_size - real_size

    def morel_step(
        runner_state: tuple[jax.Array, AgentTrainState, Transition],
        _: None,
    ) -> tuple[tuple[jax.Array, AgentTrainState, Transition], dict[str, jax.Array]]:
        rng, agent_state, rollout_buffer = runner_state

        # Conditionally refresh rollout buffer every rollout_interval SAC steps.
        # actor.step == 0 on the first call → immediate rollout (buffer starts zero-filled).
        rng, rng_rollout = jax.random.split(rng)
        rollout_buffer = jax.lax.cond(
            agent_state.actor.step % rollout_interval == 0,
            lambda: rollout_fn(rng_rollout, agent_state.actor.params, rollout_buffer),
            lambda: rollout_buffer,
        )

        # Mix real and synthetic transitions.
        # real_size ≈ 1% of batch (default dataset_sample_ratio=0.01).
        rng, rng_real, rng_synthetic = jax.random.split(rng, 3)
        real_batch = sample_batch(dataset, real_size, rng_real)
        synthetic_batch = sample_batch(rollout_buffer, synthetic_size, rng_synthetic)
        batch = jax.tree.map(
            lambda r, s: jnp.concatenate([r, s]),
            real_batch,
            synthetic_batch,
        )

        # SAC-N gradient update on the mixed batch.
        (rng, agent_state), metrics = sac_update(rng, agent_state, batch)

        return (rng, agent_state, rollout_buffer), metrics

    return morel_step


def make_train_step(
    world_model: EnsembleDynamics,
    dataset: Transition,
    cfg: DictConfig,
    rng: jax.Array,
) -> tuple[Callable, tuple[jax.Array, AgentTrainState, Transition]]:
    """Build the MoReL step function and initial runner state.

    Convention function: every policy optimizer module exports this.

    Args:
        world_model: Pre-trained ensemble dynamics model with precomputed term stats.
        dataset: Offline dataset of real transitions.
        cfg: Policy optimizer config (see configs/policy_optimizer/morel.yaml).
        rng: JAX random key.

    Returns:
        ``(morel_step, runner_state)`` where ``runner_state = (rng, agent_state, rollout_buffer)``.
    """
    obs_dim = dataset.obs.shape[1]
    act_dim = dataset.action.shape[1]

    rng, rng_agent = jax.random.split(rng)
    agent_state = create_agent_state(obs_dim, act_dim, cfg, rng_agent)

    # Pre-allocate rollout buffer as a zero-filled Transition pytree.
    # buffer_size = rollout_batch_size * rollout_length * model_retain_epochs
    max_buffer_size = (
        int(cfg.rollout_batch_size) * int(cfg.rollout_length) * int(cfg.model_retain_epochs)
    )
    rollout_buffer = jax.tree.map(
        lambda x: jnp.zeros((max_buffer_size, *x.shape[1:])),
        dataset,
    )

    # Build the function pipeline.
    # actor_apply_fn is immutable (set at TrainState creation, never changes).
    sac_update = make_sac_update(
        agent_state.actor.apply_fn,
        agent_state.vec_q.apply_fn,
        agent_state.alpha.apply_fn,
        cfg,
    )
    rollout_fn = make_rollout_fn(world_model, agent_state.actor.apply_fn, dataset.obs, cfg)
    morel_step = make_morel_step(sac_update, rollout_fn, dataset, cfg)

    runner_state = (rng, agent_state, rollout_buffer)
    return morel_step, runner_state


def extract_actor(
    runner_state: tuple[jax.Array, AgentTrainState, Transition],
) -> tuple[dict, int]:
    """Return ``(actor_params, step)`` from the MoReL runner state.

    Convention function: every policy optimizer module exports this.
    """
    _, agent_state, _ = runner_state
    return dict(agent_state.actor.params), int(agent_state.actor.step)


def train(
    world_model: EnsembleDynamics,
    dataset: Transition,
    cfg: DictConfig,
    rng: jax.Array,
) -> AgentTrainState:
    """Train a policy with MoReL using the provided world model.

    Args:
        world_model: Pre-trained ensemble dynamics model with precomputed term stats.
        dataset: Offline dataset of real transitions.
        cfg: Policy optimizer config (see configs/policy_optimizer/morel.yaml).
        rng: JAX random key.

    Returns:
        Trained AgentTrainState (actor, vec_q, vec_q_target, alpha).
    """
    morel_step, runner_state = make_train_step(world_model, dataset, cfg, rng)
    num_policy_updates = int(cfg.num_policy_updates)
    runner_state, _ = jax.lax.scan(morel_step, runner_state, None, length=num_policy_updates)
    _, agent_state, _ = runner_state
    return agent_state
