"""Shared EGGROLL training utilities.

Provides state management, iterinfo generation, and the common
convert_fitnesses + do_updates cycle shared by both world model training
(mbrl.world_models.eggroll) and policy search (mbrl.policy_optimizers.eggroll).
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from mbrl.eggroll.primitives import CommonInit, EggRoll, simple_es_tree_key


class EGGROLLState(NamedTuple):
    """All mutable and frozen state needed for an EGGROLL training loop.

    Note on JIT: ``frozen_noiser_params`` contains the optax solver (a Python
    callable), so the full state cannot be passed as a dynamic argument to
    ``jax.jit``. Follow the notebook pattern: capture ``frozen_noiser_params``
    as a closure variable and pass only ``noiser_params`` and ``params``
    dynamically.
    """

    frozen_noiser_params: dict  # static config: solver, rank, group_size, …
    noiser_params: dict  # mutable: sigma, opt_state — updated by do_updates
    frozen_params: dict | None  # e.g. {"activation": "relu"} — read by MLP._forward
    params: Any  # model weights — updated by do_updates
    es_tree_key: Any  # per-parameter PRNG keys — fixed after init
    es_map: Any  # per-parameter update classification (PARAM / MM_PARAM / …)


def init_eggroll_state(
    common_init: CommonInit,
    es_key: jax.Array,
    sigma: float,
    lr: float,
    rank: int = 1,
    **eggroll_kwargs: Any,
) -> EGGROLLState:
    """Construct an EGGROLLState from the output of Model.rand_init.

    Args:
        common_init: Output of ``Model.rand_init`` — contains ``params``,
            ``frozen_params``, ``scan_map``, and ``es_map``.
        es_key: JAX PRNG key used to generate the per-parameter tree of base
            keys (``es_tree_key``).
        sigma: Initial perturbation scale.
        lr: Learning rate for the internal optax optimiser.
        rank: LoRA rank for weight-matrix perturbations (default 1).
        **eggroll_kwargs: Additional keyword arguments forwarded to
            ``EggRoll.init_noiser`` (e.g. ``group_size``, ``freeze_nonlora``,
            ``noise_reuse``, ``use_batched_update``).

    Returns:
        EGGROLLState ready for use in a training loop.
    """
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        common_init.params, sigma, lr, rank=rank, **eggroll_kwargs
    )
    es_tree_key = simple_es_tree_key(common_init.params, es_key, common_init.scan_map)
    return EGGROLLState(
        frozen_noiser_params=frozen_noiser_params,
        noiser_params=noiser_params,
        frozen_params=common_init.frozen_params,
        params=common_init.params,
        es_tree_key=es_tree_key,
        es_map=common_init.es_map,
    )


def get_iterinfos(epoch: int, num_envs: int) -> tuple[jax.Array, jax.Array]:
    """Construct the ``(epochs, thread_ids)`` iterinfo tuple for a training step.

    Args:
        epoch: Current training epoch (0-indexed update step count).
        num_envs: Number of parallel perturbations (population size).

    Returns:
        Tuple of ``(epochs, thread_ids)`` where ``epochs`` is an int32 array of
        shape ``(num_envs,)`` filled with ``epoch``, and ``thread_ids`` is
        ``jnp.arange(num_envs, dtype=jnp.int32)``.
    """
    return (jnp.full(num_envs, epoch, dtype=jnp.int32), jnp.arange(num_envs, dtype=jnp.int32))


def eggroll_step(
    state: EGGROLLState,
    fitnesses: jax.Array,
    iterinfos: tuple[jax.Array, jax.Array],
) -> EGGROLLState:
    """Apply one EGGROLL parameter update.

    Normalises raw fitnesses via ``convert_fitnesses``, then calls
    ``do_updates`` to compute and apply gradients. Returns a new state with
    updated ``noiser_params`` and ``params``.

    Does **not** apply sigma decay — sigma scheduling is the caller's
    responsibility. Update ``state.noiser_params["sigma"]`` between steps as
    needed (e.g. ``state.noiser_params["sigma"] = sigma * (1 - t / T)``).

    Args:
        state: Current EGGROLL training state.
        fitnesses: Raw fitness scores, shape ``(num_envs,)``. Higher is better.
        iterinfos: ``(epochs, thread_ids)`` tuple from ``get_iterinfos``.

    Returns:
        New EGGROLLState with updated ``params`` and ``noiser_params``.
    """
    normalized = EggRoll.convert_fitnesses(
        state.frozen_noiser_params, state.noiser_params, fitnesses
    )
    # Shallow-copy noiser_params before passing to do_updates: the upstream
    # implementation mutates the dict in-place (noiser_params["opt_state"] = ...)
    # so without this copy the original state's dict would be silently modified.
    noiser_params, params = EggRoll.do_updates(
        state.frozen_noiser_params,
        dict(state.noiser_params),
        state.params,
        state.es_tree_key,
        normalized,
        iterinfos,
        state.es_map,
    )
    return state._replace(noiser_params=noiser_params, params=params)
