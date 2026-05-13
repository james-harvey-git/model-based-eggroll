"""Shared EGGROLL training utilities.

Provides state management, iterinfo generation, and the common
convert_fitnesses + do_updates cycle shared by both world model training
(mbrl.world_models.eggroll) and policy search (mbrl.policy_optimizers.eggroll).
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from mbrl.eggroll.primitives import (
    LOGVAR_PARAM,
    MM_PARAM,
    CommonInit,
    EggRoll,
    simple_es_tree_key,
)


def build_sigma_tree(params: Any, es_map: Any, groups: dict[str, float]) -> Any:
    """Build a pytree mirroring ``params`` with per-leaf scalar sigmas.

    Per-leaf assignment dispatches on the leaf's ``es_map`` marker:
      - ``MM_PARAM``     → ``groups["lora"]``
      - ``LOGVAR_PARAM`` → ``groups["logvar"]``
      - otherwise        → ``groups["nonlora"]``

    Args:
        params: A pytree of leaf arrays.
        es_map: A pytree mirroring ``params`` with integer marker leaves.
        groups: Dict with keys ``{"lora", "nonlora", "logvar"}`` mapping to
            scalar sigma values.

    Returns:
        Pytree with the same structure as ``params`` where each leaf is a
        ``jnp.float32`` scalar — the per-leaf sigma.
    """
    lora = jnp.float32(groups["lora"])
    nonlora = jnp.float32(groups["nonlora"])
    logvar = jnp.float32(groups["logvar"])

    def assign(_leaf, marker):
        m = int(marker)
        if m == MM_PARAM:
            return lora
        if m == LOGVAR_PARAM:
            return logvar
        return nonlora

    return jax.tree.map(assign, params, es_map)


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


_OPTAX_SOLVERS: dict[str, Any] = {
    "sgd": optax.sgd,
    "adam": optax.adam,
    "adamw": optax.adamw,
}


def resolve_optax_solver(name: str) -> Any:
    """Resolve a supported Optax solver name for EGGROLL updates."""
    solver_name = name.lower()
    if solver_name not in _OPTAX_SOLVERS:
        supported = ", ".join(sorted(_OPTAX_SOLVERS))
        raise ValueError(
            f"Unsupported EGGROLL solver '{name}'. Supported solvers: {supported}"
        )
    return _OPTAX_SOLVERS[solver_name]


_GROUP_KEYS = frozenset({"lora", "nonlora", "logvar"})


def _is_groups_dict(value: Any) -> bool:
    """Distinguish a {lora, nonlora, logvar} groups dict from a params-shaped pytree.

    Both are dicts at the top level, so we need to inspect keys: only a groups
    dict has *exactly* the three group keys.
    """
    return isinstance(value, dict) and set(value.keys()) == _GROUP_KEYS


def _resolve_groups(value: Any) -> dict[str, float]:
    """Normalise a scalar or groups dict into ``{lora, nonlora, logvar}`` floats."""
    if _is_groups_dict(value):
        return {k: float(value[k]) for k in ("lora", "nonlora", "logvar")}
    v = float(value)
    return {"lora": v, "nonlora": v, "logvar": v}


def init_eggroll_state(
    common_init: CommonInit,
    es_key: jax.Array,
    sigma: Any,
    lr: float,
    rank: int = 1,
    sigma_decay_rate: Any = 1.0,
    **eggroll_kwargs: Any,
) -> EGGROLLState:
    """Construct an EGGROLLState from the output of Model.rand_init.

    Args:
        common_init: Output of ``Model.rand_init`` — contains ``params``,
            ``frozen_params``, ``scan_map``, and ``es_map``.
        es_key: JAX PRNG key used to generate the per-parameter tree of base
            keys (``es_tree_key``).
        sigma: Initial perturbation scale. Either a scalar (uniform across
            all params — backward-compat), a ``{lora, nonlora, logvar}`` dict
            (per-group, issue #32), or a pre-built sigma pytree mirroring
            ``common_init.params``.
        lr: Learning rate for the internal optax optimiser.
        rank: LoRA rank for weight-matrix perturbations (default 1).
        sigma_decay_rate: Per-step decay rate(s). Either a scalar (uniform)
            or a ``{lora, nonlora, logvar}`` dict. Stored in
            ``frozen_noiser_params["sigma_decay_rate"]`` as a per-leaf tree
            the caller's decay step can consume via ``jax.tree.map``.
            Default ``1.0`` (no decay).
        **eggroll_kwargs: Additional keyword arguments forwarded to
            ``EggRoll.init_noiser`` (e.g. ``group_size``, ``freeze_nonlora``,
            ``noise_reuse``, ``use_batched_update``).

    Returns:
        EGGROLLState ready for use in a training loop.
    """
    # Resolve sigma into a per-leaf pytree.
    #   - {lora, nonlora, logvar} groups dict → build per-leaf tree via es_map.
    #   - Scalar → pass through; init_noiser splats it into a uniform tree.
    #   - Anything else with multiple leaves → assumed pre-built pytree mirroring params.
    if _is_groups_dict(sigma):
        sigma_tree = build_sigma_tree(
            common_init.params, common_init.es_map, _resolve_groups(sigma),
        )
    elif isinstance(sigma, (int, float)) or jax.tree.structure(sigma).num_leaves == 1:
        sigma_tree = jnp.float32(sigma)
    else:
        sigma_tree = sigma  # pre-built pytree mirroring params

    decay_tree = build_sigma_tree(
        common_init.params, common_init.es_map, _resolve_groups(sigma_decay_rate),
    )

    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        common_init.params, sigma_tree, lr, rank=rank, decay_tree=decay_tree,
        **eggroll_kwargs,
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
