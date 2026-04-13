"""Core EGGROLL primitives.

Vendored from https://github.com/ESHyperscale/HyperscaleES at commit b77f7d6.
Source files: models/base_model.py, noiser/base_noiser.py, noiser/eggroll.py, models/common.py

Changes from upstream:
- Removed print("LORA UPDATE", ...) debug statement from _simple_lora_update
- Removed relative imports (consolidated into single module)
- Removed duplicate `import optax` (appeared twice in upstream noiser/eggroll.py)
- Removed assignment to `broadcasted_sigma` in _simple_full_update (only used in a
  commented-out line; marked noqa: F841 in upstream, removed here to keep ruff clean)
- Changed `any` → `Any` (typing) in CommonInit/CommonParams NamedTuple fields; upstream
  used the builtin `any` function as a type annotation, which is incorrect
"""
# ruff: noqa: E501  -- vendored code; do not reformat to fit line-length

from collections import defaultdict
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
import optax

# ── NamedTuples and constants ──────────────────────────────────────────────────

PARAM = 0
MM_PARAM = 1
EMB_PARAM = 2
EXCLUDED = 3


class CommonInit(NamedTuple):
    frozen_params: Any
    params: Any
    scan_map: Any
    es_map: Any


class CommonParams(NamedTuple):
    noiser: Any
    frozen_noiser_params: Any
    noiser_params: Any
    frozen_params: Any
    params: Any
    es_tree_key: Any
    iterinfo: Any


# ── Noiser (base class with no-op defaults) ────────────────────────────────────

class Noiser:
    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, solver=None, solver_kwargs=None, **kwargs):
        """
        params: parameters of the model
        sigma: initial sigma of noiser
        lr: learning rate of optimization process
        solver (optional, keyword arg): the optax solver to use (i.e. optax.adamw)
        solver_kwargs (optional, keyword arg): the optax solver's keyword arguments (excluding learning rate)

        Return frozen_noiser_params and noiser_params
        """
        return {}, {}

    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        return x @ param.T

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        return x @ param

    @classmethod
    def do_emb(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        return param[x]

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        return param

    @classmethod
    def convert_fitnesses(cls, frozen_noiser_params, noiser_params, raw_scores, num_episodes_list=None):
        return raw_scores

    @classmethod
    def do_updates(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        return noiser_params, params


# ── EggRoll ────────────────────────────────────────────────────────────────────

def get_lora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    a, b = param.shape
    lora_params = jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx), (a + b, frozen_noiser_params["rank"]), dtype=param.dtype)
    B = lora_params[:b]  # b x r
    A = lora_params[b:]  # a x r

    # update is A @ B.T
    return A * sigma, B


def get_nonlora_update_params(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = 0 if frozen_noiser_params["noise_reuse"] == 0 else epoch // frozen_noiser_params["noise_reuse"]

    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    updates = jax.random.normal(jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx), param.shape, dtype=param.dtype)
    return updates * sigma


def _simple_full_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    if frozen_noiser_params["freeze_nonlora"]:
        return jnp.zeros_like(param)
    _, thread_ids = iterinfo
    updates = jax.vmap(partial(get_nonlora_update_params, frozen_noiser_params), in_axes=(None, 0, None, None))(base_sigma, iterinfo, param, key)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1,) * len(param.shape))
    return jnp.astype(jnp.mean(broadcasted_scores * updates, axis=0), param.dtype)


def _simple_lora_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    A, B = jax.vmap(partial(get_lora_update_params, frozen_noiser_params), in_axes=(None, 0, None, None))(base_sigma / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, key)
    broadcasted_scores = jnp.reshape(scores, scores.shape + (1, 1))
    A = broadcasted_scores * A  # N x a x r for A vs N x b x r for B -> final update is just a x b
    num_envs = scores.shape[0]
    return jnp.einsum('nir,njr->ij', A, B) / num_envs


def _noop_update(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    return jnp.zeros_like(param)


class EggRoll(Noiser):
    @classmethod
    def init_noiser(cls, params, sigma, lr, *args, solver=None, solver_kwargs=None, group_size=0, freeze_nonlora=False, noise_reuse=0, rank=1, use_batched_update: bool = False, **kwargs):
        """
        Return frozen_noiser_params and noiser_params
        """
        if solver is None:
            solver = optax.sgd
        if solver_kwargs is None:
            solver_kwargs = {}
        true_solver = solver(lr, **solver_kwargs)
        opt_state = true_solver.init(params)

        return {"group_size": group_size, "freeze_nonlora": freeze_nonlora, "noise_reuse": noise_reuse, "solver": true_solver, "rank": rank, "use_batched_update": use_batched_update}, {"sigma": sigma, "opt_state": opt_state}

    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base_ans = x @ param.T
        if iterinfo is None:
            return base_ans
        A, B = get_lora_update_params(frozen_noiser_params, noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, base_key)
        return base_ans + x @ B @ A.T

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base_ans = x @ param
        if iterinfo is None:
            return base_ans
        A, B = get_lora_update_params(frozen_noiser_params, noiser_params["sigma"] / jnp.sqrt(frozen_noiser_params["rank"]), iterinfo, param, base_key)
        return base_ans + x @ A @ B.T

    @classmethod
    def do_emb(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        raise NotImplementedError("Embedding is not implemented")

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        if iterinfo is None or frozen_noiser_params["freeze_nonlora"]:
            return param
        return param + get_nonlora_update_params(frozen_noiser_params, noiser_params["sigma"], iterinfo, param, base_key)

    @classmethod
    def convert_fitnesses(cls, frozen_noiser_params, noiser_params, raw_scores, num_episodes_list=None):
        group_size = frozen_noiser_params["group_size"]
        if group_size == 0:
            true_scores = (raw_scores - jnp.mean(raw_scores, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
        else:
            group_scores = raw_scores.reshape((-1, group_size))
            true_scores = (group_scores - jnp.mean(group_scores, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
            true_scores = true_scores.ravel()
        return true_scores

    @classmethod
    def _do_update(cls, param, base_key, fitnesses, iterinfos, map_classification, sigma, frozen_noiser_params, **kwargs):
        update_fn = [_simple_full_update, _simple_lora_update, _noop_update, _noop_update][map_classification]

        if len(base_key.shape) == 0:
            new_grad = update_fn(sigma, param, base_key, fitnesses, iterinfos, frozen_noiser_params)
        else:
            new_grad = jax.lax.scan(lambda _, x: (0, update_fn(sigma, x[0], x[1], fitnesses, iterinfos, frozen_noiser_params)), 0, xs=(param, base_key))[1]

        return -(new_grad * jnp.sqrt(fitnesses.size)).astype(param.dtype)

    @classmethod
    def do_updates(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        if frozen_noiser_params["use_batched_update"]:
            return cls._do_updates_batched(frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map)
        else:
            return cls._do_updates_original(frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map)

    @classmethod
    def _do_updates_original(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        new_grad = jax.tree.map(lambda p, k, m: cls._do_update(p, k, fitnesses, iterinfos, m, noiser_params["sigma"], frozen_noiser_params), params, base_keys, es_map)
        updates, noiser_params["opt_state"] = frozen_noiser_params["solver"].update(new_grad, noiser_params["opt_state"], params)
        return noiser_params, optax.apply_updates(params, updates)

    @classmethod
    def _do_updates_batched(cls, frozen_noiser_params, noiser_params, params, base_keys, fitnesses, iterinfos, es_map):
        """Batched update: groups weight matrices of the same shape to vmap together,
        avoiding a Python loop over every individual matrix. Makes XLA compilation
        significantly faster for large models.
        """
        # Flatten all elements from the params, keys and map pytrees
        flat_params, treedef = tree_flatten(params)
        flat_keys, _ = tree_flatten(base_keys)
        flat_es, _ = tree_flatten(es_map)

        # Group param matrices based on (a) their shape and (b) which update fn (full or lora) they use
        buckets = defaultdict(list)
        for i, (param, map_class) in enumerate(zip(flat_params, flat_es)):
            key = param.shape, map_class
            buckets[key].append(i)

        # Fill these grads as we go through the batched process
        new_flat_grads = [None] * len(flat_params)

        # Process each bucket in a batched way instead of separately
        for (_, map_class), indices in buckets.items():
            # makes arrays of shape (num_arrays_of_this_shape, w, h)
            batched_params = jnp.stack([flat_params[i] for i in indices])
            batched_keys = jnp.stack([flat_keys[i] for i in indices])

            # we vmap only over the param and rng, and nothing else
            grads_for_this_batch = jax.vmap(
                lambda param, rng: cls._do_update(param, rng, fitnesses, iterinfos, map_class, noiser_params["sigma"], frozen_noiser_params),
            )(batched_params, batched_keys)

            # Unstack back into the flat list
            for i, idx in enumerate(indices):
                new_flat_grads[idx] = grads_for_this_batch[i]

        # Unflatten back into the original pytree def and apply updates
        new_grad = tree_unflatten(treedef, new_flat_grads)
        updates, noiser_params["opt_state"] = frozen_noiser_params["solver"].update(new_grad, noiser_params["opt_state"], params)
        return noiser_params, optax.apply_updates(params, updates)


# ── Model (base class) ─────────────────────────────────────────────────────────

class Model:
    @classmethod
    def rand_init(cls, key, *args, **kwargs):
        """
        Initialize model.

        Returns frozen_params, params, scan_map, es_map as CommonInit.
        """
        raise NotImplementedError("Randomize Weights is not implemented")

    @classmethod
    def forward(cls, noiser, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, iterinfo, *args, **kwargs):
        """
        Forward pass of model.

        Returns just the output.
        """
        return cls._forward(CommonParams(noiser, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, iterinfo), *args, **kwargs)

    @classmethod
    def _forward(cls, common_params, *args, **kwargs):
        raise NotImplementedError("Forward pass is not implemented")


# ── Helpers ────────────────────────────────────────────────────────────────────

def layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std


ACTIVATIONS = {
    'relu': jax.nn.relu,
    'silu': jax.nn.silu,
    'pqn': lambda x: jax.nn.relu(layer_norm(x)),
}


def recursive_scan_split(param, base_key, scan_tuple):
    # scan_tuple = tuple() implies no split
    if len(scan_tuple) == 0:
        return base_key
    # otherwise, it is (0, 1, ...)
    split_keys = jax.random.split(base_key, param.shape[scan_tuple[0]])
    return jax.vmap(recursive_scan_split, in_axes=(None, 0, None))(param, split_keys, scan_tuple[1:])


def simple_es_tree_key(params, base_key, scan_map):
    vals, treedef = jax.tree.flatten(params)
    all_keys = jax.random.split(base_key, len(vals))
    partial_key_tree = jax.tree.unflatten(treedef, all_keys)
    return jax.tree.map(recursive_scan_split, params, partial_key_tree, scan_map)


def merge_inits(**kwargs):
    params = {}
    frozen_params = {}
    scan_map = {}
    es_map = {}
    for k in kwargs:
        params[k] = kwargs[k].params
        scan_map[k] = kwargs[k].scan_map
        es_map[k] = kwargs[k].es_map
        if kwargs[k].frozen_params is not None:
            frozen_params[k] = kwargs[k].frozen_params
    if not frozen_params:
        frozen_params = None

    return CommonInit(frozen_params, params, scan_map, es_map)


def merge_frozen(common, **kwargs):
    new_frozen_params = common.frozen_params or {}
    new_frozen_params = new_frozen_params | kwargs
    return common._replace(frozen_params=new_frozen_params)


def call_submodule(cls, name, common_params, *args, **kwargs):
    sub_common_params = common_params._replace(
        frozen_params=common_params.frozen_params[name] if common_params.frozen_params and name in common_params.frozen_params else None,
        params=common_params.params[name],
        es_tree_key=common_params.es_tree_key[name],
    )
    return cls._forward(sub_common_params, *args, **kwargs)


# ── Atomic primitives ──────────────────────────────────────────────────────────

class Parameter(Model):
    @classmethod
    def rand_init(cls, key, shape, scale, raw_value, dtype, *args, **kwargs):
        if raw_value is not None:
            params = raw_value.astype(dtype=dtype)
        else:
            params = (jax.random.normal(key, shape) * scale).astype(dtype=dtype)

        frozen_params = None
        scan_map = ()
        es_map = PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, *args, **kwargs):
        return common_params.noiser.get_noisy_standard(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo)


class MM(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = (jax.random.normal(key, (out_dim, in_dim)) * scale).astype(dtype=dtype)
        frozen_params = None
        scan_map = ()
        es_map = MM_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_mm(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)


class TMM(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = jax.random.normal(key, (in_dim, out_dim), dtype=dtype) * scale
        frozen_params = None
        scan_map = ()
        es_map = MM_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_Tmm(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)


class Embedding(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = jax.random.normal(key, (in_dim, out_dim), dtype=dtype) * scale
        frozen_params = None
        scan_map = ()
        es_map = EMB_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_emb(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)


# ── Compound primitives ────────────────────────────────────────────────────────

class Linear(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, use_bias, dtype, *args, **kwargs):
        if use_bias:
            return merge_inits(
                weight=MM.rand_init(key, in_dim, out_dim, dtype),
                bias=Parameter.rand_init(key, None, None, jnp.zeros(out_dim, dtype=dtype), dtype),
            )
        else:
            return merge_inits(
                weight=MM.rand_init(key, in_dim, out_dim, dtype),
            )

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        ans = call_submodule(MM, 'weight', common_params, x)
        if "bias" in common_params.params:
            ans += call_submodule(Parameter, 'bias', common_params)
        return ans


class MLP(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, hidden_dims, use_bias, activation, dtype, *args, **kwargs):
        input_dims = [in_dim] + list(hidden_dims)
        output_dims = list(hidden_dims) + [out_dim]

        all_keys = jax.random.split(key, len(input_dims))

        merged_params = merge_inits(**{str(t): Linear.rand_init(all_keys[t], input_dims[t], output_dims[t], use_bias, dtype) for t in range(len(input_dims))})
        return merge_frozen(merged_params, activation=activation)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        num_blocks = len(common_params.params)
        for t in range(num_blocks):
            x = call_submodule(Linear, str(t), common_params, x)
            if t != num_blocks - 1:
                x = ACTIVATIONS[common_params.frozen_params['activation']](x)
        return x
