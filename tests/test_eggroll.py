"""Tests for the EGGROLL integration layer (primitives and training utilities).

Organised in two sections:
  - TestVendoredPrimitives  — smoke tests for mbrl/eggroll/primitives.py (vendored
                               from HyperscaleES; intentionally minimal).
  - TestInitEggrollState    — mbrl/eggroll/training.py: EGGROLLState construction.
  - TestGetIterinfos        — mbrl/eggroll/training.py: iterinfo generation.
  - TestEggrollStep         — mbrl/eggroll/training.py: update cycle.
"""

import jax
import jax.numpy as jnp
import jax.tree_util

from mbrl.eggroll.primitives import MLP, EggRoll
from mbrl.eggroll.training import EGGROLLState, eggroll_step, get_iterinfos, init_eggroll_state

# ── Vendored primitives ────────────────────────────────────────────────────────


class TestVendoredPrimitives:
    """Smoke tests for the vendored primitives module.

    These tests are intentionally minimal. The primitives are vendored
    third-party code (HyperscaleES); we don't test their internal correctness
    here. What we do verify:
      1. All expected names are importable (catches accidental omissions).
      2. A basic end-to-end forward pass works (catches consolidation mistakes
         such as wrong definition ordering or missing cross-references from
         merging 4 files into 1).
    """

    def test_all_names_importable(self):
        from mbrl.eggroll.primitives import (  # noqa: F401
            EMB_PARAM,
            EXCLUDED,
            MLP,
            MM,
            MM_PARAM,
            PARAM,
            TMM,
            CommonInit,
            CommonParams,
            EggRoll,
            Embedding,
            Linear,
            Model,
            Noiser,
            Parameter,
            call_submodule,
            merge_frozen,
            merge_inits,
            simple_es_tree_key,
        )

    def test_mlp_forward_pass(self):
        """Smoke test: init an MLP, generate es_tree_key, run a forward pass in eval
        mode (iterinfo=None). Verifies the 4-file consolidation didn't break any
        cross-references between Noiser, Model, and the helper functions."""
        from mbrl.eggroll.primitives import MLP, EggRoll, simple_es_tree_key

        model_key, es_key = jax.random.split(jax.random.key(0))
        init = MLP.rand_init(
            model_key,
            in_dim=4,
            out_dim=2,
            hidden_dims=[8],
            use_bias=True,
            activation="relu",
            dtype="float32",
        )
        es_tree_key = simple_es_tree_key(init.params, es_key, init.scan_map)
        frozen_noiser_params, noiser_params = EggRoll.init_noiser(
            init.params, sigma=0.1, lr=1e-3, rank=1
        )

        out = MLP.forward(
            EggRoll,
            frozen_noiser_params,
            noiser_params,
            init.frozen_params,
            init.params,
            es_tree_key,
            None,  # iterinfo=None → eval mode, no perturbation
            jnp.ones(4),
        )
        assert out.shape == (2,)
        assert jnp.all(jnp.isfinite(out))


# ── Training utilities ─────────────────────────────────────────────────────────


def _make_state(key: jax.Array) -> EGGROLLState:
    """Helper: small MLP → EGGROLLState."""
    model_key, es_key = jax.random.split(key)
    common_init = MLP.rand_init(
        model_key,
        in_dim=4,
        out_dim=2,
        hidden_dims=[8],
        use_bias=True,
        activation="relu",
        dtype="float32",
    )
    return init_eggroll_state(common_init, es_key, sigma=0.1, lr=1e-3)


class TestInitEggrollState:
    def test_all_fields_present(self):
        state = _make_state(jax.random.key(0))
        assert isinstance(state, EGGROLLState)
        assert isinstance(state.frozen_noiser_params, dict)
        assert isinstance(state.noiser_params, dict)
        assert "sigma" in state.noiser_params
        assert "opt_state" in state.noiser_params
        assert state.frozen_params is not None
        assert state.params is not None
        assert state.es_tree_key is not None
        assert state.es_map is not None

    def test_frozen_params_contains_activation(self):
        # MLP stores activation in frozen_params — callers need this during forward.
        state = _make_state(jax.random.key(1))
        assert isinstance(state.frozen_params, dict)
        assert "activation" in state.frozen_params

    def test_sigma_matches_requested(self):
        model_key, es_key = jax.random.split(jax.random.key(2))
        common_init = MLP.rand_init(
            model_key, in_dim=4, out_dim=2, hidden_dims=[8],
            use_bias=True, activation="relu", dtype="float32",
        )
        state = init_eggroll_state(common_init, es_key, sigma=0.05, lr=1e-3)
        assert state.noiser_params["sigma"] == 0.05


class TestGetIterinfos:
    def test_shapes_and_dtypes(self):
        epochs, thread_ids = get_iterinfos(epoch=5, num_envs=8)
        assert epochs.shape == (8,)
        assert thread_ids.shape == (8,)
        assert epochs.dtype == jnp.int32
        assert thread_ids.dtype == jnp.int32

    def test_values(self):
        epochs, thread_ids = get_iterinfos(epoch=5, num_envs=8)
        assert jnp.all(epochs == 5)
        assert jnp.array_equal(thread_ids, jnp.arange(8, dtype=jnp.int32))


class TestEggrollStep:
    def test_updates_params(self):
        """eggroll_step should change the parameter values."""
        state = _make_state(jax.random.key(10))
        num_envs = 8
        iterinfos = get_iterinfos(epoch=0, num_envs=num_envs)
        # Use varied fitnesses so normalised scores are non-zero.
        fitnesses = jnp.arange(num_envs, dtype=jnp.float32)
        new_state = eggroll_step(state, fitnesses, iterinfos)

        orig_leaves = jax.tree_util.tree_leaves(state.params)
        new_leaves = jax.tree_util.tree_leaves(new_state.params)
        total_change = sum(jnp.sum(jnp.abs(o - n)) for o, n in zip(orig_leaves, new_leaves))
        assert float(total_change) > 0.0

    def test_preserves_param_structure(self):
        """Pytree structure of params must be unchanged after an update."""
        state = _make_state(jax.random.key(11))
        num_envs = 8
        iterinfos = get_iterinfos(epoch=0, num_envs=num_envs)
        fitnesses = jnp.arange(num_envs, dtype=jnp.float32)
        new_state = eggroll_step(state, fitnesses, iterinfos)

        orig_leaves = jax.tree_util.tree_leaves(state.params)
        new_leaves = jax.tree_util.tree_leaves(new_state.params)
        assert len(orig_leaves) == len(new_leaves)
        for o, n in zip(orig_leaves, new_leaves):
            assert o.shape == n.shape
            assert o.dtype == n.dtype

    def test_does_not_mutate_input_state(self):
        """eggroll_step must not mutate the input state's noiser_params dict.

        EggRoll.do_updates modifies the dict in-place; eggroll_step must copy
        it first so callers that retain a reference to the old state see
        consistent values.
        """
        state = _make_state(jax.random.key(13))
        original_opt_state = state.noiser_params["opt_state"]
        num_envs = 8
        iterinfos = get_iterinfos(epoch=0, num_envs=num_envs)
        fitnesses = jnp.arange(num_envs, dtype=jnp.float32)
        _ = eggroll_step(state, fitnesses, iterinfos)
        # The original state's opt_state must be the same object as before.
        assert state.noiser_params["opt_state"] is original_opt_state

    def test_preserves_sigma(self):
        """eggroll_step must not modify sigma — decay is the caller's job."""
        sigma = 0.1
        state = _make_state(jax.random.key(12))
        num_envs = 8
        iterinfos = get_iterinfos(epoch=0, num_envs=num_envs)
        fitnesses = jnp.arange(num_envs, dtype=jnp.float32)
        new_state = eggroll_step(state, fitnesses, iterinfos)
        assert new_state.noiser_params["sigma"] == sigma

    def test_roundtrip_with_vmapped_forward(self):
        """Full cycle: init → get_iterinfos → vmapped forward → eggroll_step."""
        key = jax.random.key(20)
        model_key, es_key, data_key = jax.random.split(key, 3)
        num_envs = 8
        in_dim, out_dim = 4, 2

        common_init = MLP.rand_init(
            model_key, in_dim=in_dim, out_dim=out_dim, hidden_dims=[8],
            use_bias=True, activation="relu", dtype="float32",
        )
        state = init_eggroll_state(common_init, es_key, sigma=0.1, lr=1e-3)
        iterinfos = get_iterinfos(epoch=0, num_envs=num_envs)

        xs = jax.random.normal(data_key, (num_envs, in_dim))
        outputs = jax.vmap(
            lambda iterinfo, x: MLP.forward(
                EggRoll,
                state.frozen_noiser_params,
                state.noiser_params,
                state.frozen_params,
                state.params,
                state.es_tree_key,
                iterinfo,
                x,
            ),
            in_axes=(0, 0),
        )(iterinfos, xs)
        assert outputs.shape == (num_envs, out_dim)
        assert jnp.all(jnp.isfinite(outputs))

        # Negative MSE as fitness (higher = better).
        fitnesses = -jnp.mean((outputs - jnp.zeros_like(outputs)) ** 2, axis=-1)
        new_state = eggroll_step(state, fitnesses, iterinfos)

        assert isinstance(new_state, EGGROLLState)
        new_leaves = jax.tree_util.tree_leaves(new_state.params)
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in new_leaves)
