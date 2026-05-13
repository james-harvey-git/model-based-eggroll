"""Tests for the EGGROLL integration layer (primitives and training utilities).

Organised in sections:
  - TestVendoredPrimitives  — smoke tests for mbrl/eggroll/primitives.py (vendored
                               from HyperscaleES; intentionally minimal).
  - TestInitEggrollState    — mbrl/eggroll/training.py: EGGROLLState construction.
  - TestGetIterinfos        — mbrl/eggroll/training.py: iterinfo generation.
  - TestEggrollStep         — mbrl/eggroll/training.py: update cycle.
  - TestDynamicsNet         — mbrl/eggroll/networks.py: DynamicsNet architecture.
  - TestPolicyNet           — mbrl/eggroll/networks.py: PolicyNet architecture.
"""

from flax.linen.linear import default_kernel_init
import jax
import jax.numpy as jnp
import jax.tree_util
import optax
import pytest

from mbrl.eggroll.networks import DynamicsNet, PolicyNet
from mbrl.eggroll.primitives import LOGVAR_PARAM, MLP, MM_PARAM, PARAM, EggRoll, Linear
from mbrl.eggroll.training import (
    EGGROLLState,
    build_sigma_tree,
    eggroll_step,
    get_iterinfos,
    init_eggroll_state,
    resolve_optax_solver,
)

# Dimensions used for network tests — lightweight smoke-test sizes.
# HalfCheetah in this repo is (17, 6); use smaller dims here for speed.
_OBS_DIM = 11
_ACT_DIM = 3
_HIDDEN = [32, 32]

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

    def test_linear_flax_dense_init_matches_flax_default(self):
        key = jax.random.key(123)
        in_dim, out_dim = 4, 3
        init = Linear.rand_init(
            key,
            in_dim=in_dim,
            out_dim=out_dim,
            use_bias=True,
            dtype="float32",
            init_scheme="flax_dense",
        )

        expected_weight = default_kernel_init(key, (in_dim, out_dim), jnp.float32).T
        weight = init.params["weight"]
        bias = init.params["bias"]

        assert weight.shape == (out_dim, in_dim)
        assert jnp.array_equal(weight, expected_weight)
        assert jnp.array_equal(bias, jnp.zeros((out_dim,), dtype=jnp.float32))

    def test_linear_unknown_init_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported linear init scheme"):
            Linear.rand_init(
                jax.random.key(124),
                in_dim=4,
                out_dim=3,
                use_bias=True,
                dtype="float32",
                init_scheme="unknown",
            )


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
        # Sigma is stored as a per-leaf tree mirroring params; a scalar input
        # splats to a uniform tree where every leaf == the scalar.
        for leaf in jax.tree.leaves(state.noiser_params["sigma"]):
            assert jnp.allclose(leaf, 0.05)
        assert jax.tree.structure(state.noiser_params["sigma"]) == jax.tree.structure(state.params)

    def test_accepts_nondefault_solver(self):
        model_key, es_key = jax.random.split(jax.random.key(3))
        common_init = MLP.rand_init(
            model_key, in_dim=4, out_dim=2, hidden_dims=[8],
            use_bias=True, activation="relu", dtype="float32",
        )
        state = init_eggroll_state(
            common_init,
            es_key,
            sigma=0.05,
            lr=1e-3,
            solver=resolve_optax_solver("adam"),
        )
        assert state.noiser_params["opt_state"] is not None


class TestBuildSigmaTree:
    """build_sigma_tree dispatches per-leaf sigmas off es_map markers (#32)."""

    def test_dispatches_on_es_map_markers(self):
        # Construct a synthetic params/es_map with one leaf per relevant marker.
        params = {
            "lora_weight": jnp.zeros((3, 4)),
            "nonlora_bias": jnp.zeros((4,)),
            "max_logvar": jnp.zeros((4,)),
        }
        es_map = {
            "lora_weight": MM_PARAM,
            "nonlora_bias": PARAM,
            "max_logvar": LOGVAR_PARAM,
        }
        groups = {"lora": 0.1, "nonlora": 0.01, "logvar": 0.001}
        sigma_tree = build_sigma_tree(params, es_map, groups)
        assert jnp.allclose(sigma_tree["lora_weight"], 0.1)
        assert jnp.allclose(sigma_tree["nonlora_bias"], 0.01)
        assert jnp.allclose(sigma_tree["max_logvar"], 0.001)

    def test_structure_mirrors_params(self):
        params = {"a": jnp.zeros(3), "b": {"c": jnp.zeros((2, 2))}}
        es_map = {"a": PARAM, "b": {"c": MM_PARAM}}
        groups = {"lora": 1.0, "nonlora": 2.0, "logvar": 3.0}
        sigma_tree = build_sigma_tree(params, es_map, groups)
        assert jax.tree.structure(sigma_tree) == jax.tree.structure(params)
        # Leaves are scalar float32.
        for leaf in jax.tree.leaves(sigma_tree):
            assert leaf.shape == ()
            assert leaf.dtype == jnp.float32


class TestResolveOptaxSolver:
    def test_supported_solver_names(self):
        assert resolve_optax_solver("sgd") is optax.sgd
        assert resolve_optax_solver("adam") is optax.adam
        assert resolve_optax_solver("adamw") is optax.adamw

    def test_solver_names_are_case_insensitive(self):
        assert resolve_optax_solver("AdamW") is optax.adamw

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError, match="Unsupported EGGROLL solver"):
            resolve_optax_solver("rmsprop")


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
        # Sigma is a per-leaf tree; assert every leaf still equals the
        # requested sigma (eggroll_step does not decay).
        for leaf in jax.tree.leaves(new_state.noiser_params["sigma"]):
            assert jnp.allclose(leaf, sigma)

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


# ── DynamicsNet ────────────────────────────────────────────────────────────────


class TestDynamicsNet:
    def test_rand_init(self):
        init = DynamicsNet.rand_init(
            jax.random.key(30), _OBS_DIM, _ACT_DIM, _HIDDEN,
        )
        assert set(init.params.keys()) == {"backbone", "max_logvar", "min_logvar"}
        assert init.frozen_params["backbone"]["activation"] == "relu"
        assert init.params["max_logvar"].shape == (_OBS_DIM + 1,)
        assert init.params["min_logvar"].shape == (_OBS_DIM + 1,)

    def test_logvar_bounds_tagged_with_logvar_param(self):
        """max_logvar/min_logvar carry the LOGVAR_PARAM es_map marker (issue #32).

        Backbone leaves keep their standard markers (MM_PARAM for matmul
        weights, PARAM for biases), so the per-group sigma classifier sees
        the right structure.
        """
        init = DynamicsNet.rand_init(
            jax.random.key(30), _OBS_DIM, _ACT_DIM, _HIDDEN,
        )
        assert init.es_map["max_logvar"] == LOGVAR_PARAM
        assert init.es_map["min_logvar"] == LOGVAR_PARAM
        # Backbone is an MLP → Linear → {weight (MM_PARAM), bias (PARAM)}.
        backbone_layer_0 = init.es_map["backbone"]["0"]
        assert backbone_layer_0["weight"] == MM_PARAM
        assert backbone_layer_0["bias"] == PARAM

    def test_forward_eval_shape(self):
        key = jax.random.key(30)
        init = DynamicsNet.rand_init(key, _OBS_DIM, _ACT_DIM, _HIDDEN)
        state = init_eggroll_state(init, key, sigma=0.1, lr=1e-3)
        obs = jnp.zeros(_OBS_DIM)
        action = jnp.zeros(_ACT_DIM)
        mean, logvar = DynamicsNet.forward(
            EggRoll,
            state.frozen_noiser_params,
            state.noiser_params,
            state.frozen_params,
            state.params,
            state.es_tree_key,
            None,  # eval mode
            obs,
            action,
        )
        assert mean.shape == (_OBS_DIM + 1,)
        assert logvar.shape == (_OBS_DIM + 1,)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))

    def test_logvar_clamped(self):
        key = jax.random.key(30)
        init = DynamicsNet.rand_init(
            key, _OBS_DIM, _ACT_DIM, _HIDDEN,
            max_logvar_init=0.5, min_logvar_init=-10.0,
        )
        state = init_eggroll_state(init, key, sigma=0.1, lr=1e-3)
        obs = jax.random.normal(key, (_OBS_DIM,))
        action = jax.random.normal(key, (_ACT_DIM,))
        _, logvar = DynamicsNet.forward(
            EggRoll,
            state.frozen_noiser_params,
            state.noiser_params,
            state.frozen_params,
            state.params,
            state.es_tree_key,
            None,
            obs,
            action,
        )
        assert jnp.all(logvar <= 0.5)
        assert jnp.all(logvar >= -10.0)

    def test_private_forward_with_bounds_matches_forward(self):
        key = jax.random.key(30)
        init = DynamicsNet.rand_init(key, _OBS_DIM, _ACT_DIM, _HIDDEN)
        state = init_eggroll_state(init, key, sigma=0.1, lr=1e-3)
        obs = jax.random.normal(key, (_OBS_DIM,))
        action = jax.random.normal(key, (_ACT_DIM,))
        mean, logvar = DynamicsNet.forward(
            EggRoll,
            state.frozen_noiser_params,
            state.noiser_params,
            state.frozen_params,
            state.params,
            state.es_tree_key,
            None,
            obs,
            action,
        )
        mean_aux, logvar_aux, max_logvar, min_logvar = DynamicsNet._forward_noisy_with_bounds(
            EggRoll,
            state.frozen_noiser_params,
            state.noiser_params,
            state.frozen_params,
            state.params,
            state.es_tree_key,
            None,
            obs,
            action,
        )
        assert jnp.allclose(mean, mean_aux)
        assert jnp.allclose(logvar, logvar_aux)
        assert max_logvar.shape == (_OBS_DIM + 1,)
        assert min_logvar.shape == (_OBS_DIM + 1,)

    def test_forward_train_shape(self):
        num_envs = 8
        key = jax.random.key(30)
        init = DynamicsNet.rand_init(key, _OBS_DIM, _ACT_DIM, _HIDDEN)
        state = init_eggroll_state(init, key, sigma=0.1, lr=1e-3)
        iterinfos = get_iterinfos(epoch=0, num_envs=num_envs)
        obs = jax.random.normal(key, (num_envs, _OBS_DIM))
        action = jax.random.normal(key, (num_envs, _ACT_DIM))
        mean, logvar = jax.vmap(
            lambda iterinfo, o, a: DynamicsNet.forward(
                EggRoll,
                state.frozen_noiser_params,
                state.noiser_params,
                state.frozen_params,
                state.params,
                state.es_tree_key,
                iterinfo,
                o,
                a,
            ),
            in_axes=(0, 0, 0),
        )(iterinfos, obs, action)
        assert mean.shape == (num_envs, _OBS_DIM + 1)
        assert logvar.shape == (num_envs, _OBS_DIM + 1)


# ── Per-group sigma ────────────────────────────────────────────────────────────


class TestPerGroupSigma:
    """End-to-end checks for per-leaf sigma plumbing on DynamicsNet.

    Uses DynamicsNet specifically because it's the only network in the repo
    that emits LOGVAR_PARAM leaves alongside MM_PARAM and PARAM.
    """

    @staticmethod
    def _make_dyn_state(key, sigma_groups):
        """Build a DynamicsNet EGGROLLState with per-group sigmas."""
        model_key, es_key = jax.random.split(key)
        common_init = DynamicsNet.rand_init(
            model_key, _OBS_DIM, _ACT_DIM, _HIDDEN,
        )
        sigma_tree = build_sigma_tree(
            common_init.params, common_init.es_map, sigma_groups,
        )
        # Use SGD so update magnitudes are directly proportional to gradient
        # magnitudes (Adam would normalise across the per-group scale).
        return init_eggroll_state(
            common_init, es_key, sigma=sigma_tree, lr=1e-2, solver=optax.sgd,
        )

    def test_logvar_sigma_zero_freezes_bounds(self):
        """With sigma_logvar=0, max/min_logvar are byte-equal after a step."""
        state = self._make_dyn_state(
            jax.random.key(50),
            sigma_groups={"lora": 0.1, "nonlora": 0.1, "logvar": 0.0},
        )
        num_envs = 8
        iterinfos = get_iterinfos(epoch=0, num_envs=num_envs)
        fitnesses = jnp.arange(num_envs, dtype=jnp.float32)
        new_state = eggroll_step(state, fitnesses, iterinfos)

        # Logvar bounds unchanged.
        assert jnp.array_equal(
            new_state.params["max_logvar"], state.params["max_logvar"]
        )
        assert jnp.array_equal(
            new_state.params["min_logvar"], state.params["min_logvar"]
        )

        # Backbone leaves did change (sanity that the step ran nontrivially).
        backbone_changed = any(
            not jnp.array_equal(o, n)
            for o, n in zip(
                jax.tree.leaves(state.params["backbone"]),
                jax.tree.leaves(new_state.params["backbone"]),
            )
        )
        assert backbone_changed

    def test_per_group_sigma_takes_effect(self):
        """Update magnitudes follow the per-group sigma ordering (SGD).

        With sigma_lora >> sigma_nonlora >> sigma_logvar (all non-zero), the
        ES gradient estimate magnitude scales with sigma, and SGD's step does
        too. Loose ordering check — allow for stochasticity by averaging the
        L2 update magnitude per group across leaves.
        """
        state = self._make_dyn_state(
            jax.random.key(51),
            sigma_groups={"lora": 0.5, "nonlora": 0.05, "logvar": 0.005},
        )
        num_envs = 16
        iterinfos = get_iterinfos(epoch=0, num_envs=num_envs)
        fitnesses = jax.random.normal(jax.random.key(99), (num_envs,))
        new_state = eggroll_step(state, fitnesses, iterinfos)

        def avg_l2_delta(orig_tree, new_tree):
            deltas = [
                jnp.linalg.norm((n - o).ravel()) / jnp.sqrt(n.size)
                for o, n in zip(jax.tree.leaves(orig_tree), jax.tree.leaves(new_tree))
            ]
            return float(jnp.mean(jnp.stack(deltas)))

        # Bucket params by group via es_map. backbone "weight" leaves are MM_PARAM,
        # backbone "bias" leaves are PARAM, max/min_logvar are LOGVAR_PARAM.
        lora_orig = {k: v["weight"] for k, v in state.params["backbone"].items()}
        lora_new = {k: v["weight"] for k, v in new_state.params["backbone"].items()}
        nonlora_orig = {k: v["bias"] for k, v in state.params["backbone"].items()}
        nonlora_new = {k: v["bias"] for k, v in new_state.params["backbone"].items()}
        logvar_orig = {
            "max": state.params["max_logvar"], "min": state.params["min_logvar"],
        }
        logvar_new = {
            "max": new_state.params["max_logvar"], "min": new_state.params["min_logvar"],
        }

        lora_delta = avg_l2_delta(lora_orig, lora_new)
        nonlora_delta = avg_l2_delta(nonlora_orig, nonlora_new)
        logvar_delta = avg_l2_delta(logvar_orig, logvar_new)

        # Strict ordering of average update magnitude.
        assert lora_delta > nonlora_delta > logvar_delta > 0.0

    def test_per_group_decay_applied(self):
        """init_eggroll_state stores per-group decay_tree in frozen_noiser_params.

        Verifies that decay rates classified by es_map land on the right
        leaves and that a manual N-step decay matches expected per-group
        sigma values.
        """
        model_key, es_key = jax.random.split(jax.random.key(60))
        common_init = DynamicsNet.rand_init(
            model_key, _OBS_DIM, _ACT_DIM, _HIDDEN,
        )
        state = init_eggroll_state(
            common_init, es_key,
            sigma={"lora": 0.1, "nonlora": 0.05, "logvar": 0.001},
            lr=1e-3,
            sigma_decay_rate={"lora": 0.9, "nonlora": 0.8, "logvar": 0.5},
        )
        decay_tree = state.frozen_noiser_params["sigma_decay_rate"]
        assert jnp.allclose(decay_tree["max_logvar"], 0.5)
        assert jnp.allclose(decay_tree["min_logvar"], 0.5)
        assert jnp.allclose(decay_tree["backbone"]["0"]["weight"], 0.9)
        assert jnp.allclose(decay_tree["backbone"]["0"]["bias"], 0.8)

        # Apply the decay step manually three times and verify per-group sigmas.
        sigma_tree = state.noiser_params["sigma"]
        for _ in range(3):
            sigma_tree = jax.tree.map(lambda s, d: s * d, sigma_tree, decay_tree)
        assert jnp.allclose(sigma_tree["max_logvar"], 0.001 * 0.5 ** 3)
        assert jnp.allclose(sigma_tree["backbone"]["0"]["weight"], 0.1 * 0.9 ** 3)
        assert jnp.allclose(sigma_tree["backbone"]["0"]["bias"], 0.05 * 0.8 ** 3)


# ── PolicyNet ──────────────────────────────────────────────────────────────────


class TestPolicyNet:
    def test_rand_init(self):
        init = PolicyNet.rand_init(
            jax.random.key(31), _OBS_DIM, _ACT_DIM, _HIDDEN,
        )
        assert init.frozen_params["activation"] == "relu"

    def test_forward_eval_shape(self):
        key = jax.random.key(31)
        init = PolicyNet.rand_init(key, _OBS_DIM, _ACT_DIM, _HIDDEN)
        state = init_eggroll_state(init, key, sigma=0.1, lr=1e-3)
        obs = jnp.zeros(_OBS_DIM)
        action = PolicyNet.forward(
            EggRoll,
            state.frozen_noiser_params,
            state.noiser_params,
            state.frozen_params,
            state.params,
            state.es_tree_key,
            None,
            obs,
        )
        assert action.shape == (_ACT_DIM,)
        assert jnp.all(action > -1.0)
        assert jnp.all(action < 1.0)

    def test_forward_train_shape(self):
        num_envs = 8
        key = jax.random.key(31)
        init = PolicyNet.rand_init(key, _OBS_DIM, _ACT_DIM, _HIDDEN)
        state = init_eggroll_state(init, key, sigma=0.1, lr=1e-3)
        iterinfos = get_iterinfos(epoch=0, num_envs=num_envs)
        obs = jax.random.normal(key, (num_envs, _OBS_DIM))
        actions = jax.vmap(
            lambda iterinfo, o: PolicyNet.forward(
                EggRoll,
                state.frozen_noiser_params,
                state.noiser_params,
                state.frozen_params,
                state.params,
                state.es_tree_key,
                iterinfo,
                o,
            ),
            in_axes=(0, 0),
        )(iterinfos, obs)
        assert actions.shape == (num_envs, _ACT_DIM)
