"""Tests for vendored EGGROLL primitives (mbrl/eggroll/primitives.py).

These tests are intentionally minimal. The primitives are vendored third-party
code (HyperscaleES); we don't test their internal correctness here. What we do
verify:
  1. All expected names are importable (catches accidental omissions in vendoring).
  2. A basic end-to-end forward pass works (catches consolidation mistakes such as
     wrong definition ordering or missing cross-references from merging 4 files into 1).
"""

import jax
import jax.numpy as jnp


class TestVendoredPrimitives:
    def test_all_names_importable(self):
        from mbrl.eggroll.primitives import (  # noqa: F401
            EXCLUDED,
            EMB_PARAM,
            MM_PARAM,
            PARAM,
            CommonInit,
            CommonParams,
            EggRoll,
            Embedding,
            Linear,
            MLP,
            MM,
            Model,
            Noiser,
            Parameter,
            TMM,
            call_submodule,
            merge_frozen,
            merge_inits,
            simple_es_tree_key,
        )

    def test_mlp_forward_pass(self):
        """Smoke test: init an MLP, generate es_tree_key, run a forward pass in eval
        mode (iterinfo=None). Verifies the 4-file consolidation didn't break any
        cross-references between Noiser, Model, and the helper functions."""
        from mbrl.eggroll.primitives import EggRoll, MLP, simple_es_tree_key

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
