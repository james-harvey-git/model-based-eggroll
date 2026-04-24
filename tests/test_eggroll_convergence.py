"""EGGROLL convergence and scale tests (Steps 17 and 18 in PLAN.md).

Step 17 — toy convergence (TestToyConvergence):
  Replicates the eggroll.ipynb x² + x + 1 regression example using
  primitives.py and training.py. Verifies end-to-end correctness of the
  vendored EGGROLL pipeline in a low-budget setting (64 envs, 300 epochs).

Step 18 — network-scale stress test (TestNetworkScale):
  Runs DynamicsNet / PolicyNet at HalfCheetah scale (obs=17, act=6,
  hidden=[256, 256]) with population_size=256. Confirms vmap over the
  population works, outputs are finite and distinct across perturbations,
  and prints peak process RSS for manual inspection.
"""

import sys

import jax
import jax.numpy as jnp
import pytest

from mbrl.eggroll.networks import DynamicsNet, PolicyNet
from mbrl.eggroll.primitives import MLP, EggRoll
from mbrl.eggroll.training import eggroll_step, get_iterinfos, init_eggroll_state

# HalfCheetah dimensions (configs/networks/mlp.yaml and D4RL env spec).
_HC_OBS_DIM = 17
_HC_ACT_DIM = 6
_HC_HIDDEN = [256, 256]
_HC_NUM_ENVS = 256  # reduced test budget; production config may use a larger population


def _rss_mb() -> float:
    """Current process RSS in MB; returns 0.0 if resource module is unavailable."""
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: ru_maxrss in bytes; Linux: ru_maxrss in KB.
        factor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return ru.ru_maxrss / factor
    except Exception:
        return 0.0


# ── Step 17: Toy convergence ───────────────────────────────────────────────────


class TestToyConvergence:
    """Replicate the eggroll.ipynb x² + x + 1 regression toy via our training utilities.

    Low-budget replication: 64 perturbations, 300 epochs, hidden=[32, 32].
    Follows the notebook training loop exactly (JIT-compiled forward + update,
    manual linear sigma decay) but via init_eggroll_state / EggRoll primitives
    rather than the notebook's inline setup.

    Goal: correctness (vendored primitives compose with training.py end-to-end),
    not peak convergence speed.
    """

    @pytest.mark.slow
    def test_loss_decreases_on_quadratic(self):
        """MSE on x² + x + 1 should halve after 300 epochs with 64 perturbations."""
        rng = jax.random.key(42)
        data_key, model_key, es_key = jax.random.split(rng, 3)

        n_data = 1024
        x_train = jax.random.uniform(data_key, (n_data,), minval=-5.0, maxval=5.0)
        y_train = x_train**2 + x_train + 1.0

        num_envs = 64
        num_epochs = 300
        sigma_init = 1.0

        common_init = MLP.rand_init(
            model_key,
            in_dim=1,
            out_dim=1,
            hidden_dims=[32, 32],
            use_bias=True,
            activation="relu",
            dtype="float32",
        )
        state = init_eggroll_state(common_init, es_key, sigma=sigma_init, lr=1e-2)

        # Extract frozen parts (fixed after init) as closure variables for JIT.
        # Following the notebook pattern: frozen_noiser_params and es_tree_key are
        # captured as Python-level constants; mutable noiser_params / params are
        # passed as dynamic arguments.
        fnp = state.frozen_noiser_params
        fp = state.frozen_params
        etk = state.es_tree_key
        em = state.es_map

        jit_forward = jax.jit(
            jax.vmap(
                lambda n, p, it, xi: MLP.forward(EggRoll, fnp, n, fp, p, etk, it, xi),
                in_axes=(None, None, 0, 0),
            )
        )
        jit_eval = jax.jit(
            jax.vmap(
                lambda n, p, xi: MLP.forward(EggRoll, fnp, n, fp, p, etk, None, xi),
                in_axes=(None, None, 0),
            )
        )
        # do_updates mutates its second arg in-place; the JIT boundary keeps the
        # mutation inside the trace and returns the updated dict correctly.
        jit_update = jax.jit(
            lambda n, p, f, it: EggRoll.do_updates(fnp, n, p, etk, f, it, em)
        )

        noiser_params = state.noiser_params
        params = state.params

        def eval_mse() -> float:
            preds = jit_eval(noiser_params, params, x_train[:, None])[:, 0]
            return float(jnp.mean((preds - y_train) ** 2))

        initial_mse = eval_mse()

        for epoch in range(num_epochs):
            iterinfos = get_iterinfos(epoch, num_envs)
            # Cycle through training data deterministically across epochs.
            idx = (jnp.arange(num_envs) + epoch * num_envs) % n_data
            batch_x = x_train[idx][:, None]  # (num_envs, 1)
            batch_y = y_train[idx]  # (num_envs,)

            preds = jit_forward(noiser_params, params, iterinfos, batch_x)[:, 0]
            raw_fitnesses = -(preds - batch_y) ** 2  # per-perturbation, higher is better
            normalized = EggRoll.convert_fitnesses(fnp, noiser_params, raw_fitnesses)
            noiser_params, params = jit_update(noiser_params, params, normalized, iterinfos)
            # Linear sigma decay — caller's responsibility per eggroll_step docstring.
            noiser_params["sigma"] = sigma_init * (1 - (epoch + 1) / num_epochs)

        final_mse = eval_mse()
        assert final_mse < initial_mse * 0.5, (
            f"Loss should halve over {num_epochs} epochs: "
            f"initial={initial_mse:.2f}, final={final_mse:.2f}"
        )


# ── Step 18: Network-scale stress test ────────────────────────────────────────


class TestNetworkScale:
    """DynamicsNet / PolicyNet at HalfCheetah scale with realistic population.

    Uses obs_dim=17, act_dim=6, hidden=[256, 256], pop=256 as a manageable
    test-budget stress case. Confirms:
      - vmap over the full population produces correct shapes
      - all outputs are finite
      - each perturbation produces a distinct output (vmap is non-trivial)
      - one eggroll_step completes without error
    Peak process RSS is printed for manual inspection.
    """

    @pytest.mark.slow
    def test_dynamics_net_scale(self):
        """DynamicsNet: forward + update at HalfCheetah scale, pop=256."""
        key = jax.random.key(50)
        init_key, es_key, data_key = jax.random.split(key, 3)

        init = DynamicsNet.rand_init(init_key, _HC_OBS_DIM, _HC_ACT_DIM, _HC_HIDDEN)
        state = init_eggroll_state(init, es_key, sigma=0.01, lr=1e-3)

        rss_before = _rss_mb()

        iterinfos = get_iterinfos(epoch=0, num_envs=_HC_NUM_ENVS)
        obs = jax.random.normal(data_key, (_HC_NUM_ENVS, _HC_OBS_DIM))
        actions = jax.random.normal(data_key, (_HC_NUM_ENVS, _HC_ACT_DIM))

        means, logvars = jax.vmap(
            lambda it, o, a: DynamicsNet.forward(
                EggRoll,
                state.frozen_noiser_params,
                state.noiser_params,
                state.frozen_params,
                state.params,
                state.es_tree_key,
                it,
                o,
                a,
            ),
            in_axes=(0, 0, 0),
        )(iterinfos, obs, actions)
        jax.block_until_ready((means, logvars))

        rss_after = _rss_mb()
        print(
            f"\n[Step 18] DynamicsNet pop={_HC_NUM_ENVS} obs={_HC_OBS_DIM} "
            f"act={_HC_ACT_DIM} hidden={_HC_HIDDEN}  "
            f"RSS: {rss_before:.0f} → {rss_after:.0f} MB "
            f"(+{rss_after - rss_before:.0f} MB)"
        )

        assert means.shape == (_HC_NUM_ENVS, _HC_OBS_DIM + 1)
        assert logvars.shape == (_HC_NUM_ENVS, _HC_OBS_DIM + 1)
        assert jnp.all(jnp.isfinite(means))
        assert jnp.all(jnp.isfinite(logvars))
        # Perturbations must differ — if all identical, vmap didn't apply EGGROLL noise.
        assert not jnp.allclose(means[0], means[1], atol=1e-6), (
            "All perturbations produced identical outputs — EGGROLL noise not applied"
        )

        # One update step completes and yields finite params.
        fitnesses = -jnp.mean(means**2, axis=-1)
        new_state = eggroll_step(state, fitnesses, iterinfos)
        assert all(
            jnp.all(jnp.isfinite(leaf))
            for leaf in jax.tree_util.tree_leaves(new_state.params)
        )

    @pytest.mark.slow
    def test_policy_net_scale(self):
        """PolicyNet: forward + update at HalfCheetah scale, pop=256."""
        key = jax.random.key(51)
        init_key, es_key, data_key = jax.random.split(key, 3)

        init = PolicyNet.rand_init(init_key, _HC_OBS_DIM, _HC_ACT_DIM, _HC_HIDDEN)
        state = init_eggroll_state(init, es_key, sigma=0.01, lr=1e-3)

        iterinfos = get_iterinfos(epoch=0, num_envs=_HC_NUM_ENVS)
        obs = jax.random.normal(data_key, (_HC_NUM_ENVS, _HC_OBS_DIM))

        actions = jax.vmap(
            lambda it, o: PolicyNet.forward(
                EggRoll,
                state.frozen_noiser_params,
                state.noiser_params,
                state.frozen_params,
                state.params,
                state.es_tree_key,
                it,
                o,
            ),
            in_axes=(0, 0),
        )(iterinfos, obs)
        jax.block_until_ready(actions)

        assert actions.shape == (_HC_NUM_ENVS, _HC_ACT_DIM)
        assert jnp.all(jnp.isfinite(actions))
        assert jnp.all(actions > -1.0) and jnp.all(actions < 1.0)
        assert not jnp.allclose(actions[0], actions[1], atol=1e-6), (
            "All perturbations produced identical outputs — EGGROLL noise not applied"
        )

        # One update step completes and yields finite params.
        fitnesses = jnp.sum(actions, axis=-1)
        new_state = eggroll_step(state, fitnesses, iterinfos)
        assert all(
            jnp.all(jnp.isfinite(leaf))
            for leaf in jax.tree_util.tree_leaves(new_state.params)
        )
