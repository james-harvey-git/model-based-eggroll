"""Domain-specific EGGROLL Model subclasses.

Defines DynamicsNet and PolicyNet as compositions of EGGROLL's MLP and Parameter
primitives (from mbrl.eggroll.primitives). Used exclusively by the EGGROLL training
paths (world_models/eggroll.py and policy_optimizers/eggroll.py). The MLE baseline
uses separate Flax models defined in world_models/mle.py.

These are network architectures only — analogous to SingleDynamicsModel (the Flax
nn.Module) in world_models/mle.py. They are not implementations of the
EnsembleDynamics ABC; the ABC implementations live in Phases 5/6.
"""

import jax
import jax.numpy as jnp

from mbrl.eggroll.primitives import (
    ACTIVATIONS,
    MLP,
    CommonInit,
    CommonParams,
    Linear,
    Model,
    Parameter,
    call_submodule,
    merge_frozen,
    merge_inits,
)

# ── ResidualMLP ────────────────────────────────────────────────────────────────


class ResidualMLP(Model):
    """Pre-activation ResNet backbone for DynamicsNet.

    Architecture:
        x = Linear_in(x)                              # in_dim → h
        for block in num_blocks:                      # each block consumes
            h_b = Linear_b(act(Linear_a(act(x))))     #   two hidden layers
            x = x + h_b                               # residual skip
        x = act(x)                                    # final pre-head activation
        x = Linear_out(x)                             # h → out_dim

    ``hidden_dims`` must be a non-empty even-length list of identical widths;
    ``num_blocks = len(hidden_dims) // 2``. The all-equal constraint exists
    because the residual stream has a single width throughout. The output of
    the final block is passed through ``act`` before ``Linear_out`` so the head
    sees bounded features (without it the additive residual accumulator can
    grow unbounded).
    """

    @classmethod
    def rand_init(
        cls,
        key,
        in_dim,
        out_dim,
        hidden_dims,
        use_bias,
        activation,
        dtype,
        init_scheme="eggroll",
    ):
        assert len(hidden_dims) > 0, "ResidualMLP requires hidden_dims to be non-empty"
        assert len(hidden_dims) % 2 == 0, (
            "ResidualMLP requires an even number of hidden layers (each residual "
            f"block consumes two); got {hidden_dims}"
        )
        assert len(set(hidden_dims)) == 1, (
            f"ResidualMLP requires all hidden_dims equal; got {hidden_dims}"
        )
        h = hidden_dims[0]
        num_blocks = len(hidden_dims) // 2

        keys = jax.random.split(key, 2 + 2 * num_blocks)
        sub_inits: dict[str, CommonInit] = {
            "in_lin": Linear.rand_init(
                keys[0], in_dim, h, use_bias, dtype, init_scheme=init_scheme
            ),
            "out_lin": Linear.rand_init(
                keys[1], h, out_dim, use_bias, dtype, init_scheme=init_scheme
            ),
        }
        for b in range(num_blocks):
            sub_inits[f"block_{b}_a"] = Linear.rand_init(
                keys[2 + 2 * b], h, h, use_bias, dtype, init_scheme=init_scheme
            )
            sub_inits[f"block_{b}_b"] = Linear.rand_init(
                keys[3 + 2 * b], h, h, use_bias, dtype, init_scheme=init_scheme
            )

        merged = merge_inits(**sub_inits)
        return merge_frozen(merged, activation=activation, num_blocks=num_blocks)

    @classmethod
    def _forward(cls, common_params, x):
        act = ACTIVATIONS[common_params.frozen_params["activation"]]
        num_blocks = common_params.frozen_params["num_blocks"]
        x = call_submodule(Linear, "in_lin", common_params, x)
        for b in range(num_blocks):
            h = act(x)
            h = call_submodule(Linear, f"block_{b}_a", common_params, h)
            h = act(h)
            h = call_submodule(Linear, f"block_{b}_b", common_params, h)
            x = x + h
        x = act(x)
        x = call_submodule(Linear, "out_lin", common_params, x)
        return x


_BACKBONES: dict[str, type[Model]] = {"mlp": MLP, "residual_mlp": ResidualMLP}

# ── DynamicsNet ────────────────────────────────────────────────────────────────


class DynamicsNet(Model):
    """Probabilistic dynamics network: (obs, action) → (mean, logvar).

    Predicts a Gaussian over (delta_obs, reward) — i.e. the output represents
    the *change* in observation, not next_obs directly. Callers reconstruct
    next_obs as ``obs + mean[..., :-1]``, matching the MLE baseline convention.

    Architecture analogue in world_models/mle.py:
      - Backbone MLP shape  →  SingleDynamicsModel (lines 27–38)
      - Learnable logvar bounds  →  EnsembleDynamicsModel (lines 41–81)

    The backbone is an MLP from (obs_dim + act_dim) through hidden_dims to
    2*(obs_dim+1). The output is split into mean and raw_logvar, then
    soft-clamped using learnable max_logvar / min_logvar Parameter objects.
    """

    @classmethod
    def rand_init(
        cls,
        key: jax.Array,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int],
        use_bias: bool = True,
        activation: str = "relu",
        dtype: str = "float32",
        init_scheme: str = "eggroll",
        backbone: str = "mlp",
        max_logvar_init: float = 0.5,
        min_logvar_init: float = -10.0,
    ) -> CommonInit:
        """Initialise DynamicsNet parameters.

        Args:
            key: JAX PRNG key.
            obs_dim: Observation dimension.
            act_dim: Action dimension.
            hidden_dims: Hidden layer sizes for the backbone. For
                ``backbone="residual_mlp"``, must be a non-empty even-length list
                of identical widths (one residual block per pair).
            use_bias: Whether to include bias terms in the linear layers.
            activation: Activation function name (``"relu"``, ``"silu"``, ``"pqn"``).
            dtype: Parameter dtype (e.g. ``"float32"``).
            init_scheme: Linear-layer initialisation scheme (e.g. ``"eggroll"``,
                ``"flax_dense"``).
            backbone: Backbone architecture — ``"mlp"`` (default, flat MLP)
                or ``"residual_mlp"`` (pre-activation ResNet blocks).
            max_logvar_init: Initial value for the learnable logvar upper bound.
            min_logvar_init: Initial value for the learnable logvar lower bound.

        Returns:
            CommonInit with params structure:
              ``{"backbone": ..., "max_logvar": array, "min_logvar": array}``
        """
        if backbone not in _BACKBONES:
            raise ValueError(
                f"Unknown backbone {backbone!r}; expected one of {sorted(_BACKBONES)}"
            )
        backbone_cls = _BACKBONES[backbone]
        backbone_key, _ = jax.random.split(key)
        backbone_init = backbone_cls.rand_init(
            backbone_key,
            in_dim=obs_dim + act_dim,
            out_dim=2 * (obs_dim + 1),
            hidden_dims=hidden_dims,
            use_bias=use_bias,
            activation=activation,
            dtype=dtype,
            init_scheme=init_scheme,
        )
        # key is reused here because raw_value is provided — the key is never used
        max_logvar = Parameter.rand_init(
            key, shape=None, scale=None,
            raw_value=jnp.full((obs_dim + 1,), max_logvar_init, dtype=dtype),
            dtype=dtype,
        )
        min_logvar = Parameter.rand_init(
            key, shape=None, scale=None,
            raw_value=jnp.full((obs_dim + 1,), min_logvar_init, dtype=dtype),
            dtype=dtype,
        )
        merged = merge_inits(
            backbone=backbone_init, max_logvar=max_logvar, min_logvar=min_logvar
        )
        # Stash backbone choice so _forward_with_bounds dispatches to the right class.
        # Old checkpoints without this key fall back to "mlp" via .get().
        return merge_frozen(merged, backbone_type=backbone)

    @classmethod
    def _forward(
        cls,
        common_params: CommonParams,
        obs: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass: (obs, action) → (mean, logvar).

        Args:
            common_params: Full CommonParams bundle (noiser, params, keys, etc.).
            obs: Observation vector, shape ``(obs_dim,)``.
            action: Action vector, shape ``(act_dim,)``.

        Returns:
            Tuple of ``(mean, logvar)``, each shape ``(obs_dim + 1,)``.
            ``mean[..., :-1]`` is delta_obs, ``mean[..., -1]`` is reward.
        """
        mean, logvar, _, _ = cls._forward_with_bounds(common_params, obs, action)
        return mean, logvar

    @classmethod
    def _forward_noisy_with_bounds(
        cls,
        noiser,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        iterinfo,
        obs: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Private helper returning bounds alongside predictions for training losses."""
        return cls._forward_with_bounds(
            CommonParams(
                noiser,
                frozen_noiser_params,
                noiser_params,
                frozen_params,
                params,
                es_tree_key,
                iterinfo,
            ),
            obs,
            action,
        )

    @classmethod
    def _forward_with_bounds(
        cls,
        common_params: CommonParams,
        obs: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Forward pass returning mean, logvar, max_logvar, and min_logvar."""
        obs_action = jnp.concatenate([obs, action], axis=-1)
        # .get(..., "mlp") keeps pre-residual checkpoints loadable; their
        # frozen_params predates the backbone_type key.
        backbone_name = (
            common_params.frozen_params.get("backbone_type", "mlp")
            if common_params.frozen_params is not None
            else "mlp"
        )
        backbone_cls = _BACKBONES[backbone_name]
        output = call_submodule(backbone_cls, "backbone", common_params, obs_action)
        half = output.shape[-1] // 2
        mean, raw_logvar = output[:half], output[half:]

        # Soft-clamp log-variance to prevent unbounded growth (matching mle.py)
        max_logvar = call_submodule(Parameter, "max_logvar", common_params)
        min_logvar = call_submodule(Parameter, "min_logvar", common_params)
        logvar = max_logvar - jax.nn.softplus(max_logvar - raw_logvar)
        logvar = min_logvar + jax.nn.softplus(logvar - min_logvar)

        return mean, logvar, max_logvar, min_logvar


# ── PolicyNet ──────────────────────────────────────────────────────────────────


class PolicyNet(Model):
    """Deterministic policy network: obs → tanh-squashed action.

    A single MLP with tanh applied to the output, producing actions bounded
    in ``(-1, 1)^act_dim``. Callers rescale to the environment's action range
    as needed.

    Design note: EGGROLL policy search uses population-based perturbations for
    exploration rather than a stochastic policy, so a deterministic tanh head
    is the natural choice here.
    """

    @classmethod
    def rand_init(
        cls,
        key: jax.Array,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int],
        use_bias: bool = True,
        activation: str = "relu",
        dtype: str = "float32",
        init_scheme: str = "eggroll",
    ) -> CommonInit:
        """Initialise PolicyNet parameters.

        Returns the raw MLP CommonInit — no merge_inits wrapper since there is
        only one sub-module. ``common_params.params`` in ``_forward`` is therefore
        the MLP's layer dict directly.
        """
        return MLP.rand_init(
            key,
            obs_dim,
            act_dim,
            hidden_dims,
            use_bias,
            activation,
            dtype,
            init_scheme=init_scheme,
        )

    @classmethod
    def _forward(
        cls,
        common_params: CommonParams,
        obs: jax.Array,
    ) -> jax.Array:
        """Forward pass: obs → bounded action.

        Args:
            common_params: Full CommonParams bundle.
            obs: Observation vector, shape ``(obs_dim,)``.

        Returns:
            Action vector, shape ``(act_dim,)``, values in ``(-1, 1)``.
        """
        # Delegate directly to MLP._forward rather than call_submodule because
        # rand_init returns the MLP's CommonInit unwrapped — common_params.params
        # IS the MLP layer dict, not a nested dict with a "backbone" key.
        raw = MLP._forward(common_params, obs)
        return jnp.tanh(raw)
