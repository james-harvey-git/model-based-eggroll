"""EGGROLL-trained ensemble world model.

Uses a single DynamicsNet trained with EGGROLL. At inference, perturbed population
members serve as the ensemble (population-as-ensemble). Training runs inside a
jax.lax.fori_loop so the entire loop is JIT-compiled.

Rollout semantics:
- step() uses iterinfo=None (unperturbed base params) for rollout dynamics. The base
  parameters are the trained best estimate; perturbations are EGGROLL exploration noise,
  not independently-trained models. Aleatoric noise comes from sampling N(mean, logvar).
- predict_ensemble() uses N perturbed members to estimate epistemic uncertainty for
  model-based policy algorithms (e.g. MOPO's penalty term).
"""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from mbrl.data import Transition, train_val_split
from mbrl.eggroll.networks import DynamicsNet
from mbrl.eggroll.primitives import EggRoll
from mbrl.eggroll.training import EGGROLLState, get_iterinfos, init_eggroll_state
from mbrl.world_models.base import EnsembleDynamics
from mbrl.world_models.termination_fns import get_termination_fn


class EGGROLLEnsemble(EnsembleDynamics):
    """Ensemble of dynamics models fitted via EGGROLL.

    A single DynamicsNet is trained with EGGROLL evolution strategy. At inference,
    ``num_members`` perturbed population members (positive-sigma, even thread_ids)
    serve as the ensemble, providing diverse predictions without training N independent
    models.
    """

    def __init__(self, obs_dim: int, act_dim: int, dataset_id: str, cfg: DictConfig):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        pop = int(cfg.eggroll.population_size)
        num_members = int(cfg.num_members)
        assert num_members <= pop // 2, (
            "num_members must be <= population_size // 2 so predict_ensemble() only uses "
            f"positive-sigma perturbations seen during training ({num_members} > {pop // 2})"
        )
        self._cfg = cfg
        self._termination_fn = get_termination_fn(dataset_id)
        self._state: EGGROLLState | None = None  # populated by train()
        self._last_train_epoch: int = 0  # anchor epoch for predict_ensemble()

    @classmethod
    def load_from_checkpoint(cls, path: str | Path) -> "EGGROLLEnsemble":
        """Reconstruct a trained EGGROLLEnsemble from a checkpoint file."""
        import pickle

        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        wm_cfg = OmegaConf.create(ckpt["world_model_cfg"])
        instance = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["dataset_id"], wm_cfg)  # type: ignore[arg-type]
        instance._state = ckpt["eggroll_state"]
        instance._last_train_epoch = ckpt["last_train_epoch"]
        return instance

    def checkpoint_state(self) -> EGGROLLState:
        """Return an inference-safe EGGROLLState for checkpointing.

        The training-time state contains the optax solver object in
        ``frozen_noiser_params["solver"]``, which is not pickleable on the
        cluster. For inference we only need the perturbation metadata plus the
        current ``sigma``, so strip the non-serializable training-only fields.
        Resume-training from checkpoints is not supported yet.
        """
        assert self._state is not None, "Must call train() before checkpoint_state()"
        frozen_noiser_params = dict(self._state.frozen_noiser_params)
        frozen_noiser_params.pop("solver", None)
        noiser_params = {"sigma": self._state.noiser_params["sigma"]}
        return self._state._replace(
            frozen_noiser_params=frozen_noiser_params,
            noiser_params=noiser_params,
        )

    @property
    def termination_fn(self) -> Callable:
        return self._termination_fn

    def predict_ensemble(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (ensemble_mean, ensemble_std) for ``num_members`` perturbed members.

        Uses even thread_ids (0, 2, 4, …) at ``_last_train_epoch`` so all members
        are positive-sigma perturbations — genuine diversity rather than ±-mirroring.

        Returns:
            ensemble_mean: ``(num_members, obs_dim + 1)`` — delta_obs + reward
            ensemble_std:  ``(num_members, obs_dim + 1)``
        """
        assert self._state is not None, "Must call train() before predict_ensemble()"
        state = self._state
        num_members = self._cfg.num_members

        # Even thread_ids → positive-sigma perturbations; odd are their mirrors.
        # Anchor to _last_train_epoch for a consistent, reproducible ensemble.
        iterinfos = (
            jnp.full(num_members, self._last_train_epoch, dtype=jnp.int32),
            jnp.arange(num_members, dtype=jnp.int32) * 2,
        )
        means, logvars = jax.vmap(
            lambda it: DynamicsNet.forward(
                EggRoll,
                state.frozen_noiser_params,
                state.noiser_params,
                state.frozen_params,
                state.params,
                state.es_tree_key,
                it,
                obs,
                action,
            ),
        )(iterinfos)

        ensemble_std = jnp.exp(0.5 * logvars)
        return means, ensemble_std

    def step(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample (next_obs, reward, done) from the unperturbed base model.

        Uses ``iterinfo=None`` for the unperturbed base parameters — the trained best
        estimate. Aleatoric noise comes from sampling the model's own predicted Gaussian
        ``N(mean, exp(0.5 * logvar))``.
        """
        assert self._state is not None, "Must call train() before step()"
        state = self._state

        mean, logvar = DynamicsNet.forward(
            EggRoll,
            state.frozen_noiser_params,
            state.noiser_params,
            state.frozen_params,
            state.params,
            state.es_tree_key,
            None,  # iterinfo=None → unperturbed base params
            obs,
            action,
        )
        std = jnp.exp(0.5 * logvar)
        noise = jax.random.normal(rng, mean.shape)
        sample = mean + noise * std

        delta_obs, reward = sample[:-1], sample[-1]
        next_obs = obs + delta_obs
        done = self.termination_fn(obs, action, next_obs)
        return next_obs, reward, done

    def train(
        self,
        dataset: Transition,
        cfg: DictConfig,
        rng: jax.Array,
        log_fn: Callable[..., None] | None = None,
    ) -> None:
        """Fit the ensemble to the offline dataset via EGGROLL.

        The full training loop runs inside ``jax.lax.fori_loop`` so it is
        JIT-compiled. ``frozen_noiser_params`` (which contains the optax solver,
        a Python object) is captured as a closure outside the loop — only the
        mutable arrays ``(rng, noiser_params, params)`` form the traced carry.

        Logs train NLL every ``log_interval`` epochs. Full-validation MSE is
        computed and logged every ``full_validation_interval`` epochs via
        ``jax.lax.cond`` + ``jax.debug.callback`` when ``log_fn`` is provided.
        """
        assert int(cfg.eggroll.group_size) % 2 == 0, (
            f"group_size must be even (got {cfg.eggroll.group_size})"
        )
        assert int(cfg.eggroll.population_size) % int(cfg.eggroll.group_size) == 0, (
            f"group_size must divide population_size "
            f"({cfg.eggroll.population_size} % {cfg.eggroll.group_size} != 0)"
        )

        assert int(cfg.log_interval) > 0, f"log_interval must be positive (got {cfg.log_interval})"
        full_validation_interval = int(cfg.get("full_validation_interval", cfg.log_interval))
        assert full_validation_interval > 0, (
            "full_validation_interval must be positive "
            f"(got {full_validation_interval})"
        )

        # Allocate all keys up front
        rng, split_rng, init_rng, es_rng = jax.random.split(rng, 4)

        # Split into train / val partitions
        train_data, val_data = train_val_split(dataset, cfg.validation_split, split_rng)

        # Targets: (delta_obs, reward) — matches DynamicsNet output convention
        def _targets(data: Transition) -> jnp.ndarray:
            delta_obs = data.next_obs - data.obs
            return jnp.concatenate([delta_obs, data.reward[:, None]], axis=-1)

        train_targets = _targets(train_data)
        val_targets = _targets(val_data)

        # Initialise DynamicsNet + EGGROLL state
        common_init = DynamicsNet.rand_init(
            init_rng,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dims=list(cfg.hidden_dims),
            activation=cfg.activation,
        )
        state = init_eggroll_state(
            common_init,
            es_rng,
            sigma=float(cfg.eggroll.sigma),
            lr=float(cfg.eggroll.lr),
            group_size=int(cfg.eggroll.group_size),
            noise_reuse=int(cfg.eggroll.noise_reuse),
        )

        # Static closures — captured once outside fori_loop.
        # frozen_noiser_params holds the optax solver (a Python callable) and cannot
        # be a traced JAX value; see EGGROLLState docstring in training.py.
        fnp = state.frozen_noiser_params
        fp = state.frozen_params
        etk = state.es_tree_key
        em = state.es_map
        pop = int(cfg.eggroll.population_size)
        group_size = int(cfg.eggroll.group_size)
        n_prompts = pop // group_size
        sigma_decay = float(cfg.eggroll.sigma_decay_rate)
        log_interval = int(cfg.log_interval)
        n_train = train_data.obs.shape[0]

        def train_epoch(epoch: int, carry: tuple) -> tuple:
            rng, noiser_params, params = carry
            rng, batch_rng = jax.random.split(rng)

            # Sample n_prompts distinct transitions, repeat each group_size times.
            # Consecutive thread-id groups see the same transition, matching
            # EggRoll.convert_fitnesses grouping.
            prompt_idxs = jax.random.randint(batch_rng, (n_prompts,), 0, n_train)
            idxs = jnp.repeat(prompt_idxs, group_size)
            obs_b = train_data.obs[idxs]
            action_b = train_data.action[idxs]
            target_b = train_targets[idxs]

            iterinfos = get_iterinfos(epoch, pop)
            means, logvars = jax.vmap(
                lambda it, o, a: DynamicsNet.forward(
                    EggRoll, fnp, noiser_params, fp, params, etk, it, o, a
                ),
                in_axes=(0, 0, 0),
            )(iterinfos, obs_b, action_b)

            # Per-perturbation Gaussian NLL (negated: higher fitness = lower NLL)
            fitnesses = -0.5 * jnp.sum(
                logvars + (target_b - means) ** 2 / jnp.exp(logvars),
                axis=-1,
            )

            # EGGROLL parameter update.
            # dict(noiser_params) shallow-copies before do_updates mutates it in-place.
            normalized = EggRoll.convert_fitnesses(fnp, noiser_params, fitnesses)
            noiser_params, params = EggRoll.do_updates(
                fnp, dict(noiser_params), params, etk, normalized, iterinfos, em
            )

            # Functional sigma decay — no in-place mutation on the traced carry dict
            noiser_params = {**noiser_params, "sigma": noiser_params["sigma"] * sigma_decay}

            # Log train loss every log_interval and full-validation MSE every
            # full_validation_interval. `if log_fn is not None` runs at trace
            # time (zero overhead when disabled); lax.cond handles runtime conditions.
            if log_fn is not None:
                _log_fn = log_fn  # capture narrowed (non-None) type for Pyright
                train_nll = -jnp.mean(fitnesses)
                should_full_validate = (epoch + 1) % full_validation_interval == 0

                def _log_train_only(_: None) -> None:
                    jax.debug.callback(_log_fn, epoch, train_nll, jnp.nan)

                def _log_with_full_validation(_: None) -> None:
                    val_means, _ = jax.vmap(
                        lambda o, a: DynamicsNet.forward(
                            EggRoll, fnp, noiser_params, fp, params, etk, None, o, a
                        ),
                        in_axes=(0, 0),
                    )(val_data.obs, val_data.action)
                    val_mse = jnp.mean((val_means - val_targets) ** 2)
                    jax.debug.callback(_log_fn, epoch, train_nll, val_mse)

                def _do_log(_: None) -> None:
                    jax.lax.cond(
                        should_full_validate,
                        _log_with_full_validation,
                        _log_train_only,
                        None,
                    )

                jax.lax.cond(
                    ((epoch + 1) % log_interval == 0) | should_full_validate,
                    _do_log,
                    lambda _: None,
                    None,
                )

            return rng, noiser_params, params

        _, noiser_params, params = jax.lax.fori_loop(
            0, int(cfg.num_epochs), train_epoch, (rng, state.noiser_params, state.params)
        )
        self._state = state._replace(noiser_params=noiser_params, params=params)
        self._last_train_epoch = int(cfg.num_epochs) - 1
