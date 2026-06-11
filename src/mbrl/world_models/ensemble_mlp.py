"""Ensemble of `DynamicsNet` world models, trainable by backprop or EGGROLL.

A genuine ensemble of ``num_ensemble`` independently-initialised `DynamicsNet`
members (stacked on a leading axis). ``cfg.trainer`` selects the optimiser:

- ``backprop``: one optax optimiser over the stacked members; NLL summed over the
  member axis. Diversity comes from independent init (the deep-ensembles recipe).
- ``eggroll``: each member runs its own EGGROLL search, vmapped over members. The
  prompt batch is shared across members; ``use_shared_perturbations`` toggles whether
  members get independent perturbation directions (default, an init-orthogonal
  diversity source backprop lacks) or a single broadcast set (init-only diversity).

Both trainers produce the same stacked-member checkpoint, so a checkpoint from either
can be fine-tuned by either via ``init_checkpoint``. The best ``num_elites`` members by
validation MSE are recorded in ``elite_idxs``; ``predict_ensemble``/``step`` use elites.
Inference is identical for both trainers — unperturbed member forwards (no population).
"""

from collections.abc import Callable
from pathlib import Path
import pickle
from typing import cast

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
import optax

from mbrl.data import (
    EpisodeBatch,
    TrajectoryWindows,
    Transition,
    create_epoch_iterator,
    derive_episode_train_val_split,
    derive_train_val_split,
    tile_episodes_to_windows,
)
from mbrl.eggroll.networks import DynamicsNet
from mbrl.eggroll.primitives import (
    EXCLUDED,
    CommonParams,
    EggRoll,
    Noiser,
    simple_es_tree_key,
)
from mbrl.eggroll.training import build_schedule, get_iterinfos, resolve_optax_solver
from mbrl.world_models.base import EnsembleDynamics
from mbrl.world_models.termination_fns import get_termination_fn


def _eggroll_work_counters(
    step: int,
    n_prompts: int,
    population_size: int,
    n_val: int,
    full_validation_interval: int,
) -> tuple[int, int]:
    """Cumulative (transitions_seen, forward_evals) as Python ints.

    ``transitions_seen`` counts distinct training transitions consumed and is NOT
    scaled by the ensemble size — the prompt batch is shared across members. Callers
    pass ``population_size`` and ``n_val`` already scaled by num_ensemble so that
    ``forward_evals`` (a compute proxy) reflects the per-member population + val passes.
    """
    transitions_seen = n_prompts * step
    num_full_validations = step // full_validation_interval
    if full_validation_interval != 1:
        num_full_validations += 1
    forward_evals = step * population_size + (1 + num_full_validations) * n_val
    return transitions_seen, forward_evals


def _build_optimizer(cfg: DictConfig) -> optax.GradientTransformation:
    """Backprop optimiser from ``cfg.optimizer`` (+ ``cfg.optimizer_kwargs``)."""
    solver = resolve_optax_solver(str(cfg.get("optimizer", "adamw")))
    return solver(cfg.lr, **dict(cfg.get("optimizer_kwargs") or {}))


def _targets(data: Transition) -> jnp.ndarray:
    """(delta_obs, reward) target matching DynamicsNet output convention."""
    delta_obs = data.next_obs - data.obs
    return jnp.concatenate([delta_obs, data.reward[:, None]], axis=-1)


def _dynamics_init_kwargs(cfg: DictConfig, obs_dim: int, act_dim: int) -> dict:
    return dict(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=list(cfg.hidden_dims),
        activation=cfg.activation,
        init_scheme=str(cfg.get("init_scheme", "eggroll")),
        backbone=str(cfg.get("backbone", "mlp")),
        max_logvar_init=float(cfg.get("max_logvar_init", 0.5)),
        min_logvar_init=float(cfg.get("min_logvar_init", -10.0)),
        predict_logvar=not bool(cfg.get("disable_logvar_predictions", False)),
    )


def _assert_params_loadable(new_params, ckpt_params) -> None:
    """Guard ``init_checkpoint`` params against an incompatible model.

    ``jax.tree.structure`` alone misses ``num_ensemble`` and layer-width mismatches
    (the leading axis is ``num_ensemble``; hidden widths are leaf shapes — both are
    invisible to the structural comparison). Compare leaf shapes too.
    """
    if jax.tree.structure(new_params) != jax.tree.structure(ckpt_params):
        raise ValueError(
            "init_checkpoint params structure mismatch — check hidden_dims / "
            "activation / init_scheme / backbone match the checkpoint."
        )
    new_shapes = [jnp.shape(x) for x in jax.tree.leaves(new_params)]
    ckpt_shapes = [jnp.shape(x) for x in jax.tree.leaves(ckpt_params)]
    if new_shapes != ckpt_shapes:
        raise ValueError(
            "init_checkpoint params shape mismatch — num_ensemble or layer widths "
            "differ from the checkpoint. Match num_ensemble / hidden_dims to the "
            "checkpoint, or start from scratch (drop init_checkpoint)."
        )


class EnsembleMLP(EnsembleDynamics):
    """Ensemble of `DynamicsNet` members trained by backprop or EGGROLL."""

    def __init__(self, obs_dim: int, act_dim: int, dataset_id: str, cfg: DictConfig):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._cfg = cfg
        self._termination_fn = get_termination_fn(dataset_id)
        self.num_ensemble = int(cfg.num_ensemble)
        self.num_elites = int(cfg.num_elites)
        assert 1 <= self.num_elites <= self.num_ensemble, (
            f"num_elites must be in [1, num_ensemble]; got num_elites={self.num_elites}, "
            f"num_ensemble={self.num_ensemble}"
        )
        self.trainer = str(cfg.trainer)
        assert self.trainer in ("backprop", "eggroll", "eggroll_trajectory"), (
            f"cfg.trainer must be 'backprop', 'eggroll' or 'eggroll_trajectory'; "
            f"got {self.trainer!r}"
        )

        # Deterministic-dynamics mode: the head predicts only mean + reward, training
        # reduces to MSE, and all uncertainty becomes epistemic (ensemble disagreement).
        # Backs the inherited EnsembleDynamics.predicts_logvar property.
        self._predicts_logvar = not bool(cfg.get("disable_logvar_predictions", False))
        if not self.predicts_logvar and (
            bool(cfg.get("use_mse_fitness", False))
            or bool(cfg.get("freeze_logvar_clamp", False))
        ):
            raise ValueError(
                "disable_logvar_predictions is incompatible with use_mse_fitness / "
                "freeze_logvar_clamp: there is no log-variance head to weight or freeze."
            )

        # Populated by train() / load_from_checkpoint().
        self._params = None  # stacked (num_ensemble, ...) DynamicsNet params
        self._frozen_params: dict | None = None
        self._scan_map = None
        self._es_tree_key = None  # single dummy key tree; values unused at iterinfo=None
        self._elite_idxs = None
        self._opt_state = None  # trainer-specific; for fine-tune resume
        self._update_steps_completed: int = 0
        self._validation_split: float | None = None
        # Episode-level held-out fraction used by the trajectory trainer (issue #42);
        # separate from the transition-level _validation_split so both stay reproducible.
        self._trajectory_validation_split: float | None = None
        self._seed: int | None = None
        # MoReL halt-penalty stats; populated by precompute_term_stats() / load.
        self._discrepancy: float | None = None
        self._min_r: float | None = None

    # ── network init / inference ─────────────────────────────────────────────

    def _init_members(self, init_rng: jax.Array):
        """Return (stacked_params, single_params, frozen_params, scan_map, es_map).

        Each member is initialised from a distinct key (independent init = diversity).
        Structural fields (frozen_params/scan_map/es_map) are identical across members.
        """
        kw = _dynamics_init_kwargs(self._cfg, self.obs_dim, self.act_dim)
        structural = DynamicsNet.rand_init(jax.random.fold_in(init_rng, 0), **kw)
        member_keys = jax.random.split(init_rng, self.num_ensemble)
        stacked_params = jax.vmap(lambda k: DynamicsNet.rand_init(k, **kw).params)(member_keys)
        return (
            stacked_params,
            structural.params,
            structural.frozen_params,
            structural.scan_map,
            structural.es_map,
        )

    def _set_inference_structure(self, single_params, frozen_params, scan_map) -> None:
        self._frozen_params = frozen_params
        self._scan_map = scan_map
        # Structurally required by call_submodule; values unused at iterinfo=None.
        self._es_tree_key = simple_es_tree_key(single_params, jax.random.key(0), scan_map)

    def _member_meanlogvar(
        self, params_m, obs: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jax.Array, jax.Array | None]:
        """Unperturbed forward of a single member (base Noiser, iterinfo=None).

        ``logvar`` is ``None`` when the model runs without a variance head
        (``disable_logvar_predictions``).
        """
        common = CommonParams(
            Noiser, {}, {}, self._frozen_params, params_m, self._es_tree_key, None
        )
        mean, logvar, _, _ = DynamicsNet._forward_with_bounds(common, obs, action)
        return mean, logvar

    def _elite_params(self):
        assert self._params is not None and self._elite_idxs is not None
        return jax.tree.map(lambda x: x[self._elite_idxs], self._params)

    @property
    def termination_fn(self) -> Callable:
        return self._termination_fn

    def predict_ensemble(
        self, obs: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """(ensemble_mean, ensemble_std) over elite members; each (num_elites, obs+1).

        ``ensemble_std`` is ``None`` for a deterministic model (no variance head); in
        that case uncertainty is epistemic only — the spread *across* member means.
        """
        assert self._params is not None, "Must train() / load before predict_ensemble()"
        elite_params = self._elite_params()
        if not self.predicts_logvar:
            means = jax.vmap(
                lambda pm: self._member_meanlogvar(pm, obs, action)[0]
            )(elite_params)
            return means, None
        means, logvars = jax.vmap(
            lambda pm: self._member_meanlogvar(pm, obs, action)
        )(elite_params)
        assert logvars is not None  # predicts_logvar True ⇒ variance head present
        return means, jnp.exp(0.5 * logvars)

    def step(
        self, obs: jnp.ndarray, action: jnp.ndarray, rng: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample (next_obs, reward, done) from a randomly-selected elite member.

        For a deterministic model the per-step member resampling is the only source of
        stochasticity (PETS-style trajectory sampling); no aleatoric noise is added.
        """
        assert self._elite_idxs is not None, "Must train() / load before step()"
        rng_elite, rng_noise = jax.random.split(rng)
        ensemble_mean, ensemble_std = self.predict_ensemble(obs, action)
        sample_idx = jax.random.randint(rng_elite, (), 0, self.num_elites)
        mean = ensemble_mean[sample_idx]
        if ensemble_std is None:
            sample = mean
        else:
            std = ensemble_std[sample_idx]
            sample = mean + jax.random.normal(rng_noise, shape=mean.shape) * std
        delta_obs, reward = sample[:-1], sample[-1]
        next_obs = obs + delta_obs
        done = self.termination_fn(obs, action, next_obs)
        return next_obs, reward, done

    def _per_member_val_mse(self, params, val_obs, val_action, val_targets) -> jax.Array:
        """(num_ensemble,) unperturbed mean MSE per member over the val set."""
        means, _ = jax.vmap(  # over members
            lambda pm: jax.vmap(  # over val transitions
                lambda o, a: self._member_meanlogvar(pm, o, a), in_axes=(0, 0)
            )(val_obs, val_action)
        )(params)
        return ((means - val_targets[None]) ** 2).mean(axis=(1, 2))

    def compute_val_mse(self, dataset: Transition) -> jax.Array:
        """Elite-mean MSE over *dataset* (matches the training-time ``val_mse_elite``)."""
        assert self._params is not None and self._elite_idxs is not None
        targets = _targets(dataset)
        per_member = self._per_member_val_mse(
            self._params, dataset.obs, dataset.action, targets
        )
        return per_member[self._elite_idxs].mean()

    # ── checkpointing ────────────────────────────────────────────────────────

    def checkpoint_state(self) -> dict:
        assert self._params is not None, "Must call train() before checkpoint_state()"
        return {
            "params": self._params,
            "elite_idxs": self._elite_idxs,
            "num_elites": self.num_elites,
            "trainer": self.trainer,
            "opt_state": self._opt_state,
            "update_steps_completed": self._update_steps_completed,
            "validation_split": self._validation_split,
            "trajectory_validation_split": self._trajectory_validation_split,
            "seed": self._seed,
            "discrepancy": self._discrepancy,
            "min_r": self._min_r,
        }

    @classmethod
    def load_from_checkpoint(cls, path: str | Path) -> "EnsembleMLP":
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        wm_cfg = OmegaConf.create(ckpt["world_model_cfg"])
        assert isinstance(wm_cfg, DictConfig)
        instance = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["dataset_id"], wm_cfg)
        kw = _dynamics_init_kwargs(wm_cfg, ckpt["obs_dim"], ckpt["act_dim"])
        structural = DynamicsNet.rand_init(jax.random.key(0), **kw)
        instance._set_inference_structure(
            structural.params, structural.frozen_params, structural.scan_map
        )
        instance._params = ckpt["params"]
        instance._elite_idxs = ckpt["elite_idxs"]
        instance.num_elites = int(ckpt["num_elites"])
        instance._opt_state = ckpt.get("opt_state")
        instance._update_steps_completed = int(ckpt.get("update_steps_completed", 0))
        instance._validation_split = ckpt.get("validation_split")
        instance._trajectory_validation_split = ckpt.get("trajectory_validation_split")
        instance._seed = ckpt.get("seed")
        # .get for back-compat with checkpoints saved before term stats existed.
        instance._discrepancy = ckpt.get("discrepancy")
        instance._min_r = ckpt.get("min_r")
        return instance

    # ── training ─────────────────────────────────────────────────────────────

    def train(
        self,
        dataset: Transition,
        cfg: DictConfig,
        rng: jax.Array,
        log_fn: Callable[..., None] | None = None,
        episodes: EpisodeBatch | None = None,
    ) -> None:
        seed = cfg.get("seed", None)
        assert seed is not None, (
            "EnsembleMLP requires cfg.seed (set by experiments/world_model.py from the "
            "top-level cfg.seed) so the train/val split is reproducible for fine-tuning."
        )
        if self.trainer == "backprop":
            self._train_backprop(dataset, cfg, rng, int(seed), log_fn)
        elif self.trainer == "eggroll":
            self._train_eggroll(dataset, cfg, rng, int(seed), log_fn)
        else:
            assert episodes is not None, (
                "trainer='eggroll_trajectory' requires episode-structured data; "
                "experiments/world_model.py must pass episodes=load_episodes(...)."
            )
            self._train_eggroll_trajectory(
                episodes, dataset, cfg, rng, int(seed), log_fn
            )

    def _finalise(
        self, params, opt_state, val_data, steps_completed, val_split, split_seed
    ) -> None:
        self._params = params
        self._opt_state = opt_state
        self._update_steps_completed = int(steps_completed)
        self._validation_split = float(val_split)
        # Persist the seed that actually determined the train/val split (the source
        # checkpoint's seed when fine-tuning) so a later fine-tune reproduces the
        # same split — not cfg.seed, which only drives the training RNG.
        self._seed = int(split_seed)
        per_member = self._per_member_val_mse(
            params, val_data.obs, val_data.action, _targets(val_data)
        )
        self._elite_idxs = jnp.argsort(per_member)[: self.num_elites]

    # ---- backprop trainer ----

    def _train_backprop(self, dataset, cfg, rng, seed, log_fn) -> None:
        ckpt = cfg.get("init_checkpoint", None)
        reset_optax = bool(cfg.get("reset_optax_state", False))
        if ckpt is not None:
            with open(ckpt, "rb") as f:
                ckpt = pickle.load(f)
            val_split = float(ckpt["validation_split"])
            split_seed = int(ckpt["seed"])
        else:
            val_split = float(cfg.validation_split)
            split_seed = seed

        train_data, val_data = derive_train_val_split(dataset, val_split, split_seed)
        train_obs, train_action = train_data.obs, train_data.action
        train_targets = _targets(train_data)
        val_obs, val_action, val_targets = val_data.obs, val_data.action, _targets(val_data)

        rng, init_rng = jax.random.split(rng)
        stacked_params, single_params, frozen_params, scan_map, _ = self._init_members(init_rng)
        self._set_inference_structure(single_params, frozen_params, scan_map)

        tx = _build_optimizer(cfg)
        if ckpt is not None:
            _assert_params_loadable(stacked_params, ckpt["params"])
            stacked_params = ckpt["params"]
        opt_state = tx.init(stacked_params)
        if ckpt is not None and not reset_optax and ckpt.get("opt_state") is not None:
            assert jax.tree.structure(opt_state) == jax.tree.structure(ckpt["opt_state"]), (
                "init_checkpoint opt_state structure mismatch (different solver?). "
                "Set reset_optax_state=true to start a fresh optimiser."
            )
            opt_state = ckpt["opt_state"]

        batch_size = int(cfg.batch_size)
        coef = float(cfg.get("logvar_diff_coef", 0.01))
        n_ens = self.num_ensemble
        batches_per_epoch = max(train_obs.shape[0] // batch_size, 1)
        log_interval = int(cfg.get("log_interval", batches_per_epoch))
        full_validation_interval = int(cfg.get("full_validation_interval", batches_per_epoch))
        n_val = val_obs.shape[0]
        num_elites = self.num_elites
        lr_value = float(cfg.lr)  # constant backprop step size, logged for trainer parity

        def _loss_fn(params, obs_b, action_b, target_b):
            def per_member(pm):
                return jax.vmap(
                    lambda o, a: DynamicsNet._forward_with_bounds(
                        CommonParams(Noiser, {}, {}, frozen_params, pm, self._es_tree_key, None),
                        o,
                        a,
                    ),
                    in_axes=(0, 0),
                )(obs_b, action_b)

            means, logvars, maxlv, minlv = jax.vmap(per_member)(params)  # (N,B,D),(N,D)
            if not self.predicts_logvar:  # deterministic: plain MSE, no variance/clamp terms
                return ((means - target_b[None]) ** 2).sum(0).mean()
            assert logvars is not None and maxlv is not None and minlv is not None
            mse_loss = (((means - target_b[None]) ** 2) * jnp.exp(-logvars)).sum(0).mean()
            var_loss = logvars.sum(0).mean()
            logvar_diff = (maxlv - minlv).sum()
            return mse_loss + var_loss + coef * logvar_diff

        def _train_step(carry, batch):
            params, opt_state, step = carry
            obs_b, action_b, target_b = batch
            loss, grads = jax.value_and_grad(_loss_fn)(params, obs_b, action_b, target_b)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            step = step + 1
            if log_fn is not None:
                _emit_backprop_log(
                    log_fn, params, step, loss, val_obs, val_action, val_targets,
                    self._per_member_val_mse, n_ens, num_elites, batch_size, n_val,
                    log_interval, full_validation_interval, lr_value,
                )
            return (params, opt_state, step), loss

        if log_fn is not None and ckpt is None:
            init_per_member = self._per_member_val_mse(
                stacked_params, val_obs, val_action, val_targets
            )
            log_fn(
                0, float("nan"), float(init_per_member.mean()), 0, n_val * n_ens,
                epoch=0, lr=lr_value,
                val_mse_elite=float(jnp.sort(init_per_member)[:num_elites].mean()),
            )

        rng, shuffle_rng = jax.random.split(rng)

        def train_epoch(epoch, carry):
            params, opt_state, step = carry
            epoch_rng = jax.random.fold_in(shuffle_rng, epoch)
            it = create_epoch_iterator(
                (train_obs, train_action, train_targets), batch_size, epoch_rng
            )
            (params, opt_state, step), _ = jax.lax.scan(_train_step, (params, opt_state, step), it)
            return params, opt_state, step

        params, opt_state, _ = jax.lax.fori_loop(
            0, int(cfg.num_epochs), train_epoch, (stacked_params, opt_state, jnp.int32(0))
        )
        self._finalise(
            params, opt_state, val_data, int(cfg.num_epochs) * batches_per_epoch,
            val_split, split_seed,
        )

    # ---- eggroll trainer ----

    def _train_eggroll(self, dataset, cfg, rng, seed, log_fn) -> None:
        group_size = int(cfg.eggroll.group_size)
        assert group_size >= 0
        if group_size > 0:
            assert group_size % 2 == 0 and int(cfg.eggroll.population_size) % group_size == 0

        full_validation_interval = int(cfg.get("full_validation_interval", cfg.log_interval))
        log_interval = int(cfg.log_interval)
        assert log_interval > 0 and full_validation_interval > 0

        ckpt = cfg.get("init_checkpoint", None)
        reset_optax = bool(cfg.get("reset_optax_state", False))
        lr_schedule_name = str(cfg.eggroll.get("lr_schedule", "constant"))
        if ckpt is not None and not reset_optax and lr_schedule_name != "constant":
            raise ValueError(
                "Warm-starting with reset_optax_state=false and a non-constant lr_schedule "
                f"({lr_schedule_name!r}) is unsupported: the loaded optimiser state has no "
                "schedule step counter. Set reset_optax_state=true."
            )
        step_offset = 0
        if ckpt is not None:
            with open(ckpt, "rb") as f:
                ckpt = pickle.load(f)
            step_offset = int(ckpt.get("update_steps_completed", 0))
            val_split = float(ckpt["validation_split"])
            split_seed = int(ckpt["seed"])
        else:
            val_split = float(cfg.validation_split)
            split_seed = seed

        train_data, val_data = derive_train_val_split(dataset, val_split, split_seed)
        train_targets = _targets(train_data)
        val_obs, val_action, val_targets = val_data.obs, val_data.action, _targets(val_data)
        n_train = train_data.obs.shape[0]
        n_val = val_data.obs.shape[0]
        n_ens = self.num_ensemble

        # Schedules (lr drives the optax solver internally; sigma evaluated per epoch).
        lr_schedule = _resolve_schedule(cfg.eggroll, "lr", "lr_schedule", "lr_schedule_kwargs")
        sigma_schedule = _resolve_sigma_schedule(cfg.eggroll)

        rng, init_rng, es_rng = jax.random.split(rng, 3)
        stacked_params, single_params, frozen_params, scan_map, es_map = self._init_members(
            init_rng
        )
        self._set_inference_structure(single_params, frozen_params, scan_map)

        solver_kwargs = _container(cfg.eggroll.get("solver_kwargs", {}))
        fnp, _ = EggRoll.init_noiser(
            single_params,
            float(cfg.eggroll.sigma),
            lr_schedule,
            solver=resolve_optax_solver(str(cfg.eggroll.get("solver", "sgd"))),
            solver_kwargs=solver_kwargs,
            group_size=group_size,
            noise_reuse=int(cfg.eggroll.noise_reuse),
            use_batched_update=bool(cfg.eggroll.get("use_batched_update", False)),
        )
        solver = fnp["solver"]
        opt_state = jax.vmap(solver.init)(stacked_params)

        use_shared = bool(cfg.get("use_shared_perturbations", False))
        if use_shared:
            train_es_key = simple_es_tree_key(single_params, es_rng, scan_map)
            es_axis = None
        else:
            es_keys = jax.random.split(es_rng, n_ens)
            train_es_key = jax.vmap(
                lambda k: simple_es_tree_key(single_params, k, scan_map)
            )(es_keys)
            es_axis = 0

        if ckpt is not None:
            _assert_params_loadable(stacked_params, ckpt["params"])
            stacked_params = ckpt["params"]
            if not reset_optax and ckpt.get("opt_state") is not None:
                assert jax.tree.structure(opt_state) == jax.tree.structure(ckpt["opt_state"]), (
                    "init_checkpoint opt_state structure mismatch (different solver?). "
                    "Set reset_optax_state=true to start a fresh optimiser."
                )
                opt_state = ckpt["opt_state"]

        pop = int(cfg.eggroll.population_size)
        n_prompts = pop if group_size == 0 else pop // group_size
        coef = float(cfg.get("logvar_diff_coef", 0.01))
        num_elites = self.num_elites

        def member_step(params_m, opt_state_m, es_key_m, sigma, iterinfos, obs_b, action_b, tgt_b):
            nps = {"sigma": sigma, "opt_state": opt_state_m}
            means, logvars, maxlv, minlv = jax.vmap(
                lambda it, o, a: DynamicsNet._forward_noisy_with_bounds(
                    EggRoll, fnp, nps, frozen_params, params_m, es_key_m, it, o, a
                ),
                in_axes=(0, 0, 0),
            )(iterinfos, obs_b, action_b)
            if not self.predicts_logvar:  # deterministic: plain MSE fitness
                losses = jnp.mean((tgt_b - means) ** 2, axis=-1)
            else:
                assert logvars is not None and maxlv is not None and minlv is not None
                losses = (
                    jnp.mean(((tgt_b - means) ** 2) * jnp.exp(-logvars), axis=-1)
                    + jnp.mean(logvars, axis=-1)
                    + coef * jnp.sum(maxlv - minlv, axis=-1)
                )
            normalized = EggRoll.convert_fitnesses(fnp, nps, -losses)
            new_nps, new_params = EggRoll.do_updates(
                fnp, dict(nps), params_m, es_key_m, normalized, iterinfos, es_map
            )
            return new_params, new_nps["opt_state"], jnp.mean(losses)

        vmapped_step = jax.vmap(
            member_step, in_axes=(0, 0, es_axis, None, None, None, None, None)
        )

        if log_fn is not None and ckpt is None:
            init_per_member = self._per_member_val_mse(
                stacked_params, val_obs, val_action, val_targets
            )
            log_fn(
                0, float("nan"), float(init_per_member.mean()), 0, n_val * n_ens,
                lr=float(jnp.asarray(lr_schedule(0)) if callable(lr_schedule) else lr_schedule),
                sigma=float(jnp.asarray(sigma_schedule(0))),
                val_mse_elite=float(jnp.sort(init_per_member)[:num_elites].mean()),
            )

        def train_epoch(epoch, carry):
            rng, params, opt_state = carry
            sigma = sigma_schedule(epoch)
            rng, batch_rng = jax.random.split(rng)
            prompt_idxs = jax.random.randint(batch_rng, (n_prompts,), 0, n_train)
            idxs = prompt_idxs if group_size == 0 else jnp.repeat(prompt_idxs, group_size)
            obs_b = train_data.obs[idxs]
            action_b = train_data.action[idxs]
            tgt_b = train_targets[idxs]
            iterinfos = get_iterinfos(epoch, pop)

            params, opt_state, member_losses = vmapped_step(
                params, opt_state, train_es_key, sigma, iterinfos, obs_b, action_b, tgt_b
            )

            if log_fn is not None:
                _emit_eggroll_log(
                    log_fn, params, epoch, jnp.mean(member_losses),
                    self._per_member_val_mse, val_obs, val_action, val_targets,
                    lr_schedule, sigma, n_prompts, pop, n_val, n_ens, num_elites,
                    log_interval, full_validation_interval, step_offset,
                )
            return rng, params, opt_state

        rng, loop_rng = jax.random.split(rng)
        _, params, opt_state = jax.lax.fori_loop(
            0, int(cfg.num_epochs), train_epoch, (loop_rng, stacked_params, opt_state)
        )
        self._finalise(
            params, opt_state, val_data, int(cfg.num_epochs), val_split, split_seed
        )

    # ---- eggroll trajectory (Phase-2) trainer ----

    def _per_member_traj_mse(self, params, windows: TrajectoryWindows):
        """Open-loop multi-step rollout MSE per member, unperturbed (base Noiser).

        Returns ``(scalar (num_ensemble,), curve (num_ensemble, T))``: per-step MSE is the
        mean over the D=(obs_dim+1) features of the squared error between the rolled-out
        *absolute* next-state+reward and the real ones, masked over padded/post-terminal
        steps. The scalar is the masked mean over all valid steps/windows; the curve is the
        masked mean over windows at each horizon step (the compounding-error curve).
        """

        def member(params_m):
            def one_window(start_obs, actions_w, target_obs_w, target_reward_w, mask_w):
                def step(carry_obs, inp):
                    a_t, tgt_o, tgt_r, m = inp
                    mean, _ = self._member_meanlogvar(params_m, carry_obs, a_t)
                    next_obs = carry_obs + mean[:-1]
                    pred = jnp.concatenate([next_obs, mean[-1:]])
                    tgt = jnp.concatenate([tgt_o, tgt_r[None]])
                    se = jnp.mean((pred - tgt) ** 2)
                    return next_obs, jnp.where(m > 0, se, 0.0)

                _, ses = jax.lax.scan(
                    step, start_obs, (actions_w, target_obs_w, target_reward_w, mask_w)
                )
                return ses  # (T,)

            ses = jax.vmap(one_window)(
                windows.start_obs, windows.actions, windows.target_obs,
                windows.target_reward, windows.mask,
            )  # (W, T)
            curve = ses.sum(axis=0) / jnp.maximum(windows.mask.sum(axis=0), 1.0)
            scalar = ses.sum() / jnp.maximum(windows.mask.sum(), 1.0)
            return scalar, curve

        return jax.vmap(member)(params)  # (E,), (E, T)

    def _ensemble_disagreement(self, params, obs, action) -> jax.Array:
        """Mean across features of the cross-member std of predicted means (diversity proxy).

        Logged at the start and end of Phase 2 (when ``log_ensemble_disagreement``) to check
        whether the fine-tune erodes the ensemble disagreement MOPO/MoReL rely on.
        """
        means = jax.vmap(  # over members
            lambda pm: jax.vmap(  # over transitions
                lambda o, a: self._member_meanlogvar(pm, o, a)[0], in_axes=(0, 0)
            )(obs, action)
        )(params)  # (E, N, D)
        return jnp.mean(jnp.std(means, axis=0))

    def _train_eggroll_trajectory(self, episodes, dataset, cfg, rng, seed, log_fn) -> None:
        pop = int(cfg.eggroll.population_size)
        assert pop % 2 == 0, "population_size must be even (EGGROLL antithetic ±sigma pairs)."
        batch_size = int(cfg.batch_size)
        log_interval = int(cfg.log_interval)
        full_validation_interval = int(cfg.get("full_validation_interval", log_interval))
        assert log_interval > 0 and full_validation_interval > 0
        coef = float(cfg.get("logvar_diff_coef", 0.01))
        # A deterministic model has no logvar head/clamp, so freezing is moot and the
        # fitness is necessarily pure MSE (the __init__ conflict check forbids the flags).
        freeze_clamp = bool(cfg.get("freeze_logvar_clamp", False)) and self.predicts_logvar
        # Diagnostic: drive the fitness by pure trajectory MSE (mean head only), dropping the
        # NLL precision weighting and the logvar terms. The variance head then receives no
        # fitness signal and drifts on noise, so this is for analysis, not production fine-tuning.
        # Forced on (no choice) when the model has no log-variance head.
        use_mse = bool(cfg.get("use_mse_fitness", False)) or not self.predicts_logvar
        log_transition = bool(cfg.get("val_transition_mse", True))
        log_disagreement = bool(cfg.get("log_ensemble_disagreement", False))
        n_ens = self.num_ensemble
        num_elites = self.num_elites

        # Warm-start is required: Phase 2 fine-tunes a Phase-1 (or earlier Phase-2) model.
        ckpt_path = cfg.get("init_checkpoint", None)
        assert ckpt_path is not None, (
            "trainer='eggroll_trajectory' requires init_checkpoint (a Phase-1 EnsembleMLP "
            "checkpoint to fine-tune)."
        )
        reset_optax = bool(cfg.get("reset_optax_state", True))
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        # Transition split params come from the checkpoint (reproduce Phase-1's exact val
        # set so the transition-level metric stays comparable across phases).
        val_split = float(ckpt["validation_split"])
        split_seed = int(ckpt.get("seed", seed))
        # Episode split fraction: from the checkpoint if it recorded one (Phase-2 →
        # Phase-2 / curriculum continuation), else from this run's cfg.
        traj_val_split = ckpt.get("trajectory_validation_split", None)
        traj_val_split = (
            float(cfg.trajectory_validation_split)
            if traj_val_split is None
            else float(traj_val_split)
        )
        step_offset = int(ckpt.get("update_steps_completed", 0))

        # Episode split (training windows + trajectory val) and the reproduced Phase-1
        # transition val set (for the comparable transition-level metric).
        train_eps, val_eps = derive_episode_train_val_split(
            episodes, traj_val_split, split_seed
        )
        _, transition_val = derive_train_val_split(dataset, val_split, split_seed)
        tv_obs, tv_action, tv_targets = (
            transition_val.obs, transition_val.action, _targets(transition_val)
        )

        lr_schedule = _resolve_schedule(cfg.eggroll, "lr", "lr_schedule", "lr_schedule_kwargs")
        sigma_schedule = _resolve_sigma_schedule(cfg.eggroll)

        rng, init_rng, es_rng = jax.random.split(rng, 3)
        stacked_params, single_params, frozen_params, scan_map, es_map = self._init_members(
            init_rng
        )
        self._set_inference_structure(single_params, frozen_params, scan_map)

        fnp, _ = EggRoll.init_noiser(
            single_params,
            float(cfg.eggroll.sigma),
            lr_schedule,
            solver=resolve_optax_solver(str(cfg.eggroll.get("solver", "sgd"))),
            solver_kwargs=_container(cfg.eggroll.get("solver_kwargs", {})),
            group_size=0,  # decoupled batch: all perturbations share the window batch
            noise_reuse=int(cfg.eggroll.get("noise_reuse", 1)),
            use_batched_update=bool(cfg.eggroll.get("use_batched_update", True)),
        )
        solver = fnp["solver"]
        opt_state = jax.vmap(solver.init)(stacked_params)

        # Independent per-member perturbation populations (es_axis=0), as in _train_eggroll.
        es_keys = jax.random.split(es_rng, n_ens)
        train_es_key = jax.vmap(
            lambda k: simple_es_tree_key(single_params, k, scan_map)
        )(es_keys)

        _assert_params_loadable(stacked_params, ckpt["params"])
        stacked_params = ckpt["params"]
        if not reset_optax and ckpt.get("opt_state") is not None:
            assert jax.tree.structure(opt_state) == jax.tree.structure(ckpt["opt_state"]), (
                "init_checkpoint opt_state structure mismatch (different solver?). "
                "Set reset_optax_state=true to start a fresh optimiser."
            )
            opt_state = ckpt["opt_state"]

        # Freeze the learnable logvar clamp bounds at their Phase-1 values. Two halves,
        # both required: EXCLUDED in es_map stops do_updates from persisting any change,
        # and freeze_clamp_bounds in the forward (passed below) evaluates them unperturbed
        # — so they are neither perturbed within a generation nor updated across them.
        if freeze_clamp:
            es_map = {**es_map, "max_logvar": EXCLUDED, "min_logvar": EXCLUDED}

        # ── per-generation fitness (decoupled batch shared across the whole population) ──
        def member_step(params_m, opt_state_m, es_key_m, sigma, iterinfos, windows):
            nps = {"sigma": sigma, "opt_state": opt_state_m}

            def per_perturbation(iterinfo_p):
                def per_window(start_obs, actions_w, target_obs_w, target_reward_w, mask_w):
                    def step(carry_obs, inp):
                        a_t, tgt_o, tgt_r, m = inp
                        # iterinfo held FIXED across the unroll → identical perturbation
                        # every step (primitives derive noise from (es_key, epoch, thread).)
                        mean, logvar, maxlv, minlv = DynamicsNet._forward_noisy_with_bounds(
                            EggRoll, fnp, nps, frozen_params, params_m, es_key_m,
                            iterinfo_p, carry_obs, a_t, freeze_clamp_bounds=freeze_clamp,
                        )
                        next_obs = carry_obs + mean[:-1]
                        # Absolute-state match: effective obs delta target is (real_next - ô).
                        tgt = jnp.concatenate([tgt_o - carry_obs, tgt_r[None]])
                        if use_mse:  # diagnostic: mean head only, no precision weighting
                            loss_t = jnp.mean((mean - tgt) ** 2)
                        else:
                            assert logvar is not None  # use_mse False ⇒ logvar head present
                            loss_t = jnp.mean((mean - tgt) ** 2 * jnp.exp(-logvar)) + jnp.mean(
                                logvar
                            )
                        # NaN-safe mask: a padded forward on garbage carry may be non-finite,
                        # and 0 * NaN = NaN would poison the accumulated fitness.
                        return next_obs, (jnp.where(m > 0, loss_t, 0.0), maxlv, minlv)

                    _, (nlls, maxlv, minlv) = jax.lax.scan(
                        step, start_obs,
                        (actions_w, target_obs_w, target_reward_w, mask_w),
                    )
                    loss = jnp.sum(nlls)
                    # No clamp regulariser under MSE fitness: the variance head is out of the loss.
                    if not freeze_clamp and not use_mse:  # clamp regulariser added once per rollout
                        assert maxlv is not None and minlv is not None
                        loss = loss + coef * jnp.sum(maxlv[0] - minlv[0])
                    return loss

                # vmap over the B windows (shared across the population), mean the losses.
                losses = jax.vmap(per_window)(
                    windows.start_obs, windows.actions, windows.target_obs,
                    windows.target_reward, windows.mask,
                )
                return jnp.mean(losses)

            per_pert = jax.vmap(per_perturbation)(iterinfos)  # (pop,)
            normalized = EggRoll.convert_fitnesses(fnp, nps, -per_pert)
            new_nps, new_params = EggRoll.do_updates(
                fnp, dict(nps), params_m, es_key_m, normalized, iterinfos, es_map
            )
            return new_params, new_nps["opt_state"], jnp.mean(per_pert)

        # vmap over members; windows + sigma + iterinfos shared (in_axes=None) so every
        # perturbation, including each antithetic ±sigma pair, sees identical inputs.
        vmapped_step = jax.vmap(member_step, in_axes=(0, 0, 0, None, None, None))

        curriculum = cfg.get("curriculum", None)
        if curriculum:
            stages = [(int(s["horizon"]), int(s["num_epochs"])) for s in curriculum]
        else:
            stages = [(int(cfg.horizon), int(cfg.num_epochs))]
        final_horizon = stages[-1][0]

        # Baseline log (gen 0): before-finetune trajectory/transition MSE + disagreement,
        # plus the train/val compounding-error curves of the warm-start (Phase-1) model.
        if log_fn is not None:
            init_windows = tile_episodes_to_windows(val_eps, stages[0][0])
            tj_pm, init_val_curve = self._per_member_traj_mse(stacked_params, init_windows)
            init_train_windows = _sample_train_curve_windows(
                train_eps, stages[0][0], int(init_windows.start_obs.shape[0]), split_seed
            )
            _, init_train_curve = self._per_member_traj_mse(stacked_params, init_train_windows)
            metrics = dict(
                val_traj_mse=float(tj_pm.mean()),
                val_traj_mse_elite=float(jnp.sort(tj_pm)[:num_elites].mean()),
                lr=float(jnp.asarray(lr_schedule(0)) if callable(lr_schedule) else lr_schedule),
                sigma=float(jnp.asarray(sigma_schedule(0))),
                transitions_seen=0, forward_evals=0,
                val_traj_mse_curve_init=_mean_curve_list(init_val_curve),
                train_traj_mse_curve_init=_mean_curve_list(init_train_curve),
            )
            if log_transition:
                tr_pm = self._per_member_val_mse(stacked_params, tv_obs, tv_action, tv_targets)
                metrics["val_transition_mse"] = float(tr_pm.mean())
                metrics["val_transition_mse_elite"] = float(jnp.sort(tr_pm)[:num_elites].mean())
            if log_disagreement:
                metrics["ensemble_disagreement"] = float(
                    self._ensemble_disagreement(stacked_params, tv_obs, tv_action)
                )
            log_fn(step_offset, **metrics)

        rng, loop_rng = jax.random.split(rng)
        params, opt_state = stacked_params, opt_state
        gen_base = 0
        for stage_horizon, stage_epochs in stages:
            train_windows = tile_episodes_to_windows(train_eps, stage_horizon)
            val_windows = tile_episodes_to_windows(val_eps, stage_horizon)
            n_windows = int(train_windows.start_obs.shape[0])
            base = gen_base  # bind for the closure

            def train_epoch(epoch, carry, base=base, horizon=stage_horizon,
                            train_windows=train_windows, val_windows=val_windows,
                            n_windows=n_windows):
                rng, params, opt_state = carry
                global_gen = base + epoch
                sigma = sigma_schedule(global_gen)
                rng, batch_rng = jax.random.split(rng)
                idxs = jax.random.randint(batch_rng, (batch_size,), 0, n_windows)
                batch = jax.tree.map(lambda x: x[idxs], train_windows)
                iterinfos = get_iterinfos(global_gen, pop)
                params, opt_state, member_losses = vmapped_step(
                    params, opt_state, train_es_key, sigma, iterinfos, batch
                )
                if log_fn is not None:
                    lr_value = lr_schedule(global_gen) if callable(lr_schedule) else lr_schedule
                    _emit_traj_log(
                        log_fn, params, global_gen, jnp.mean(member_losses),
                        self._per_member_traj_mse, val_windows,
                        self._per_member_val_mse, tv_obs, tv_action, tv_targets, log_transition,
                        lr_value, sigma, num_elites, batch_size, horizon, pop, n_ens,
                        log_interval, full_validation_interval, step_offset,
                    )
                return rng, params, opt_state

            loop_rng, stage_rng = jax.random.split(loop_rng)
            stage_rng, params, opt_state = jax.lax.fori_loop(
                0, stage_epochs, train_epoch, (stage_rng, params, opt_state)
            )
            gen_base += stage_epochs

        # Elite re-selection by trajectory val MSE at the final horizon (validation is
        # MSE-only — NLL is a training diagnostic — so elites are not chosen on the fitness).
        final_val_windows = tile_episodes_to_windows(val_eps, final_horizon)
        per_member_scalar, curve = self._per_member_traj_mse(params, final_val_windows)
        self._params = params
        self._opt_state = opt_state
        self._elite_idxs = jnp.argsort(per_member_scalar)[: self.num_elites]
        self._update_steps_completed = step_offset + sum(e for _, e in stages)
        self._validation_split = float(val_split)
        self._trajectory_validation_split = float(traj_val_split)
        self._seed = int(split_seed)

        # Final per-step MSE-vs-step curves (headline, train + val) + end-of-run disagreement.
        if log_fn is not None:
            final_train_windows = _sample_train_curve_windows(
                train_eps, final_horizon, int(final_val_windows.start_obs.shape[0]), split_seed
            )
            _, train_curve = self._per_member_traj_mse(params, final_train_windows)
            # Each per-step compounding-error curve goes to a single line panel (see
            # Logger.log_world_model_finetune_curve), not T separate scalar metrics.
            metrics: dict = {
                "val_traj_mse_curve": _mean_curve_list(curve),
                "train_traj_mse_curve": _mean_curve_list(train_curve),
            }
            if log_disagreement:
                metrics["ensemble_disagreement"] = float(
                    self._ensemble_disagreement(params, tv_obs, tv_action)
                )
            log_fn(self._update_steps_completed, **metrics)


# ── module-level helpers (kept out of the class to keep closures explicit) ──────


def _mean_curve_list(curve) -> list[float]:
    """All-member mean of a (num_ensemble, T) per-step MSE curve, as a plain list."""
    mean = curve.mean(axis=0)
    return [float(mean[t]) for t in range(int(mean.shape[0]))]


def _sample_train_curve_windows(
    train_eps, horizon: int, n_val_windows: int, seed: int
) -> TrajectoryWindows:
    """Train-episode windows for the train compounding-error curve, subsampled (without
    replacement) to the val-window count so the train curve matches the val curve's eval
    cost and estimation noise. The sample depends only on (seed, horizon), not the
    training RNG stream, so enabling/disabling logging cannot change the evolved params.
    """
    windows = tile_episodes_to_windows(train_eps, horizon)
    n_windows = int(windows.start_obs.shape[0])
    if n_windows <= n_val_windows:
        return windows
    key = jax.random.fold_in(jax.random.key(seed), horizon)
    idxs = jax.random.choice(key, n_windows, (n_val_windows,), replace=False)
    return jax.tree.map(lambda x: x[idxs], windows)


def _container(maybe_cfg):
    return (
        OmegaConf.to_container(maybe_cfg, resolve=True)
        if isinstance(maybe_cfg, DictConfig)
        else dict(maybe_cfg or {})
    )


def _resolve_schedule(eg_cfg, value_key, name_key, kwargs_key):
    name = str(eg_cfg.get(name_key, "constant"))
    kwargs = cast(dict, _container(eg_cfg.get(kwargs_key, {})))
    return build_schedule(float(eg_cfg[value_key]), name, kwargs)


def _resolve_sigma_schedule(eg_cfg) -> optax.Schedule:
    name = str(eg_cfg.get("sigma_schedule", "exponential"))
    kwargs = cast(dict, _container(eg_cfg.get("sigma_schedule_kwargs", {})))
    if name == "exponential" and not kwargs:
        kwargs = {"transition_steps": 1, "decay_rate": float(eg_cfg.sigma_decay_rate)}
    sched = build_schedule(float(eg_cfg.sigma), name, kwargs)
    return sched if callable(sched) else optax.constant_schedule(sched)


def _emit_backprop_log(
    log_fn, params, step, train_loss, val_obs, val_action, val_targets,
    per_member_val_mse, n_ens, num_elites, batch_size, n_val,
    log_interval, full_validation_interval, lr,
) -> None:
    should_validate = (step % full_validation_interval == 0)
    should_log = (step % log_interval == 0) | should_validate

    def _ts(s: int) -> int:  # transitions seen (shared batch -> not scaled by n_ens)
        return s * batch_size

    def _fe(s: int) -> int:  # forward evals (compute proxy -> scaled by n_ens)
        return s * batch_size * n_ens + (s // full_validation_interval) * n_val * n_ens

    def _train_cb(step_i, loss_i) -> None:
        s = int(step_i)
        log_fn(s, float(loss_i), float("nan"), _ts(s), _fe(s), lr=lr)

    def _val_cb(step_i, loss_i, val_mse_i, val_elite_i) -> None:
        s = int(step_i)
        log_fn(
            s, float(loss_i), float(val_mse_i), _ts(s), _fe(s),
            lr=lr, val_mse_elite=float(val_elite_i),
        )

    def _with_val(_):
        per_member = per_member_val_mse(params, val_obs, val_action, val_targets)
        jax.debug.callback(
            _val_cb, step, train_loss, per_member.mean(),
            jnp.sort(per_member)[:num_elites].mean(),
        )

    def _train_only(_):
        jax.debug.callback(_train_cb, step, train_loss)

    jax.lax.cond(
        should_log,
        lambda _: jax.lax.cond(should_validate, _with_val, _train_only, None),
        lambda _: None,
        None,
    )


def _emit_eggroll_log(
    log_fn, params, epoch, train_loss, per_member_val_mse, val_obs, val_action, val_targets,
    lr_schedule, sigma, n_prompts, pop, n_val, n_ens, num_elites,
    log_interval, full_validation_interval, step_offset,
) -> None:
    step = epoch + 1
    should_validate = (epoch == 0) | (step % full_validation_interval == 0)
    if callable(lr_schedule):
        lr_value = lr_schedule(epoch)
    else:
        lr_value = jnp.asarray(lr_schedule, jnp.float32)
    # forward_evals scales by n_ens (per-member population + val); transitions do not.
    pop_n, nval_n = pop * n_ens, n_val * n_ens

    def _train_cb(step_i, loss_i, lr_i, sigma_i) -> None:
        s = int(step_i)
        t, f = _eggroll_work_counters(s, n_prompts, pop_n, nval_n, full_validation_interval)
        log_fn(
            s + step_offset, float(loss_i), float("nan"), t, f,
            lr=float(lr_i), sigma=float(sigma_i),
        )

    def _val_cb(step_i, loss_i, val_mse_i, val_elite_i, lr_i, sigma_i) -> None:
        s = int(step_i)
        t, f = _eggroll_work_counters(s, n_prompts, pop_n, nval_n, full_validation_interval)
        log_fn(
            s + step_offset, float(loss_i), float(val_mse_i), t, f,
            lr=float(lr_i), sigma=float(sigma_i), val_mse_elite=float(val_elite_i),
        )

    def _with_val(_):
        per_member = per_member_val_mse(params, val_obs, val_action, val_targets)
        jax.debug.callback(
            _val_cb, step, train_loss, per_member.mean(),
            jnp.sort(per_member)[:num_elites].mean(), lr_value, sigma,
        )

    def _train_only(_):
        jax.debug.callback(_train_cb, step, train_loss, lr_value, sigma)

    jax.lax.cond(
        ((epoch + 1) % log_interval == 0) | should_validate,
        lambda _: jax.lax.cond(should_validate, _with_val, _train_only, None),
        lambda _: None,
        None,
    )


def _emit_traj_log(
    log_fn, params, global_gen, train_nll,
    per_member_traj_mse, val_windows,
    per_member_val_mse, tv_obs, tv_action, tv_targets, log_transition,
    lr_value, sigma, num_elites, batch_size, horizon, pop, n_ens,
    log_interval, full_validation_interval, step_offset,
) -> None:
    """Phase-2 logging: training NLL + MSE-only validation, on the ``world_model_ft`` axis.

    The Phase-2 ``log_fn`` takes ``(generation, **metrics)`` (its own W&B namespace), unlike
    the positional one-step ``log_fn``. Validation is MSE-only (NLL is a training diagnostic);
    each metric is reported as all-member mean + elite-mean, per the Phase-1 convention.
    """
    step = global_gen + 1
    should_validate = (global_gen == 0) | (step % full_validation_interval == 0)

    def _ts(s: int) -> int:  # transitions-seen proxy (shared batch -> not scaled by n_ens)
        return s * batch_size * horizon

    def _fe(s: int) -> int:  # forward-evals proxy (scaled by population and ensemble)
        return s * batch_size * horizon * pop * n_ens

    def _train_cb(step_i, nll_i, lr_i, sigma_i) -> None:
        s = int(step_i)
        log_fn(
            s + step_offset, train_loss=float(nll_i), lr=float(lr_i), sigma=float(sigma_i),
            transitions_seen=_ts(s), forward_evals=_fe(s),
        )

    def _val_cb(step_i, nll_i, tj_i, tj_e_i, tr_i, tr_e_i, lr_i, sigma_i) -> None:
        s = int(step_i)
        metrics = dict(
            train_loss=float(nll_i), val_traj_mse=float(tj_i),
            val_traj_mse_elite=float(tj_e_i), lr=float(lr_i), sigma=float(sigma_i),
            transitions_seen=_ts(s), forward_evals=_fe(s),
        )
        if log_transition:
            metrics["val_transition_mse"] = float(tr_i)
            metrics["val_transition_mse_elite"] = float(tr_e_i)
        log_fn(s + step_offset, **metrics)

    def _with_val(_):
        tj_pm, _ = per_member_traj_mse(params, val_windows)
        tj, tj_e = tj_pm.mean(), jnp.sort(tj_pm)[:num_elites].mean()
        if log_transition:
            tr_pm = per_member_val_mse(params, tv_obs, tv_action, tv_targets)
            tr, tr_e = tr_pm.mean(), jnp.sort(tr_pm)[:num_elites].mean()
        else:
            tr = tr_e = jnp.asarray(jnp.nan)
        jax.debug.callback(_val_cb, step, train_nll, tj, tj_e, tr, tr_e, lr_value, sigma)

    def _train_only(_):
        jax.debug.callback(_train_cb, step, train_nll, lr_value, sigma)

    jax.lax.cond(
        (step % log_interval == 0) | should_validate,
        lambda _: jax.lax.cond(should_validate, _with_val, _train_only, None),
        lambda _: None,
        None,
    )
