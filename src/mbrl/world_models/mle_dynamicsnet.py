"""Single-network MLE world model trained over a `DynamicsNet`.

Stage 1 of the hybrid MLE→EGGROLL training paradigm (see GitHub issue #30).
Uses the EGGROLL `DynamicsNet` primitive (not the Flax `MLEEnsemble` network) so
that Stage 2 can load these params directly into `EGGROLLEnsemble` without any
shape conversion.

Trains with standard backprop + AdamW. The forward path wraps the base `Noiser`
class (whose `do_mm` and `get_noisy_standard` are no-ops) and passes
`iterinfo=None`, giving a vanilla MLP forward through `DynamicsNet`.
"""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
import optax

from mbrl.data import Transition, create_epoch_iterator, train_val_split
from mbrl.eggroll.networks import DynamicsNet
from mbrl.eggroll.primitives import CommonParams, Noiser, simple_es_tree_key
from mbrl.world_models.base import EnsembleDynamics
from mbrl.world_models.termination_fns import get_termination_fn


def _mle_dyn_work_counters(
    epoch: int,
    train_examples_per_epoch: int,
    forward_evals_per_epoch: int,
    init_forward_evals: int,
) -> tuple[int, int]:
    """Cumulative work counters as Python ints (single network → no ensemble factor)."""
    return (
        train_examples_per_epoch * epoch,
        init_forward_evals + forward_evals_per_epoch * epoch,
    )


class MLEDynamicsNet(EnsembleDynamics):
    """MLE-trained single `DynamicsNet`.

    Implements the `EnsembleDynamics` ABC for parity with the other world models.
    `predict_ensemble` returns `num_members` copies of the single prediction
    (zero epistemic diversity) — useful only as a sanity-eval baseline. The
    primary purpose of this class is to produce a Stage-1 checkpoint that
    `EGGROLLEnsemble` can warm-start from.
    """

    def __init__(self, obs_dim: int, act_dim: int, dataset_id: str, cfg: DictConfig):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._cfg = cfg
        self._termination_fn = get_termination_fn(dataset_id)

        # Populated by train() / load_from_checkpoint()
        self._params: dict | None = None
        self._opt_state = None
        self._frozen_params: dict | None = None
        self._scan_map = None
        self._es_tree_key = None
        self._update_steps_completed: int = 0
        self._validation_split: float | None = None
        self._seed: int | None = None

    @classmethod
    def load_from_checkpoint(cls, path: str | Path) -> "MLEDynamicsNet":
        """Reconstruct a trained MLEDynamicsNet from a checkpoint file."""
        import pickle

        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        wm_cfg = OmegaConf.create(ckpt["world_model_cfg"])
        assert isinstance(wm_cfg, DictConfig)
        instance = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["dataset_id"], wm_cfg)
        # Replay rand_init only to recover frozen_params + scan_map for the forward path.
        # Loaded params overwrite the random ones immediately below.
        common_init = DynamicsNet.rand_init(
            jax.random.key(0),
            obs_dim=ckpt["obs_dim"],
            act_dim=ckpt["act_dim"],
            hidden_dims=list(wm_cfg.hidden_dims),
            activation=wm_cfg.activation,
            init_scheme=str(wm_cfg.get("init_scheme", "eggroll")),
            backbone=str(wm_cfg.get("backbone", "mlp")),
            max_logvar_init=float(wm_cfg.get("max_logvar_init", 0.5)),
            min_logvar_init=float(wm_cfg.get("min_logvar_init", -10.0)),
        )
        instance._frozen_params = common_init.frozen_params
        instance._scan_map = common_init.scan_map
        instance._es_tree_key = simple_es_tree_key(
            common_init.params, jax.random.key(0), common_init.scan_map
        )
        instance._params = ckpt["params"]
        instance._opt_state = ckpt["opt_state"]
        instance._update_steps_completed = int(ckpt["update_steps_completed"])
        instance._validation_split = float(ckpt["validation_split"])
        instance._seed = int(ckpt["seed"])
        return instance

    @property
    def termination_fn(self) -> Callable:
        return self._termination_fn

    def checkpoint_state(self) -> dict:
        """Return the class-specific payload for the world-model checkpoint."""
        assert self._params is not None, "Must call train() before checkpoint_state()"
        return {
            "params": self._params,
            "opt_state": self._opt_state,
            "update_steps_completed": self._update_steps_completed,
            "validation_split": self._validation_split,
            "seed": self._seed,
        }

    def _unperturbed_forward(
        self,
        params: dict,
        obs: jnp.ndarray,
        action: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward through DynamicsNet using the base Noiser (no-op perturbations)."""
        common = CommonParams(
            noiser=Noiser,
            frozen_noiser_params={},
            noiser_params={},
            frozen_params=self._frozen_params,
            params=params,
            es_tree_key=self._es_tree_key,
            iterinfo=None,
        )
        return DynamicsNet._forward_with_bounds(common, obs, action)

    def predict_ensemble(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return `(num_members,)` copies of the single network's prediction."""
        assert self._params is not None, "Must call train() before predict_ensemble()"
        mean, logvar, _, _ = self._unperturbed_forward(self._params, obs, action)
        std = jnp.exp(0.5 * logvar)
        num_members = int(self._cfg.get("num_members", 1))
        ensemble_mean = jnp.broadcast_to(mean, (num_members,) + mean.shape)
        ensemble_std = jnp.broadcast_to(std, (num_members,) + std.shape)
        return ensemble_mean, ensemble_std

    def compute_val_mse(self, dataset: Transition) -> jax.Array:
        """Mean MSE over *dataset*, matching the single-network ``val_mse`` log.

        Iterates contiguous chunks of size 1024 with an SSE/count accumulator so
        all N transitions are covered (no shuffling, no tail-drop).
        """
        assert self._params is not None, "Must call train() before compute_val_mse()"
        params = self._params
        delta_obs = dataset.next_obs - dataset.obs
        targets = jnp.concatenate([delta_obs, dataset.reward[:, None]], axis=-1)

        @jax.jit
        def _batch_sse(obs_b: jax.Array, action_b: jax.Array, bt: jax.Array) -> jax.Array:
            mean, _, _, _ = jax.vmap(
                lambda o, a: self._unperturbed_forward(params, o, a), in_axes=(0, 0)
            )(obs_b, action_b)
            return ((mean - bt) ** 2).sum()

        n = int(dataset.obs.shape[0])
        batch_size = 1024
        num_full = n // batch_size
        sse = jnp.zeros((), dtype=jnp.float32)
        if num_full > 0:
            full_obs = dataset.obs[: num_full * batch_size].reshape(num_full, batch_size, -1)
            full_action = dataset.action[: num_full * batch_size].reshape(num_full, batch_size, -1)
            full_targets = targets[: num_full * batch_size].reshape(num_full, batch_size, -1)
            sse, _ = jax.lax.scan(
                lambda c, b: (c + _batch_sse(b[0], b[1], b[2]), None),
                sse,
                (full_obs, full_action, full_targets),
            )
        tail = n - num_full * batch_size
        if tail > 0:
            sse = sse + _batch_sse(
                dataset.obs[-tail:], dataset.action[-tail:], targets[-tail:]
            )
        elements = n * int(targets.shape[1])
        return sse / elements

    def step(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample `(next_obs, reward, done)` from the trained network."""
        assert self._params is not None, "Must call train() before step()"
        mean, logvar, _, _ = self._unperturbed_forward(self._params, obs, action)
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
        """Fit a single `DynamicsNet` to the dataset via NLL minimisation."""
        seed = cfg.get("seed", None)
        assert seed is not None, (
            "MLEDynamicsNet requires cfg.seed to be set (so Stage 2 can replay the "
            "train/val partition). experiments/world_model.py populates this from "
            "the top-level cfg.seed."
        )

        # DO NOT REORDER: Stage 2 (`EGGROLLEnsemble.train` with `init_checkpoint`)
        # replays exactly `jax.random.split(rng, 3)` from `jax.random.key(seed)` to
        # recover the same train/val partition. Changing this split chain breaks
        # the val-set parity guard (see tests/test_world_models.py
        # ::TestHybridHandoff::test_val_set_parity).
        rng, split_rng, init_rng = jax.random.split(rng, 3)

        train_data, val_data = train_val_split(dataset, cfg.validation_split, split_rng)

        # (inputs, targets): predict (delta_obs, reward) given (obs, action)
        def _make_inputs_targets(data: Transition):
            inputs = jnp.concatenate([data.obs, data.action], axis=-1)
            delta_obs = data.next_obs - data.obs
            targets = jnp.concatenate([delta_obs, data.reward[:, None]], axis=-1)
            return inputs, targets

        train_inputs, train_targets = _make_inputs_targets(train_data)
        val_inputs, val_targets = _make_inputs_targets(val_data)

        # Initialise the network and capture frozen_params / scan_map for inference.
        common_init = DynamicsNet.rand_init(
            init_rng,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dims=list(cfg.hidden_dims),
            activation=cfg.activation,
            init_scheme=str(cfg.get("init_scheme", "eggroll")),
            backbone=str(cfg.get("backbone", "mlp")),
            max_logvar_init=float(cfg.get("max_logvar_init", 0.5)),
            min_logvar_init=float(cfg.get("min_logvar_init", -10.0)),
        )
        self._frozen_params = common_init.frozen_params
        self._scan_map = common_init.scan_map
        # `es_tree_key` is structurally required by `call_submodule` even though
        # its values go unused at `iterinfo=None` (Noiser.get_noisy_standard
        # ignores the key).
        self._es_tree_key = simple_es_tree_key(
            common_init.params, jax.random.key(0), common_init.scan_map
        )
        params = common_init.params

        tx = optax.adamw(cfg.lr, eps=1e-5, weight_decay=cfg.weight_decay)
        opt_state = tx.init(params)

        batch_size = int(cfg.batch_size)
        batches_per_epoch = train_inputs.shape[0] // batch_size
        train_examples_per_epoch = batches_per_epoch * batch_size
        val_examples_per_epoch = (val_inputs.shape[0] // batch_size) * batch_size
        if val_examples_per_epoch == 0:
            val_examples_per_epoch = val_inputs.shape[0]
        forward_evals_per_epoch = train_examples_per_epoch + val_examples_per_epoch
        init_forward_evals = val_examples_per_epoch

        logvar_diff_coef = float(cfg.get("logvar_diff_coef", 0.01))

        def _loss_fn(params, inputs, targets):
            obs_b, action_b = inputs[..., : self.obs_dim], inputs[..., self.obs_dim :]
            mean, logvar, _, _ = jax.vmap(
                lambda o, a: self._unperturbed_forward(params, o, a), in_axes=(0, 0)
            )(obs_b, action_b)
            mse_loss = jnp.mean((targets - mean) ** 2 * jnp.exp(-logvar))
            var_loss = jnp.mean(logvar)
            # Use the unbroadcast (D,)-shaped parameters directly (matching
            # MLEEnsemble at mle.py:131-132). Going via the vmapped forward
            # returns max/min_logvar with a batch axis, which would scale this
            # penalty by batch_size and (paradoxically) collapse max and min
            # together to large negative values — see GitHub PR #31 discussion.
            max_logvar = params["max_logvar"]
            min_logvar = params["min_logvar"]
            logvar_diff = jnp.sum(max_logvar - min_logvar)
            loss = mse_loss + var_loss + logvar_diff_coef * logvar_diff
            return loss

        def _train_step(carry, batch):
            params, opt_state = carry
            inputs, targets = batch
            loss, grads = jax.value_and_grad(_loss_fn)(params, inputs, targets)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        def _eval_step(carry, batch):
            params = carry
            inputs, targets = batch
            obs_b, action_b = inputs[..., : self.obs_dim], inputs[..., self.obs_dim :]
            mean, _, _, _ = jax.vmap(
                lambda o, a: self._unperturbed_forward(params, o, a), in_axes=(0, 0)
            )(obs_b, action_b)
            mse = jnp.mean((mean - targets) ** 2)
            return params, mse

        def _initial_val_mse(params):
            val_iter = create_epoch_iterator(
                (val_inputs, val_targets), batch_size, jax.random.key(0)
            )
            _, val_mses = jax.lax.scan(_eval_step, params, val_iter)
            return val_mses.mean()

        if log_fn is not None:
            init_val_mse = _initial_val_mse(params)
            log_fn(
                0,
                float("nan"),
                float(init_val_mse),
                0,
                init_forward_evals,
                epoch=0,
            )

        def train_epoch(epoch, carry):
            rng, params, opt_state = carry
            rng, rng_train, rng_val = jax.random.split(rng, 3)

            train_iter = create_epoch_iterator(
                (train_inputs, train_targets), batch_size, rng_train
            )
            (params, opt_state), batch_losses = jax.lax.scan(
                _train_step, (params, opt_state), train_iter
            )

            val_iter = create_epoch_iterator((val_inputs, val_targets), batch_size, rng_val)
            _, val_mses = jax.lax.scan(_eval_step, params, val_iter)

            if log_fn is not None:
                _log_fn = log_fn

                def _log_callback(epoch_i, train_loss_i, val_mse_i) -> None:
                    epoch_py = int(epoch_i) + 1
                    update_step = epoch_py * batches_per_epoch
                    transitions_seen, forward_evals = _mle_dyn_work_counters(
                        epoch_py,
                        train_examples_per_epoch,
                        forward_evals_per_epoch,
                        init_forward_evals,
                    )
                    _log_fn(
                        update_step,
                        float(train_loss_i),
                        float(val_mse_i),
                        transitions_seen,
                        forward_evals,
                        epoch=epoch_py,
                    )

                jax.debug.callback(
                    _log_callback,
                    epoch,
                    batch_losses.mean(),
                    val_mses.mean(),
                )

            return rng, params, opt_state

        rng, train_rng = jax.random.split(rng)
        init_carry = (train_rng, params, opt_state)
        _, params, opt_state = jax.lax.fori_loop(0, cfg.num_epochs, train_epoch, init_carry)

        self._params = params
        self._opt_state = opt_state
        self._update_steps_completed = int(cfg.num_epochs) * batches_per_epoch
        self._validation_split = float(cfg.validation_split)
        self._seed = int(seed)
