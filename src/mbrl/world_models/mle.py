"""Standard MLE ensemble world model.

Closely follows the Unifloral dynamics.py implementation:
https://github.com/EmptyJackson/unifloral (algorithms/dynamics.py)
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
import optax

from mbrl.data import Transition, create_epoch_iterator, train_val_split
from mbrl.world_models.base import EnsembleDynamics
from mbrl.world_models.termination_fns import get_termination_fn

# ---------------------------------------------------------------------------
# Flax network (ported verbatim from Unifloral)
# ---------------------------------------------------------------------------


class SingleDynamicsModel(nn.Module):
    obs_dim: int
    n_layers: int
    layer_size: int

    @nn.compact
    def __call__(self, obs_action):
        x = obs_action
        for _ in range(self.n_layers):
            x = nn.relu(nn.Dense(self.layer_size)(x))
        obs_reward_stats = nn.Dense(2 * (self.obs_dim + 1))(x)
        return obs_reward_stats


class EnsembleDynamicsModel(nn.Module):
    obs_dim: int
    action_dim: int
    num_ensemble: int
    n_layers: int
    layer_size: int
    max_logvar_init: float = 0.5
    min_logvar_init: float = -10.0

    @nn.compact
    def __call__(self, obs_action):
        # Compute ensemble predictions via vmap over ensemble members
        batched_model = nn.vmap(
            SingleDynamicsModel,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,  # type: ignore[arg-type]  # Flax stubs incorrectly require int
            out_axes=0,
            axis_size=self.num_ensemble,
        )
        ensemble = batched_model(
            obs_dim=self.obs_dim,
            n_layers=self.n_layers,
            layer_size=self.layer_size,
            name="ensemble",
        )
        output = ensemble(obs_action)
        pred_mean, logvar = jnp.split(output, 2, axis=-1)

        # Soft-clamp log-variance to prevent unbounded growth
        max_logvar = self.param(
            "max_logvar",
            init_fn=lambda key: jnp.full((self.obs_dim + 1,), self.max_logvar_init),
        )
        min_logvar = self.param(
            "min_logvar",
            init_fn=lambda key: jnp.full((self.obs_dim + 1,), self.min_logvar_init),
        )
        logvar = max_logvar - nn.softplus(max_logvar - logvar)
        logvar = min_logvar + nn.softplus(logvar - min_logvar)
        return pred_mean, logvar


# ---------------------------------------------------------------------------
# Training utilities (ported from Unifloral)
# ---------------------------------------------------------------------------


def _train_dynamics(
    train_state, cfg, train_inputs, train_targets, val_inputs, val_targets, rng, log_fn=None
):
    """Train the ensemble and return (train_state, elite_idxs).

    Ported from Unifloral's train_dynamics_model function.
    """
    num_elites = cfg.num_elites
    batch_size = cfg.batch_size
    logvar_diff_coef = cfg.logvar_diff_coef
    num_ensemble = cfg.num_ensemble
    train_examples_per_epoch = (train_inputs.shape[0] // batch_size) * batch_size
    val_examples_per_epoch = (val_inputs.shape[0] // batch_size) * batch_size
    forward_evals_per_epoch = (train_examples_per_epoch + val_examples_per_epoch) * num_ensemble

    def _train_step(train_state, batch):
        inputs, targets = batch

        def _loss_fn(params):
            mean, logvar = train_state.apply_fn(params, inputs)
            # NLL loss: sum over ensemble members, mean over batch and features
            mse_loss = ((mean - targets) ** 2) * jnp.exp(-logvar)
            mse_loss = mse_loss.sum(0).mean()
            var_loss = logvar.sum(0).mean()
            max_logvar = params["params"]["max_logvar"]
            min_logvar = params["params"]["min_logvar"]
            logvar_diff = (max_logvar - min_logvar).sum()
            loss = mse_loss + var_loss + logvar_diff_coef * logvar_diff
            return loss, {"loss": loss, "mse_loss": mse_loss, "var_loss": var_loss}

        grads, aux = jax.grad(_loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, aux["loss"]

    def _eval_step(train_state, batch):
        inputs, targets = batch

        def _loss_fn(params):
            mean_predictions, _ = train_state.apply_fn(params, inputs)
            # Per-member MSE: mean over batch and output features → (num_ensemble,)
            loss = jnp.mean(((mean_predictions - targets) ** 2), axis=(1, 2))
            return loss

        loss = _loss_fn(train_state.params)
        return train_state, loss

    def train_epoch(epoch, carry):
        rng, train_state, elite_idxs = carry
        rng, rng_train, rng_val = jax.random.split(rng, 3)

        train_iter = create_epoch_iterator((train_inputs, train_targets), batch_size, rng_train)
        train_state, batch_losses = jax.lax.scan(_train_step, train_state, train_iter)

        val_iter = create_epoch_iterator((val_inputs, val_targets), batch_size, rng_val)
        _, val_losses = jax.lax.scan(_eval_step, train_state, val_iter)

        # Select elite members: lowest mean validation MSE across batches
        val_mse_per_member = val_losses.mean(axis=0)  # (num_ensemble,)
        elite_idxs = val_mse_per_member.argsort()[:num_elites]

        if log_fn is not None:
            transitions_seen = train_examples_per_epoch * (epoch + 1)
            forward_evals = forward_evals_per_epoch * (epoch + 1)
            jax.debug.callback(
                log_fn,
                epoch,
                batch_losses.mean(),
                val_mse_per_member.mean(),
                transitions_seen,
                forward_evals,
            )

        return rng, train_state, elite_idxs

    dummy_elite_idxs = jnp.zeros((num_elites,), jnp.int32)
    init_carry = (rng, train_state, dummy_elite_idxs)
    _, train_state, elite_idxs = jax.lax.fori_loop(0, cfg.num_epochs, train_epoch, init_carry)
    return train_state, elite_idxs


# ---------------------------------------------------------------------------
# MLEEnsemble
# ---------------------------------------------------------------------------


class MLEEnsemble(EnsembleDynamics):
    """Ensemble of dynamics models trained by maximum likelihood estimation.

    Closely follows Unifloral's dynamics.py implementation.
    """

    def __init__(self, obs_dim: int, act_dim: int, dataset_id: str, cfg: DictConfig):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model = EnsembleDynamicsModel(
            obs_dim=obs_dim,
            action_dim=act_dim,
            num_ensemble=cfg.num_ensemble,
            n_layers=cfg.n_layers,
            layer_size=cfg.layer_size,
        )
        self._termination_fn = get_termination_fn(dataset_id)
        self.params = None  # populated by train()
        self.num_elites = None  # populated by train()

    @classmethod
    def load_from_checkpoint(cls, path: str | Path) -> "MLEEnsemble":
        """Reconstruct a trained MLEEnsemble from a checkpoint file."""
        import pickle

        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        wm_cfg = OmegaConf.create(ckpt["world_model_cfg"])
        # Override num_ensemble to match elite-pruned checkpoint params
        wm_cfg_override = OmegaConf.create(
            {**OmegaConf.to_container(wm_cfg), "num_ensemble": ckpt["num_elites"]}  # type: ignore[arg-type]
        )
        instance = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["dataset_id"], wm_cfg_override)
        instance.params = ckpt["params"]
        instance.num_elites = ckpt["num_elites"]
        return instance

    @property
    def termination_fn(self) -> Callable:
        return self._termination_fn

    def predict_ensemble(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (ensemble_mean, ensemble_std) for all elite members."""
        assert self.params is not None, "Model must be trained before calling predict_ensemble()"
        obs_action = jnp.concatenate([obs, action], axis=-1)
        ensemble_mean, ensemble_logvar = cast(
            tuple[jax.Array, jax.Array], self.model.apply(self.params, obs_action)
        )
        ensemble_std = jnp.exp(0.5 * ensemble_logvar)
        return ensemble_mean, ensemble_std

    def train(
        self,
        dataset: Transition,
        cfg: DictConfig,
        rng: jax.Array,
        log_fn: Callable[..., None] | None = None,
    ) -> None:
        """Fit the ensemble to the offline dataset via NLL minimisation."""
        rng, split_rng, init_rng = jax.random.split(rng, 3)

        # Split into train / val partitions
        train_data, val_data = train_val_split(dataset, cfg.validation_split, split_rng)

        # Construct (inputs, targets) arrays — model predicts delta_obs + reward
        def _make_inputs_targets(data: Transition):
            inputs = jnp.concatenate([data.obs, data.action], axis=-1)
            delta_obs = data.next_obs - data.obs
            targets = jnp.concatenate([delta_obs, data.reward[:, None]], axis=-1)
            return inputs, targets

        train_inputs, train_targets = _make_inputs_targets(train_data)
        val_inputs, val_targets = _make_inputs_targets(val_data)

        # Initialise train state
        dummy_input = jnp.zeros(self.obs_dim + self.act_dim)
        params = self.model.init(init_rng, dummy_input)
        train_state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adamw(cfg.lr, eps=1e-5, weight_decay=cfg.weight_decay),
        )

        # Train
        rng, train_rng = jax.random.split(rng)
        train_state, elite_idxs = _train_dynamics(
            train_state, cfg, train_inputs, train_targets, val_inputs, val_targets, train_rng,
            log_fn=log_fn,
        )

        # Prune non-elite params and create fresh model sized to elites only
        # (matching Unifloral lines 135-147, but avoiding mutable module mutation)
        params = jax.tree.map(lambda x: x, train_state.params)  # copy
        ensemble_params = params["params"]["ensemble"]
        ensemble_params = jax.tree.map(lambda p: p[elite_idxs], ensemble_params)
        params["params"]["ensemble"] = ensemble_params
        self.params = params
        self.num_elites = len(elite_idxs)
        self.model = EnsembleDynamicsModel(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            num_ensemble=self.num_elites,
            n_layers=self.model.n_layers,
            layer_size=self.model.layer_size,
        )

    def step(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample (next_obs, reward, done) from a randomly selected elite member."""
        assert self.num_elites is not None, "Model must be trained before calling step()"
        rng_elite, rng_noise = jax.random.split(rng)

        ensemble_mean, ensemble_std = self.predict_ensemble(obs, action)

        # Randomly select one elite member
        sample_idx = jax.random.randint(rng_elite, (), 0, self.num_elites)
        mean = ensemble_mean[sample_idx]
        std = ensemble_std[sample_idx]

        # Sample from Gaussian (reparameterization)
        noise = jax.random.normal(rng_noise, shape=mean.shape)
        sample = mean + noise * std

        # Reconstruct next_obs and reward
        delta_obs, reward = sample[:-1], sample[-1]
        next_obs = obs + delta_obs
        done = self.termination_fn(obs, action, next_obs)

        return next_obs, reward, done
