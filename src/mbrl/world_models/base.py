"""Shared ensemble world model interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from mbrl.data import Transition


class EnsembleDynamics(ABC):
    """Abstract base class for ensemble dynamics models.

    All world model training methods (MLE, EGGROLL) implement this interface so
    that policy training algorithms are agnostic to how the world model was trained.
    """

    @property
    @abstractmethod
    def termination_fn(self) -> Callable:
        """Environment-specific termination function for model-based rollouts."""

    @abstractmethod
    def step(
        self,
        obs: jnp.ndarray,  # (obs_dim,)
        action: jnp.ndarray,  # (act_dim,)
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample (next_obs, reward, done) from the ensemble.

        Operates on a single (obs, action) pair. Callers vmap over batches.
        """

    @abstractmethod
    def predict_ensemble(
        self,
        obs: jnp.ndarray,  # (obs_dim,)
        action: jnp.ndarray,  # (act_dim,)
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (ensemble_mean, ensemble_std) for all elite members.

        Operates on a single (obs, action) pair. Callers vmap over batches.

        Returns:
            ensemble_mean: (num_elites, obs_dim + 1) -- predicted delta_obs + reward
            ensemble_std:  (num_elites, obs_dim + 1)
        """

    @abstractmethod
    def train(
        self,
        dataset: Transition,
        cfg: DictConfig,
        rng: jax.Array,
        log_fn: Callable[..., None] | None = None,
    ) -> None:
        """Fit the ensemble to the provided offline dataset."""
