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
    def train(
        self,
        dataset: Transition,
        cfg: DictConfig,
        rng: jax.Array,
        log_fn: Callable[..., None] | None = None,
    ) -> None:
        """Fit the ensemble to the provided offline dataset."""
