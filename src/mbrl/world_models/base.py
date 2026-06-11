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

    @property
    def discrepancy(self) -> float | None:
        """Calibrated elite-disagreement scale for MoReL's halt penalty.

        ``None`` unless :meth:`precompute_term_stats` has run (or a checkpoint that
        recorded it was loaded). See ``world_models/term_stats.py``.
        """
        return getattr(self, "_discrepancy", None)

    @property
    def min_r(self) -> float | None:
        """Minimum dataset reward, used as MoReL's halt-state reward floor."""
        return getattr(self, "_min_r", None)

    @property
    def predicts_logvar(self) -> bool:
        """Whether the model has an aleatoric (log-variance) head.

        ``False`` for deterministic-dynamics models (no variance head); then
        :meth:`predict_ensemble` returns ``ensemble_std=None`` and all uncertainty is
        epistemic (ensemble disagreement). Defaults to ``True``; subclasses override.
        """
        return getattr(self, "_predicts_logvar", True)

    def precompute_term_stats(self, dataset: Transition, rng: jax.Array) -> None:
        """Compute and store MoReL's halt-penalty statistics (``discrepancy``, ``min_r``).

        Concrete on the base class — it only needs ``predict_ensemble`` — so every
        subclass inherits it. Expensive (sweeps the dataset); call once after
        ``train()``. Subclasses persist the results via ``checkpoint_state()``.
        """
        from mbrl.world_models.term_stats import compute_model_discrepancy

        self._discrepancy: float | None = compute_model_discrepancy(self, dataset, rng)
        self._min_r: float | None = float(jnp.min(dataset.reward))

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
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Return (ensemble_mean, ensemble_std) for all elite members.

        Operates on a single (obs, action) pair. Callers vmap over batches.

        Returns:
            ensemble_mean: (num_elites, obs_dim + 1) -- predicted delta_obs + reward
            ensemble_std:  (num_elites, obs_dim + 1), or ``None`` when the model has no
                aleatoric head (``predicts_logvar`` is ``False``).
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

    @abstractmethod
    def compute_val_mse(self, dataset: Transition) -> jax.Array:
        """Return a scalar validation MSE for *dataset* matching the training-time formula.

        Each subclass implements the same scalar its training loop logs as ``val_mse``
        (or ``val_mse_elite`` in the ensemble case). Iteration is deterministic over
        contiguous chunks covering all transitions — no shuffling, no tail drop — so the
        returned number is reproducible and well-defined on arbitrary datasets.
        """


def load_world_model_from_checkpoint(path: str) -> "EnsembleDynamics":
    """Load a world model, resolving its class from the checkpoint's ``_target_``.

    Lets callers (policy training, eval) stay agnostic to which world-model class
    produced the checkpoint — the class is whatever the training run recorded in
    ``world_model_cfg._target_``.
    """
    import pickle

    from hydra.utils import get_class

    with open(path, "rb") as f:
        target = pickle.load(f)["world_model_cfg"]["_target_"]
    return get_class(target).load_from_checkpoint(path)
