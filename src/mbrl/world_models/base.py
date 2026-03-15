"""Shared ensemble world model interface.

Adapted from Unifloral dynamics.py (https://github.com/EmptyJackson/unifloral).
This module defines the data structures and interface that all world model
training methods (MLE, EGGROLL) implement.
"""


class EnsembleDynamics:
    """Abstract ensemble dynamics model.

    All world model training methods should produce an object satisfying
    this interface so that policy training algorithms are method-agnostic.
    """

    def step(self, obs, action, rng):
        """Return (next_obs, reward, terminal) sampled from the ensemble."""
        raise NotImplementedError

    def train(self, dataset, cfg, rng):
        """Fit the ensemble to the provided offline dataset."""
        raise NotImplementedError
