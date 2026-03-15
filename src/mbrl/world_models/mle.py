"""Standard MLE ensemble training baseline.

Closely follows the Unifloral dynamics.py implementation:
https://github.com/EmptyJackson/unifloral
"""

from mbrl.world_models.base import EnsembleDynamics


class MLEEnsemble(EnsembleDynamics):
    """Ensemble of dynamics models trained by maximum likelihood estimation."""

    def step(self, obs, action, rng):
        raise NotImplementedError

    def train(self, dataset, cfg, rng):
        raise NotImplementedError
