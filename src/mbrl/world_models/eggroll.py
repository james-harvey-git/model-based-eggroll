"""EGGROLL-trained ensemble world model.

Uses the EGGROLL evolution strategy (see mbrl.optimisers.eggroll) in place of
gradient-based MLE to fit the ensemble to offline D4RL data.
"""

from mbrl.world_models.base import EnsembleDynamics
from mbrl.optimisers.eggroll import EGGROLLOptimiser


class EGGROLLEnsemble(EnsembleDynamics):
    """Ensemble of dynamics models fitted via EGGROLL."""

    def step(self, obs, action, rng):
        raise NotImplementedError

    def train(self, dataset, cfg, rng):
        raise NotImplementedError
