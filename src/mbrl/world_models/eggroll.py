"""EGGROLL-trained ensemble world model.

Uses the EGGROLL evolution strategy in place of gradient-based MLE to fit the
ensemble to offline data. Full implementation deferred to Phase 4/5.
"""

from mbrl.world_models.base import EnsembleDynamics


class EGGROLLEnsemble(EnsembleDynamics):
    """Ensemble of dynamics models fitted via EGGROLL."""

    def step(self, obs, action, rng):
        raise NotImplementedError

    def train(self, dataset, cfg, rng, log_fn=None):
        raise NotImplementedError
