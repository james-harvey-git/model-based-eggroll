"""EGGROLL-trained ensemble world model.

Uses the EGGROLL evolution strategy in place of gradient-based MLE to fit the
ensemble to offline data. Full implementation deferred to Phase 4/5.
"""

from collections.abc import Callable

from mbrl.world_models.base import EnsembleDynamics
from mbrl.world_models.termination_fns import get_termination_fn


class EGGROLLEnsemble(EnsembleDynamics):
    """Ensemble of dynamics models fitted via EGGROLL."""

    def __init__(self, dataset_id: str):
        self._termination_fn = get_termination_fn(dataset_id)

    @property
    def termination_fn(self) -> Callable:
        return self._termination_fn

    def predict_ensemble(self, obs, action):
        raise NotImplementedError

    def step(self, obs, action, rng):
        raise NotImplementedError

    def train(self, dataset, cfg, rng, log_fn=None):
        raise NotImplementedError
