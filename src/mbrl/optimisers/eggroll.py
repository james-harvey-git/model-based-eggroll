"""EGGROLL optimiser wrapper.

Core implementation of the EGGROLL evolution strategy from:
https://github.com/ESHyperscale/HyperscaleES

Used by both world model training (mbrl.world_models.eggroll) and
policy search (mbrl.algorithms.eggroll).
"""


class EGGROLLOptimiser:
    """Wraps EGGROLL for use as a drop-in optimiser."""

    def ask(self):
        """Return a population of candidate parameter vectors."""
        raise NotImplementedError

    def tell(self, fitnesses):
        """Update the distribution given a vector of fitness scores."""
        raise NotImplementedError
