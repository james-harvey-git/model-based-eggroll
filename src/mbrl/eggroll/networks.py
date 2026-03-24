"""Domain-specific EGGROLL Model subclasses.

Defines EnsembleDynamics and Policy as compositions of EGGROLL's
MM, Linear, and MLP primitives (from mbrl.eggroll.primitives).
Used exclusively by the EGGROLL training paths (world_models/eggroll.py
and algorithms/eggroll.py). The MLE baseline uses separate Flax models
defined in world_models/mle.py.
"""
