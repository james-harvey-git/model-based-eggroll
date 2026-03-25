"""EGGROLL policy search within a world model.

Uses the EGGROLL evolution strategy (see mbrl.optimisers.eggroll) to search
for a policy by rolling out trajectories inside the learned world model.
"""

def train(world_model, dataset, cfg, rng):
    """Train a policy with EGGROLL using the provided world model."""
    raise NotImplementedError
