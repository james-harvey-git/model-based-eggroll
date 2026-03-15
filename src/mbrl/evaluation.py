"""Shared evaluation utilities for D4RL benchmarks."""


def evaluate_policy(policy, env, num_episodes: int = 10):
    """Roll out a policy in the real environment and return normalised D4RL score."""
    raise NotImplementedError
