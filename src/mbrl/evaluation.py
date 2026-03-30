"""Shared evaluation utilities: policy rollout and score normalisation."""

from collections.abc import Callable
from statistics import mean

import gymnasium
import minari
import numpy as np


def rollout_policy(
    policy: Callable[[np.ndarray], np.ndarray],
    env: gymnasium.Env,
    num_episodes: int,
    seed: int,
) -> list[float]:
    """Roll out a policy in a Gymnasium environment.

    Args:
        policy: Maps observation → action. May return JAX or NumPy arrays.
        env: A Gymnasium environment instance.
        num_episodes: Number of episodes to run.
        seed: Seed for the first episode; increments for subsequent episodes.

    Returns:
        A list of per-episode undiscounted returns.
    """
    returns = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        ep_return = 0.0
        done = False
        while not done:
            action = np.asarray(policy(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    return returns


def evaluate_policy(
    policy: Callable[[np.ndarray], np.ndarray],
    dataset_id: str,
    num_episodes: int = 10,
    seed: int = 0,
) -> float:
    """Evaluate a policy in the real environment and return mean episodic return.

    Creates the environment from the Minari dataset metadata via
    ``recover_environment()``, so the env always matches the dataset.

    Does NOT normalise — use :func:`compute_normalized_score` separately.
    """
    ds = minari.load_dataset(dataset_id)
    env = ds.recover_environment()
    try:
        returns = rollout_policy(policy, env, num_episodes, seed)
    finally:
        env.close()
    return mean(returns)


# ---------------------------------------------------------------------------
# Reference scores for normalisation
# ---------------------------------------------------------------------------

_reference_score_cache: dict[str, tuple[float, float]] = {}


def get_reference_scores(dataset_id: str) -> tuple[float, float]:
    """Return ``(random_ref, expert_ref)`` for the environment of *dataset_id*.

    Expert score is the mean episode return from the Minari expert dataset.
    Random score is the mean return of ``env.action_space.sample()`` over 100
    episodes (seed 0).  Results are cached per environment so repeated calls
    are free.
    """
    env_prefix = "/".join(dataset_id.split("/")[:2])  # e.g. "mujoco/halfcheetah"

    if env_prefix in _reference_score_cache:
        return _reference_score_cache[env_prefix]

    # Expert score: mean episode return from the expert dataset
    expert_ds = minari.load_dataset(f"{env_prefix}/expert-v0", download=True)
    expert_returns = [float(ep.rewards.sum()) for ep in expert_ds.iterate_episodes()]
    expert_ref = mean(expert_returns)

    # Random score: roll out a random policy in the environment
    env = expert_ds.recover_environment()
    try:
        random_policy = lambda obs: env.action_space.sample()  # noqa: E731
        random_returns = rollout_policy(random_policy, env, num_episodes=100, seed=0)
    finally:
        env.close()
    random_ref = mean(random_returns)

    _reference_score_cache[env_prefix] = (random_ref, expert_ref)
    return random_ref, expert_ref


def compute_normalized_score(dataset_id: str, raw_score: float) -> float:
    """Normalise *raw_score* to [0, 100] using random/expert reference returns."""
    random_ref, expert_ref = get_reference_scores(dataset_id)
    return (raw_score - random_ref) / (expert_ref - random_ref) * 100.0


# ---------------------------------------------------------------------------
# Vectorized evaluation (batched policy, gymnasium.make_vec)
# ---------------------------------------------------------------------------

_env_id_cache: dict[str, str] = {}


def get_env_id(dataset_id: str) -> str:
    """Derive the Gymnasium environment ID from a Minari dataset ID."""
    if dataset_id in _env_id_cache:
        return _env_id_cache[dataset_id]
    ds = minari.load_dataset(dataset_id)
    env = ds.recover_environment()
    env_id = env.spec.id  # type: ignore[union-attr]
    env.close()
    _env_id_cache[dataset_id] = env_id
    return env_id


def evaluate_policy_vectorized(
    batched_policy: Callable[[np.ndarray], np.ndarray],
    env_id: str,
    num_episodes: int,
    seed: int,
    num_envs: int = 8,
) -> list[float]:
    """Evaluate a batched policy using vectorised environments.

    Args:
        batched_policy: Maps ``(num_envs, obs_dim)`` obs → ``(num_envs, act_dim)`` actions.
        env_id: Gymnasium environment ID (use :func:`get_env_id` to derive from dataset ID).
        num_episodes: Number of complete episodes to collect.
        seed: Base seed passed to ``envs.reset()``.
        num_envs: Number of parallel environments.

    Returns:
        List of per-episode undiscounted returns (length == num_episodes).
    """
    envs = gymnasium.make_vec(env_id, num_envs=num_envs, vectorization_mode="sync")
    obs, _ = envs.reset(seed=seed)
    ep_returns = np.zeros(num_envs)
    returns: list[float] = []
    try:
        while len(returns) < num_episodes:
            actions = np.asarray(batched_policy(obs))
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            ep_returns += rewards
            dones = terminated | truncated
            for i in range(num_envs):
                if dones[i]:
                    returns.append(float(ep_returns[i]))
                    ep_returns[i] = 0.0
    finally:
        envs.close()
    return returns[:num_episodes]
