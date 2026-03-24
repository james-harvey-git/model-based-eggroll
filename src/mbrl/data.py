"""Minari data loading, preprocessing, and batched sampling."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import minari
import numpy as np


class Transition(NamedTuple):
    """A flat buffer of (s, a, r, s', done) transitions."""

    obs: jnp.ndarray  # (N, obs_dim)
    action: jnp.ndarray  # (N, act_dim)
    reward: jnp.ndarray  # (N,)
    next_obs: jnp.ndarray  # (N, obs_dim)
    done: jnp.ndarray  # (N,)


class DatasetInfo(NamedTuple):
    """Normalization statistics and metadata for a loaded dataset."""

    obs_mean: jnp.ndarray  # (obs_dim,)
    obs_std: jnp.ndarray  # (obs_dim,)
    obs_dim: int
    act_dim: int
    dataset_id: str


def load_dataset(
    dataset_id: str, download: bool = True
) -> tuple[Transition, DatasetInfo]:
    """Load a Minari dataset and return flat transitions with normalization stats.

    Args:
        dataset_id: Minari dataset ID, e.g. "mujoco/halfcheetah/medium-v0".
        download: Whether to download the dataset if not cached locally.

    Returns:
        A (Transition, DatasetInfo) tuple.
    """
    ds = minari.load_dataset(dataset_id, download=download)

    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []

    for episode in ds.iterate_episodes():
        # observations has T+1 timesteps (includes initial obs)
        all_obs.append(episode.observations[:-1])
        all_next_obs.append(episode.observations[1:])
        all_actions.append(episode.actions)
        all_rewards.append(episode.rewards)
        # Use terminations only (not truncations), matching D4RL convention
        all_dones.append(episode.terminations)

    obs = jnp.asarray(np.concatenate(all_obs), dtype=jnp.float32)
    actions = jnp.asarray(np.concatenate(all_actions), dtype=jnp.float32)
    rewards = jnp.asarray(np.concatenate(all_rewards), dtype=jnp.float32)
    next_obs = jnp.asarray(np.concatenate(all_next_obs), dtype=jnp.float32)
    dones = jnp.asarray(np.concatenate(all_dones), dtype=jnp.float32)

    dataset = Transition(
        obs=obs, action=actions, reward=rewards, next_obs=next_obs, done=dones
    )

    obs_mean = obs.mean(axis=0)
    obs_std = jnp.nan_to_num(obs.std(axis=0), nan=1.0)

    info = DatasetInfo(
        obs_mean=obs_mean,
        obs_std=obs_std,
        obs_dim=int(obs.shape[1]),
        act_dim=int(actions.shape[1]),
        dataset_id=dataset_id,
    )

    return dataset, info


def sample_batch(dataset: Transition, batch_size: int, rng: jax.Array) -> Transition:
    """Sample a random batch of transitions (with replacement)."""
    n = dataset.obs.shape[0]
    idxs = jax.random.randint(rng, (batch_size,), 0, n)
    return jax.tree.map(lambda x: x[idxs], dataset)


def create_epoch_iterator(data, batch_size: int, rng: jax.Array):
    """Shuffle a pytree of arrays and reshape into (num_batches, batch_size, ...) for scanning.

    Works with any pytree (Transition, tuple of arrays, etc.) where all leaves share
    the same leading dimension. Drops trailing elements that don't fill a complete batch.
    """
    n = jax.tree.leaves(data)[0].shape[0]
    perm = jax.random.permutation(rng, n)
    shuffled = jax.tree.map(lambda x: x[perm], data)
    num_batches = n // batch_size
    iter_size = num_batches * batch_size
    return jax.tree.map(
        lambda x: x[:iter_size].reshape(num_batches, batch_size, *x.shape[1:]),
        shuffled,
    )


def train_val_split(
    dataset: Transition, val_fraction: float, rng: jax.Array
) -> tuple[Transition, Transition]:
    """Split dataset into train and validation sets via random permutation."""
    n = dataset.obs.shape[0]
    train_size = int((1 - val_fraction) * n)
    perm = jax.random.permutation(rng, n)
    train_idxs, val_idxs = perm[:train_size], perm[train_size:]
    train = jax.tree.map(lambda x: x[train_idxs], dataset)
    val = jax.tree.map(lambda x: x[val_idxs], dataset)
    return train, val
