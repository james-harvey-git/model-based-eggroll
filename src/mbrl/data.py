"""Minari data loading, preprocessing, and batched sampling."""

import math
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


def load_dataset(dataset_id: str, download: bool = True) -> tuple[Transition, DatasetInfo]:
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

    dataset = Transition(obs=obs, action=actions, reward=rewards, next_obs=next_obs, done=dones)

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


def derive_train_val_split(
    dataset: Transition, val_fraction: float, seed: int
) -> tuple[Transition, Transition]:
    """Deterministic train/val split keyed only on (dataset, val_fraction, seed).

    The partition is a pure function of the seed, decoupled from any training rng,
    so a fine-tune run reproduces a checkpoint's exact held-out val set by passing the
    checkpoint's seed + val_fraction. Both world-model trainers use this for val parity.
    """
    return train_val_split(dataset, val_fraction, jax.random.key(int(seed)))


# ── Trajectory (episode-structured) data for EGGROLL trajectory fine-tuning ──────
#
# The flat `Transition` buffer above discards episode boundaries, which the
# trajectory fine-tuning phase (issue #42) needs: it unrolls the world model
# open-loop over contiguous real trajectories. `EpisodeBatch` keeps episodes as
# padded, fixed-width rows so the horizon curriculum can re-tile them at any T;
# `tile_episodes_to_windows` is the pure (no-I/O) tiling step, unit-testable on
# hand-built episodes without a download.


class EpisodeBatch(NamedTuple):
    """Per-episode arrays, padded to the longest episode (``Lmax`` transitions).

    ``obs`` keeps the ``Lmax + 1`` length (initial state plus every next-state) so a
    window can read both its start state and all its real next-state targets. Padded
    timesteps are zero and flagged invalid by ``step_mask``. Padding to ``Lmax`` assumes
    roughly-uniform episode lengths (true for D4RL MuJoCo; memory-wasteful for highly
    variable-length datasets).
    """

    obs: jnp.ndarray  # (E, Lmax + 1, obs_dim)
    actions: jnp.ndarray  # (E, Lmax, act_dim)
    rewards: jnp.ndarray  # (E, Lmax)
    step_mask: jnp.ndarray  # (E, Lmax) — 1 for real transitions, 0 for padding
    ep_len: jnp.ndarray  # (E,) int32 — number of real transitions per episode


class TrajectoryWindows(NamedTuple):
    """Fixed-length-``T`` rollout windows tiled from episodes (the trainer samples these).

    Obs targets are *absolute* real next-states (no ``obs_std`` normalisation, matching
    the one-step trainers). ``mask`` flags valid steps (0 for padded / post-terminal).
    """

    start_obs: jnp.ndarray  # (W, obs_dim)        — o_0 of each window
    actions: jnp.ndarray  # (W, T, act_dim)     — a_0 .. a_{T-1}
    target_obs: jnp.ndarray  # (W, T, obs_dim)     — real o_1 .. o_T (absolute)
    target_reward: jnp.ndarray  # (W, T)              — real r_0 .. r_{T-1}
    mask: jnp.ndarray  # (W, T)              — 1 valid, 0 padded


def load_episodes(
    dataset_id: str, download: bool = True
) -> tuple[EpisodeBatch, DatasetInfo]:
    """Load a Minari dataset preserving per-episode structure (padded to ``Lmax``).

    Mirrors :func:`load_dataset`'s minari iteration and normalisation stats, but keeps
    episodes separate so the trajectory fine-tuner can unroll contiguous rollouts and
    re-tile them per horizon. Stats are computed over the same transition obs as
    :func:`load_dataset` so a flat/episodic load of the same dataset agree.
    """
    ds = minari.load_dataset(dataset_id, download=download)

    obs_list, act_list, rew_list, len_list = [], [], [], []
    for episode in ds.iterate_episodes():
        o = np.asarray(episode.observations, dtype=np.float32)  # (L+1, obs_dim)
        a = np.asarray(episode.actions, dtype=np.float32)  # (L, act_dim)
        r = np.asarray(episode.rewards, dtype=np.float32)  # (L,)
        obs_list.append(o)
        act_list.append(a)
        rew_list.append(r)
        len_list.append(int(a.shape[0]))

    n_ep = len(len_list)
    lmax = max(len_list)
    obs_dim = obs_list[0].shape[1]
    act_dim = act_list[0].shape[1]

    obs_pad = np.zeros((n_ep, lmax + 1, obs_dim), np.float32)
    act_pad = np.zeros((n_ep, lmax, act_dim), np.float32)
    rew_pad = np.zeros((n_ep, lmax), np.float32)
    mask = np.zeros((n_ep, lmax), np.float32)
    for i, (o, a, r, length) in enumerate(zip(obs_list, act_list, rew_list, len_list)):
        obs_pad[i, : length + 1] = o
        act_pad[i, :length] = a
        rew_pad[i, :length] = r
        mask[i, :length] = 1.0

    episodes = EpisodeBatch(
        obs=jnp.asarray(obs_pad),
        actions=jnp.asarray(act_pad),
        rewards=jnp.asarray(rew_pad),
        step_mask=jnp.asarray(mask),
        ep_len=jnp.asarray(len_list, dtype=jnp.int32),
    )

    all_obs = np.concatenate([o[:-1] for o in obs_list])
    obs_mean = jnp.asarray(all_obs.mean(axis=0))
    obs_std = jnp.nan_to_num(jnp.asarray(all_obs.std(axis=0)), nan=1.0)
    info = DatasetInfo(
        obs_mean=obs_mean,
        obs_std=obs_std,
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
        dataset_id=dataset_id,
    )
    return episodes, info


def tile_episodes_to_windows(episodes: EpisodeBatch, horizon: int) -> TrajectoryWindows:
    """Tile each episode into non-overlapping length-``horizon`` windows (pure, no I/O).

    Tiling is *within* an episode (never across the padded boundary). Each episode yields
    ``ceil(Lmax / T)`` windows; windows that fall entirely in an episode's padding (for
    episodes shorter than ``Lmax``) are dropped, so every returned window has at least one
    valid step and cannot dilute the mean-over-batch fitness. The trailing short window of
    an episode keeps its real steps and masks the rest.
    """
    t = int(horizon)
    assert t >= 1, f"horizon must be >= 1, got {t}"
    obs = np.asarray(episodes.obs)
    actions = np.asarray(episodes.actions)
    rewards = np.asarray(episodes.rewards)
    step_mask = np.asarray(episodes.step_mask)

    n_ep, lmax = actions.shape[0], actions.shape[1]
    n_win = math.ceil(lmax / t)
    lpad = n_win * t
    pad_t = lpad - lmax
    if pad_t:
        obs = np.pad(obs, ((0, 0), (0, pad_t), (0, 0)))
        actions = np.pad(actions, ((0, 0), (0, pad_t), (0, 0)))
        rewards = np.pad(rewards, ((0, 0), (0, pad_t)))
        step_mask = np.pad(step_mask, ((0, 0), (0, pad_t)))

    obs_dim = obs.shape[-1]
    start_obs = obs[:, :lpad:t, :].reshape(n_ep * n_win, obs_dim)
    target_obs = obs[:, 1 : lpad + 1, :].reshape(n_ep * n_win, t, obs_dim)
    actions_w = actions.reshape(n_ep * n_win, t, actions.shape[-1])
    rewards_w = rewards.reshape(n_ep * n_win, t)
    mask_w = step_mask.reshape(n_ep * n_win, t)

    keep = mask_w.sum(axis=1) > 0
    return TrajectoryWindows(
        start_obs=jnp.asarray(start_obs[keep]),
        actions=jnp.asarray(actions_w[keep]),
        target_obs=jnp.asarray(target_obs[keep]),
        target_reward=jnp.asarray(rewards_w[keep]),
        mask=jnp.asarray(mask_w[keep].astype(np.float32)),
    )


def derive_episode_train_val_split(
    episodes: EpisodeBatch, val_fraction: float, seed: int
) -> tuple[EpisodeBatch, EpisodeBatch]:
    """Deterministic split over *whole episodes*, keyed only on (episodes, fraction, seed).

    Episode-level (not transition-level) so windows from one episode never straddle the
    train/val boundary. Pure in the seed, so a fine-tune reproduces the same held-out
    episodes by passing the checkpoint's seed + fraction (mirrors :func:`derive_train_val_split`).
    """
    n_ep = episodes.actions.shape[0]
    perm = jax.random.permutation(jax.random.key(int(seed)), n_ep)
    n_train = int((1 - val_fraction) * n_ep)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train = jax.tree.map(lambda x: x[train_idx], episodes)
    val = jax.tree.map(lambda x: x[val_idx], episodes)
    return train, val
