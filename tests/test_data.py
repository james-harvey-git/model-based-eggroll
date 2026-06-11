"""Tests for Minari data loading and preprocessing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mbrl.data import (
    DatasetInfo,
    EpisodeBatch,
    Transition,
    create_epoch_iterator,
    derive_episode_train_val_split,
    derive_train_val_split,
    load_dataset,
    sample_batch,
    tile_episodes_to_windows,
    train_val_split,
)

DATASET_ID = "mujoco/halfcheetah/medium-v0"


@pytest.fixture(scope="module")
def dataset_and_info():
    """Load the dataset once for all tests in this module."""
    return load_dataset(DATASET_ID)


class TestLoadDataset:
    def test_returns_correct_types(self, dataset_and_info):
        dataset, info = dataset_and_info
        assert isinstance(dataset, Transition)
        assert isinstance(info, DatasetInfo)

    def test_all_fields_float32(self, dataset_and_info):
        dataset, _ = dataset_and_info
        for field in dataset:
            assert field.dtype == jnp.float32

    def test_shapes(self, dataset_and_info):
        dataset, info = dataset_and_info
        n = dataset.obs.shape[0]
        assert dataset.obs.shape == (n, 17)
        assert dataset.next_obs.shape == (n, 17)
        assert dataset.action.shape == (n, 6)
        assert dataset.reward.shape == (n,)
        assert dataset.done.shape == (n,)
        assert info.obs_dim == 17
        assert info.act_dim == 6

    def test_info_stats(self, dataset_and_info):
        _, info = dataset_and_info
        assert info.obs_mean.shape == (17,)
        assert info.obs_std.shape == (17,)
        assert not jnp.any(jnp.isnan(info.obs_std))
        assert info.dataset_id == DATASET_ID


class TestSampleBatch:
    def test_shape(self, dataset_and_info):
        dataset, _ = dataset_and_info
        batch = sample_batch(dataset, 256, jax.random.key(0))
        assert batch.obs.shape == (256, 17)
        assert batch.action.shape == (256, 6)
        assert batch.reward.shape == (256,)

    def test_different_keys_give_different_batches(self, dataset_and_info):
        dataset, _ = dataset_and_info
        b1 = sample_batch(dataset, 64, jax.random.key(0))
        b2 = sample_batch(dataset, 64, jax.random.key(1))
        assert not jnp.array_equal(b1.obs, b2.obs)


class TestCreateEpochIterator:
    def test_shape(self, dataset_and_info):
        dataset, _ = dataset_and_info
        batch_size = 256
        batched = create_epoch_iterator(dataset, batch_size, jax.random.key(0))
        n = dataset.obs.shape[0]
        expected_num_batches = n // batch_size
        assert batched.obs.shape == (expected_num_batches, batch_size, 17)
        assert batched.action.shape == (expected_num_batches, batch_size, 6)
        assert batched.reward.shape == (expected_num_batches, batch_size)

    def test_covers_correct_count(self, dataset_and_info):
        dataset, _ = dataset_and_info
        batch_size = 256
        batched = create_epoch_iterator(dataset, batch_size, jax.random.key(0))
        n = dataset.obs.shape[0]
        expected_num_batches = n // batch_size
        assert batched.obs.shape[0] * batched.obs.shape[1] == expected_num_batches * batch_size


class TestTrainValSplit:
    def test_sizes_sum(self, dataset_and_info):
        dataset, _ = dataset_and_info
        train, val = train_val_split(dataset, 0.2, jax.random.key(0))
        assert train.obs.shape[0] + val.obs.shape[0] == dataset.obs.shape[0]

    def test_deterministic(self, dataset_and_info):
        dataset, _ = dataset_and_info
        t1, v1 = train_val_split(dataset, 0.2, jax.random.key(42))
        t2, v2 = train_val_split(dataset, 0.2, jax.random.key(42))
        assert jnp.array_equal(t1.obs, t2.obs)
        assert jnp.array_equal(v1.obs, v2.obs)

    def test_different_keys_give_different_splits(self, dataset_and_info):
        dataset, _ = dataset_and_info
        t1, _ = train_val_split(dataset, 0.2, jax.random.key(0))
        t2, _ = train_val_split(dataset, 0.2, jax.random.key(1))
        assert not jnp.array_equal(t1.obs, t2.obs)


class TestDeriveTrainValSplit:
    def test_sizes_sum_and_disjoint(self, dataset_and_info):
        dataset, _ = dataset_and_info
        train, val = derive_train_val_split(dataset, 0.2, seed=0)
        n = dataset.obs.shape[0]
        assert train.obs.shape[0] + val.obs.shape[0] == n
        assert val.obs.shape[0] == n - int(0.8 * n)

    def test_deterministic_in_seed(self, dataset_and_info):
        dataset, _ = dataset_and_info
        t1, v1 = derive_train_val_split(dataset, 0.2, seed=7)
        t2, v2 = derive_train_val_split(dataset, 0.2, seed=7)
        assert jnp.array_equal(t1.obs, t2.obs)
        assert jnp.array_equal(v1.obs, v2.obs)

    def test_different_seeds_give_different_splits(self, dataset_and_info):
        dataset, _ = dataset_and_info
        _, v0 = derive_train_val_split(dataset, 0.2, seed=0)
        _, v1 = derive_train_val_split(dataset, 0.2, seed=1)
        assert not jnp.array_equal(v0.obs, v1.obs)

    def test_matches_train_val_split_on_key_seed(self, dataset_and_info):
        # Pure function of seed: equivalent to train_val_split with key(seed).
        dataset, _ = dataset_and_info
        t_ref, v_ref = train_val_split(dataset, 0.2, jax.random.key(3))
        t, v = derive_train_val_split(dataset, 0.2, seed=3)
        assert jnp.array_equal(t.obs, t_ref.obs)
        assert jnp.array_equal(v.obs, v_ref.obs)


def _toy_episodes(lengths, obs_dim=3, act_dim=2) -> EpisodeBatch:
    """Hand-built EpisodeBatch (no download). obs[e,k] is filled with e*100+k so window
    alignment is verifiable; padded steps are zero and masked."""
    n_ep, lmax = len(lengths), max(lengths)
    obs = np.zeros((n_ep, lmax + 1, obs_dim), np.float32)
    act = np.zeros((n_ep, lmax, act_dim), np.float32)
    rew = np.zeros((n_ep, lmax), np.float32)
    mask = np.zeros((n_ep, lmax), np.float32)
    for e, length in enumerate(lengths):
        for k in range(length + 1):
            obs[e, k] = e * 100 + k
        for k in range(length):
            act[e, k] = e * 100 + k
            rew[e, k] = e * 100 + k
            mask[e, k] = 1.0
    return EpisodeBatch(
        jnp.asarray(obs), jnp.asarray(act), jnp.asarray(rew), jnp.asarray(mask),
        jnp.asarray(lengths, jnp.int32),
    )


class TestTileEpisodesToWindows:
    def test_alignment_and_targets(self):
        # One episode of length 4, T=2 -> 2 windows: steps [0,1] and [2,3].
        w = tile_episodes_to_windows(_toy_episodes([4]), horizon=2)
        assert w.start_obs.shape[0] == 2
        # Window 0: start o0, targets o1,o2; window 1: start o2, targets o3,o4.
        assert float(w.start_obs[0, 0]) == 0.0 and float(w.start_obs[1, 0]) == 2.0
        assert [float(x) for x in w.target_obs[0, :, 0]] == [1.0, 2.0]
        assert [float(x) for x in w.target_obs[1, :, 0]] == [3.0, 4.0]
        assert jnp.all(w.mask == 1.0)

    def test_short_remainder_masked(self):
        # Length 5, T=2 -> 3 windows; last is [4, pad] with mask [1, 0].
        w = tile_episodes_to_windows(_toy_episodes([5]), horizon=2)
        assert w.start_obs.shape[0] == 3
        assert [float(x) for x in w.mask[-1]] == [1.0, 0.0]

    def test_all_masked_windows_dropped(self):
        # Lmax=5 (-> 3 windows/episode at T=2); the length-3 episode's 3rd window is
        # all-padding and must be dropped. 3 + 2 = 5 windows total.
        w = tile_episodes_to_windows(_toy_episodes([5, 3]), horizon=2)
        assert w.start_obs.shape[0] == 5
        assert jnp.all(w.mask.sum(axis=1) > 0)  # no all-masked window survives

    def test_windows_stay_within_episode(self):
        # No window mixes states from two episodes (no cross-seam tiling).
        w = tile_episodes_to_windows(_toy_episodes([4, 4]), horizon=2)
        for i in range(w.start_obs.shape[0]):
            ep = int(w.start_obs[i, 0]) // 100  # episode id encoded in the value
            assert all(int(v) // 100 == ep for v in w.target_obs[i, :, 0] if v != 0)


class TestDeriveEpisodeTrainValSplit:
    def test_sizes_sum_and_deterministic(self):
        eps = _toy_episodes([4] * 10)
        tr, va = derive_episode_train_val_split(eps, 0.2, seed=0)
        assert tr.actions.shape[0] + va.actions.shape[0] == 10
        assert va.actions.shape[0] == 10 - int(0.8 * 10)
        tr2, va2 = derive_episode_train_val_split(eps, 0.2, seed=0)
        assert jnp.array_equal(tr.actions, tr2.actions)
        assert jnp.array_equal(va.actions, va2.actions)

    def test_different_seeds_differ(self):
        eps = _toy_episodes([4] * 10)
        _, v0 = derive_episode_train_val_split(eps, 0.2, seed=0)
        _, v1 = derive_episode_train_val_split(eps, 0.2, seed=1)
        assert not jnp.array_equal(v0.actions, v1.actions)
