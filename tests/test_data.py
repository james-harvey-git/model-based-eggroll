"""Tests for Minari data loading and preprocessing."""

import jax
import jax.numpy as jnp
import pytest

from mbrl.data import (
    DatasetInfo,
    Transition,
    create_epoch_iterator,
    load_dataset,
    sample_batch,
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
