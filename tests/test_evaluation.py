"""Tests for the evaluation pipeline."""

import math
from unittest.mock import patch

import gymnasium
import minari
import numpy as np
import pytest

from mbrl.evaluation import (
    _reference_score_cache,
    compute_normalized_score,
    evaluate_policy,
    get_reference_scores,
    rollout_policy,
)

DATASET_ID = "mujoco/halfcheetah/medium-v0"


@pytest.fixture(scope="module")
def halfcheetah_env():
    """Create a HalfCheetah env from the Minari dataset metadata."""
    ds = minari.load_dataset(DATASET_ID)
    env = ds.recover_environment()
    yield env
    env.close()


class TestRolloutPolicy:
    def test_returns_correct_count(self, halfcheetah_env):
        policy = lambda obs: halfcheetah_env.action_space.sample()
        returns = rollout_policy(policy, halfcheetah_env, num_episodes=3, seed=0)
        assert len(returns) == 3

    def test_returns_are_finite(self, halfcheetah_env):
        policy = lambda obs: halfcheetah_env.action_space.sample()
        returns = rollout_policy(policy, halfcheetah_env, num_episodes=3, seed=0)
        assert all(math.isfinite(r) for r in returns)


class TestEvaluatePolicy:
    def test_returns_finite_float(self):
        score = evaluate_policy(
            lambda obs: np.zeros(6), DATASET_ID, num_episodes=3, seed=0
        )
        assert isinstance(score, float)
        assert math.isfinite(score)


class TestGetReferenceScores:
    def test_expert_greater_than_random(self):
        random_ref, expert_ref = get_reference_scores(DATASET_ID)
        assert expert_ref > random_ref

    def test_both_finite(self):
        random_ref, expert_ref = get_reference_scores(DATASET_ID)
        assert math.isfinite(random_ref)
        assert math.isfinite(expert_ref)

    def test_expert_sanity_check(self):
        _, expert_ref = get_reference_scores(DATASET_ID)
        assert expert_ref > 10000  # HalfCheetah expert should be well above 10k

    def test_caching(self):
        # Ensure it's cached from a prior call
        get_reference_scores(DATASET_ID)
        env_prefix = "mujoco/halfcheetah"
        assert env_prefix in _reference_score_cache

        # Patch minari.load_dataset to raise if called — proves the cache is used
        with patch.object(
            minari, "load_dataset", side_effect=RuntimeError("should not be called")
        ):
            random_ref, expert_ref = get_reference_scores(DATASET_ID)
            assert math.isfinite(random_ref)
            assert math.isfinite(expert_ref)


class TestComputeNormalizedScore:
    def test_expert_score_near_100(self):
        _, expert_ref = get_reference_scores(DATASET_ID)
        normed = compute_normalized_score(DATASET_ID, expert_ref)
        assert abs(normed - 100.0) < 1e-6

    def test_random_score_near_0(self):
        random_ref, _ = get_reference_scores(DATASET_ID)
        normed = compute_normalized_score(DATASET_ID, random_ref)
        assert abs(normed) < 1e-6
