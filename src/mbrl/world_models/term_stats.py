"""Precomputed statistics for MoReL's halt-state (USAD) penalty.

MoReL (Kidambi et al., 2020) replaces uncertain model transitions with a halt
state: when the elite ensemble disagrees too much at a state-action, the
synthetic reward is overwritten with ``min_r + term_penalty_offset`` and the
rollout terminates. The disagreement threshold is calibrated once, on the offline
data, by :func:`compute_model_discrepancy`.

Faithful port of Unifloral's ``compute_model_discrepancy`` (algorithms/dynamics.py),
driven through the world model's ``predict_ensemble`` so it works for any
``EnsembleDynamics`` subclass. As in Unifloral, ``discrepancy`` is the largest L2
distance between any two elite mean-predictions taken over the dataset — across
different elites *and* different points — i.e. the diameter of the elite-prediction
cloud. That diameter is dominated by how far predictions spread across the visited
state space, so it is a deliberately global scale for the per-state disagreement
measured at rollout time, not a per-state quantity; ``threshold_coef`` then tunes
how readily the halt fires.

The computation is O(N^2) in the number of sampled points, so it is run once at the
end of world-model training (opt-in) and cached in the checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from mbrl.data import Transition

if TYPE_CHECKING:
    from mbrl.world_models.base import EnsembleDynamics


def compute_model_discrepancy(
    world_model: "EnsembleDynamics",
    dataset: Transition,
    rng: jax.Array,
    block_size: int = 256,
    max_samples: int = 500_000,
) -> float:
    """Max L2 distance between elite mean-predictions over the dataset.

    The max ranges over every (elite, point) pair — different elites and different
    points — so it measures the diameter of the elite-prediction cloud, used as the
    calibration scale for MoReL's halt threshold. Returns a Python float.

    Args:
        world_model: Trained ensemble exposing ``predict_ensemble`` (elite means).
        dataset: Offline dataset; predictions are evaluated on its (obs, action).
        rng: Key for the random subsample of points (keeps the estimate deterministic).
        block_size: Tile size for the pairwise computation. Affects memory/speed only
            — the result is invariant to it (an exact max over all sampled pairs).
        max_samples: Cap on points used (Unifloral's budget is ~500k); larger datasets
            are subsampled to this many points.

    The pairwise distances are computed one ``(block, block)`` tile at a time, so peak
    memory is bounded by a single tile rather than the full (N, N) matrix.
    """
    n = int(dataset.obs.shape[0])
    block = min(int(block_size), n)
    num_blocks = min(int(max_samples), n) // block
    num_samples = num_blocks * block

    # Subsample (drops the tail, as in Unifloral) then tile into whole blocks.
    perm = jax.random.permutation(rng, n)[:num_samples]
    obs = dataset.obs[perm].reshape(num_blocks, block, -1)
    action = dataset.action[perm].reshape(num_blocks, block, -1)

    # Elite mean predictions per point, batched. lax.map is sequential, so the
    # forward passes do not all materialise at once.
    def _block_means(batch: tuple[jax.Array, jax.Array]) -> jax.Array:
        o, a = batch
        means, _ = jax.vmap(world_model.predict_ensemble)(o, a)  # (block, E, D)
        return means

    means = jax.lax.map(_block_means, (obs, action))  # (num_blocks, block, E, D)
    num_elites, output_dim = means.shape[2], means.shape[3]
    # (E, num_samples, D): elites on the leading axis, matching Unifloral's layout.
    elite_samples = means.reshape(num_samples, num_elites, output_dim).transpose(1, 0, 2)

    @jax.jit
    def _diameter(samples: jax.Array) -> jax.Array:
        def _tile(i: jax.Array) -> jax.Array:
            return jax.lax.dynamic_slice(
                samples, (0, i * block, 0), (num_elites, block, output_dim)
            )

        def _outer(i: jax.Array, running_max: jax.Array) -> jax.Array:
            tile_i = _tile(i)  # (E, B, D)

            def _inner(j: jax.Array, rm: jax.Array) -> jax.Array:
                tile_j = _tile(j)  # (E, B, D)
                # (E,1,B,1,D) - (1,E,1,B,D) -> (E,E,B,B,D); squared L2 over features.
                diff = tile_i[:, None, :, None, :] - tile_j[None, :, None, :, :]
                sq = jnp.sum(diff**2, axis=-1)  # (E, E, B, B)
                return jnp.maximum(rm, jnp.max(sq))

            return jax.lax.fori_loop(0, num_blocks, _inner, running_max)

        max_sq = jax.lax.fori_loop(0, num_blocks, _outer, jnp.array(0.0, jnp.float32))
        return jnp.sqrt(max_sq)

    return float(_diameter(elite_samples))
