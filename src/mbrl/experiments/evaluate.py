"""Evaluation experiment: roll out a policy in the real environment and log the score."""

from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from mbrl.evaluation import compute_normalized_score, evaluate_policy_vectorized, get_env_id
from mbrl.logger import Logger
from mbrl.policy_optimizers.sac_n import TanhGaussianActor


def run(cfg: DictConfig, logger: Logger) -> None:
    """Load a policy checkpoint, evaluate stochastically in the real environment, and log."""
    checkpoint_path = Path(cfg.policy_checkpoint_dir) / "policy.pkl"
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    actor_net = TanhGaussianActor(ckpt["act_dim"])
    actor_params = ckpt["actor_params"]

    @jax.jit
    def _batched_sample(rng: jax.Array, obs_batch: jax.Array) -> jax.Array:
        rngs = jax.random.split(rng, obs_batch.shape[0])
        return jax.vmap(
            lambda r, o: jnp.nan_to_num(actor_net.apply(actor_params, o).sample(seed=r))  # type: ignore[union-attr]
        )(rngs, obs_batch)

    eval_rng = jax.random.key(cfg.seed)

    def batched_policy(obs_batch: np.ndarray) -> np.ndarray:
        nonlocal eval_rng
        eval_rng, rng_step = jax.random.split(eval_rng)
        return np.asarray(_batched_sample(rng_step, jnp.array(obs_batch)))

    env_id = get_env_id(ckpt["dataset_id"])
    num_envs = int(cfg.eval.eval_workers)
    num_episodes = int(cfg.eval.num_episodes)

    eval_returns = evaluate_policy_vectorized(
        batched_policy, env_id, num_episodes, seed=cfg.seed, num_envs=num_envs
    )
    raw_score = float(np.mean(eval_returns))
    normalized = compute_normalized_score(ckpt["dataset_id"], raw_score)

    print(f"Evaluation complete: raw={raw_score:.2f}  normalized={normalized:.2f}")
    logger.log_eval(ckpt["dataset_id"], raw_score, normalized)
