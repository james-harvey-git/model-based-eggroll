"""Policy training experiment."""

from datetime import datetime
import importlib
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mbrl.data import load_dataset
from mbrl.evaluation import compute_normalized_score, evaluate_policy_vectorized, get_env_id
from mbrl.logger import Logger, _algorithm_type
from mbrl.policy_optimizers.sac_n import TanhGaussianActor
from mbrl.world_models.mle import MLEEnsemble


def run(cfg: DictConfig, logger: Logger) -> None:
    """Load a world model checkpoint, train a policy with chunked eval, and save a checkpoint.

    Dispatches to the configured policy optimizer via convention functions:
      - ``make_train_step(world_model, dataset, cfg, rng) -> (step_fn, runner_state)``
      - ``extract_actor(runner_state) -> (actor_params, step)``

    Checkpoints loaded from / saved to cfg.checkpoint_dir.
    """
    rng = jax.random.key(cfg.seed)

    # Load dataset and world model checkpoint
    dataset, info = load_dataset(cfg.dataset.name)
    wm_checkpoint_path = Path(cfg.checkpoint_dir) / "world_model.pkl"
    with open(wm_checkpoint_path, "rb") as f:
        wm_ckpt = pickle.load(f)
    world_model = MLEEnsemble.load_from_checkpoint(wm_checkpoint_path)

    # Resolve the policy optimizer module via importlib convention
    target: str = cfg.policy_optimizer._target_  # e.g. "mbrl.policy_optimizers.mopo.train"
    module_path = target.rsplit(".", 1)[0]  # e.g. "mbrl.policy_optimizers.mopo"
    pol_module = importlib.import_module(module_path)

    rng, train_rng = jax.random.split(rng)
    step_fn, runner_state = pol_module.make_train_step(
        world_model, dataset, cfg.policy_optimizer, train_rng
    )

    # Build JIT-compiled batched sample function once (JIT caches by function identity).
    # actor_params passed as explicit arg so JIT compiles once for the lifetime of run().
    actor_net = TanhGaussianActor(info.act_dim)

    @jax.jit
    def _batched_sample(
        rng: jax.Array, params: dict, obs_batch: jax.Array
    ) -> jax.Array:
        rngs = jax.random.split(rng, obs_batch.shape[0])
        return jax.vmap(
            lambda r, o: jnp.nan_to_num(actor_net.apply(params, o).sample(seed=r))  # type: ignore[union-attr]
        )(rngs, obs_batch)

    # Derive env ID for vectorized evaluation
    env_id = get_env_id(info.dataset_id)
    eval_interval = int(cfg.policy_optimizer.eval_interval)
    num_updates = int(cfg.policy_optimizer.num_policy_updates)
    num_envs = int(cfg.eval.eval_workers)
    num_episodes = int(cfg.eval.num_episodes)
    num_chunks = num_updates // eval_interval
    remainder = num_updates % eval_interval

    def _eval_and_log(runner_state: tuple, chunk_label: str) -> None:
        actor_params, step = pol_module.extract_actor(runner_state)
        eval_rng = jax.random.key(cfg.seed + step)

        def batched_policy(obs_batch: np.ndarray) -> np.ndarray:
            nonlocal eval_rng
            eval_rng, rng_step = jax.random.split(eval_rng)
            return np.asarray(_batched_sample(rng_step, actor_params, jnp.array(obs_batch)))

        eval_returns = evaluate_policy_vectorized(
            batched_policy, env_id, num_episodes, seed=cfg.seed, num_envs=num_envs
        )
        raw_score = float(np.mean(eval_returns))
        normalized = compute_normalized_score(info.dataset_id, raw_score)
        logger.log_policy_step(step, raw_score=raw_score, normalized_score=normalized)
        print(
            f"[{chunk_label}] step={step}  raw={raw_score:.1f}  "
            f"normalized={normalized:.1f}"
        )

    # Chunked training loop with interleaved evaluation
    for chunk in range(num_chunks):
        runner_state, metrics = jax.lax.scan(step_fn, runner_state, None, length=eval_interval)
        _, step = pol_module.extract_actor(runner_state)
        logger.log_policy_step(
            step, **{k: float(v.mean()) for k, v in metrics.items()}
        )
        _eval_and_log(runner_state, f"chunk {chunk + 1}/{num_chunks}")

    # Handle remainder steps (if num_updates % eval_interval != 0)
    if remainder > 0:
        runner_state, metrics = jax.lax.scan(step_fn, runner_state, None, length=remainder)
        _, step = pol_module.extract_actor(runner_state)
        logger.log_policy_step(
            step, **{k: float(v.mean()) for k, v in metrics.items()}
        )
        _eval_and_log(runner_state, "remainder")

    # Save policy checkpoint
    actor_params, step = pol_module.extract_actor(runner_state)
    checkpoint = {
        "actor_params": actor_params,
        "obs_dim": info.obs_dim,
        "act_dim": info.act_dim,
        "dataset_id": info.dataset_id,
        "policy_optimizer_cfg": OmegaConf.to_container(cfg.policy_optimizer),
        "wm_group": logger.wm_group,
        "seed": int(cfg.seed),
        "stage": str(cfg.stage),
        "algorithm": _algorithm_type(cfg),
        "step": int(step),
        "world_model_dataset_id": wm_ckpt["dataset_id"],
        "world_model_checkpoint_path": str(wm_checkpoint_path.resolve()),
        "world_model_checkpoint_parent": str(wm_checkpoint_path.parent.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    checkpoint_path = Path(cfg.policy_checkpoint_dir) / "policy.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
