"""World model training experiment."""

import pickle
from pathlib import Path

import jax
from omegaconf import DictConfig, OmegaConf

from mbrl.data import load_dataset
from mbrl.logging import Logger
from mbrl.world_models.mle import MLEEnsemble


def run(cfg: DictConfig, logger: Logger) -> None:
    """Train a world model and save a checkpoint.

    Configured by cfg.world_model. Checkpoints saved to cfg.checkpoint_dir.
    """
    rng = jax.random.key(cfg.seed)
    rng, train_rng = jax.random.split(rng)

    dataset, info = load_dataset(cfg.dataset.name)

    world_model = MLEEnsemble(info.obs_dim, info.act_dim, info.dataset_id, cfg.world_model)

    def log_fn(epoch: int, train_loss: float, val_mse: float) -> None:
        logger.log_world_model_step(
            int(epoch), train_loss=float(train_loss), val_mse=float(val_mse)
        )

    world_model.train(dataset, cfg.world_model, train_rng, log_fn=log_fn)

    checkpoint = {
        "params": world_model.params,
        "num_elites": world_model.num_elites,
        "obs_dim": info.obs_dim,
        "act_dim": info.act_dim,
        "dataset_id": info.dataset_id,
        "world_model_cfg": OmegaConf.to_container(cfg.world_model),
    }
    checkpoint_path = Path(cfg.checkpoint_dir) / "world_model.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
