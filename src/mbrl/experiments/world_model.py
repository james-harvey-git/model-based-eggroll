"""World model training experiment."""

import math
from pathlib import Path
import pickle
import time

from hydra.utils import get_class
import jax
from omegaconf import DictConfig, OmegaConf

from mbrl.data import load_dataset
from mbrl.logger import Logger
from mbrl.world_models.eggroll import EGGROLLEnsemble
from mbrl.world_models.mle import MLEEnsemble


def run(cfg: DictConfig, logger: Logger) -> None:
    """Train a world model and save a checkpoint.

    Configured by cfg.world_model. Checkpoints saved to cfg.checkpoint_dir.
    Dispatches to the right world model class via cfg.world_model._target_.
    """
    rng = jax.random.key(cfg.seed)
    rng, train_rng = jax.random.split(rng)

    dataset, info = load_dataset(cfg.dataset.name)

    wm_cls = get_class(cfg.world_model._target_)
    world_model = wm_cls(info.obs_dim, info.act_dim, info.dataset_id, cfg.world_model)
    start_time = time.perf_counter()
    max_step = max(int(cfg.world_model.num_epochs), 1)

    def log_fn(
        epoch: int,
        train_loss: float,
        val_mse: float,
        transitions_seen: int,
        forward_evals: int,
    ) -> None:
        metrics: dict[str, float] = {}
        train_loss_f = float(train_loss)
        if math.isfinite(train_loss_f):
            metrics["train_loss"] = train_loss_f
        val_mse_f = float(val_mse)
        if math.isfinite(val_mse_f):
            metrics["val_mse"] = val_mse_f
        metrics["normalized_step"] = float(epoch) / max_step
        metrics["transitions_seen"] = float(transitions_seen)
        metrics["forward_evals"] = float(forward_evals)
        metrics["wall_time_sec"] = time.perf_counter() - start_time
        logger.log_world_model_step(int(epoch), **metrics)

    world_model.train(dataset, cfg.world_model, train_rng, log_fn=log_fn)

    common = {
        "obs_dim": info.obs_dim,
        "act_dim": info.act_dim,
        "dataset_id": info.dataset_id,
        "world_model_cfg": OmegaConf.to_container(cfg.world_model),
        "wm_group": logger.wm_group,
    }

    if isinstance(world_model, MLEEnsemble):
        checkpoint = {
            **common,
            "params": world_model.params,
            "num_elites": world_model.num_elites,
        }
    else:
        assert isinstance(world_model, EGGROLLEnsemble)
        checkpoint = {
            **common,
            "eggroll_state": world_model.checkpoint_state(),
            "last_train_epoch": world_model._last_train_epoch,
        }

    checkpoint_path = Path(cfg.checkpoint_dir) / "world_model.pkl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
