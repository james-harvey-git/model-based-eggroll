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
from mbrl.world_models.mle_dynamicsnet import MLEDynamicsNet


def run(cfg: DictConfig, logger: Logger) -> None:
    """Train a world model and save a checkpoint.

    Configured by cfg.world_model. Checkpoints saved to cfg.checkpoint_dir.
    Dispatches to the right world model class via cfg.world_model._target_.
    """
    rng = jax.random.key(cfg.seed)
    rng, train_rng = jax.random.split(rng)

    dataset, info = load_dataset(cfg.dataset.name)

    # When warm-starting EGGROLL from a stage-1 checkpoint, sanity-check that
    # we're finetuning on the dataset the checkpoint was pretrained on.
    init_ckpt_path = cfg.world_model.get("init_checkpoint", None)
    if init_ckpt_path is not None:
        with open(init_ckpt_path, "rb") as f:
            _ckpt_meta = pickle.load(f)
        if _ckpt_meta["dataset_id"] != info.dataset_id:
            raise ValueError(
                f"init_checkpoint dataset_id mismatch: ckpt={_ckpt_meta['dataset_id']!r} "
                f"vs current dataset={info.dataset_id!r}. Refusing to finetune on a "
                "different dataset to the one used for pretraining."
            )
        if _ckpt_meta["obs_dim"] != info.obs_dim or _ckpt_meta["act_dim"] != info.act_dim:
            raise ValueError(
                f"init_checkpoint shape mismatch: ckpt obs/act=({_ckpt_meta['obs_dim']},"
                f" {_ckpt_meta['act_dim']}) vs dataset=({info.obs_dim}, {info.act_dim})."
            )

    # Plumb the top-level seed into cfg.world_model so per-class trainers that
    # need a deterministic seed (e.g. MLEDynamicsNet, which records it in the
    # checkpoint so Stage 2 can replay the train/val split) can read it from
    # their own cfg without an extra constructor argument.
    if "seed" in cfg.world_model and cfg.world_model.seed is None:
        OmegaConf.set_struct(cfg.world_model, False)
        cfg.world_model.seed = int(cfg.seed)
        OmegaConf.set_struct(cfg.world_model, True)

    wm_cls = get_class(cfg.world_model._target_)
    world_model = wm_cls(info.obs_dim, info.act_dim, info.dataset_id, cfg.world_model)
    start_time = time.perf_counter()

    # Normalize the x-axis to update-step count (1 update per epoch for EGGROLL;
    # batches_per_epoch updates per epoch for MLE).
    if isinstance(world_model, (MLEEnsemble, MLEDynamicsNet)):
        n_train = int((1 - cfg.world_model.validation_split) * dataset.obs.shape[0])
        updates_per_epoch = max(n_train // cfg.world_model.batch_size, 1)
    else:
        updates_per_epoch = 1
    max_step = max(int(cfg.world_model.num_epochs) * updates_per_epoch, 1)

    def log_fn(
        step: int,
        train_loss: float,
        val_mse: float,
        transitions_seen: int,
        forward_evals: int,
        epoch: int | None = None,
        val_mse_elite: float | None = None,
        lr: float | None = None,
        sigma: float | None = None,
    ) -> None:
        metrics: dict[str, float] = {}
        train_loss_f = float(train_loss)
        if math.isfinite(train_loss_f):
            metrics["train_loss"] = train_loss_f
        val_mse_f = float(val_mse)
        if math.isfinite(val_mse_f):
            metrics["val_mse"] = val_mse_f
        if val_mse_elite is not None and math.isfinite(val_mse_elite):
            metrics["val_mse_elite"] = float(val_mse_elite)
        if lr is not None and math.isfinite(lr):
            metrics["lr"] = float(lr)
        if sigma is not None and math.isfinite(sigma):
            metrics["sigma"] = float(sigma)
        if epoch is not None:
            metrics["epoch"] = float(epoch)
        metrics["normalized_step"] = float(step) / max_step
        metrics["transitions_seen"] = float(transitions_seen)
        metrics["forward_evals"] = float(forward_evals)
        metrics["wall_time_sec"] = time.perf_counter() - start_time
        logger.log_world_model_step(int(step), **metrics)

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
    elif isinstance(world_model, MLEDynamicsNet):
        checkpoint = {**common, **world_model.checkpoint_state()}
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
