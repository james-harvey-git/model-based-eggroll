"""W&B sweep entry point for MLEDynamicsNet (Stage-1) hyperparameter tuning.

Called once per SLURM task via:
    wandb agent --count 1 "model-based-eggroll/model-based-eggroll/<SWEEP_ID>"

The agent sets sweep parameters on wandb.config before invoking this script.
This file bypasses Hydra's @hydra.main decorator and composes the OmegaConf
config manually, mirroring scripts/sweep_world_model.py (the EGGROLL world-model
sweep) but targeting the `mle_dynamicsnet` world model instead.

Primary use: sweeping the SGD learning rate for MLEDynamicsNet training.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import wandb

from mbrl import experiments
from mbrl.eggroll.training import resolve_optax_solver
from mbrl.logger import Logger, auto_tags


def compose_config(overrides: dict | None = None) -> DictConfig:
    """Manually replicate Hydra's config composition for an MLEDynamicsNet run."""
    raw = OmegaConf.load("configs/config.yaml")
    assert isinstance(raw, DictConfig)
    defaults = OmegaConf.to_container(raw.pop("defaults", []))
    assert isinstance(defaults, list)
    base = raw

    for item in defaults:
        if item == "_self_":
            continue
        if isinstance(item, dict):
            group, name = next(iter(item.items()))
            group_cfg = OmegaConf.load(f"configs/{group}/{name}.yaml")
            assert isinstance(group_cfg, DictConfig)
            base[group] = group_cfg  # type: ignore[index]

    OmegaConf.update(base, "stage", "world_model")
    OmegaConf.update(base, "dataset", OmegaConf.load("configs/dataset/halfcheetah_medium.yaml"))
    OmegaConf.update(
        base,
        "world_model",
        OmegaConf.load("configs/world_model/mle_dynamicsnet.yaml"),
    )
    OmegaConf.update(base, "wandb.enabled", True)
    OmegaConf.update(base, "debug", False)
    OmegaConf.update(base, "world_model.optimizer", "sgd")

    if overrides:
        # The base mle_dynamicsnet config carries AdamW-shaped optimizer_kwargs
        # ({eps, weight_decay}) which optax.sgd (and optax.adam) reject. Clear
        # them up front for any non-adamw solver, *before* applying the kwarg
        # overrides below — so a swept momentum lands in an empty dict rather
        # than merging on top of the stale AdamW eps. Resolve the solver from the
        # sweep override if present, else the base default (sgd).
        solver = str(
            overrides.get("optimizer", base.world_model.get("optimizer", "adamw"))
        ).lower()
        if solver != "adamw":
            OmegaConf.update(
                base, "world_model.optimizer_kwargs", {}, merge=False, force_add=True
            )

        mapping = {
            "lr": "world_model.lr",
            "optimizer": "world_model.optimizer",
            "momentum": "world_model.optimizer_kwargs.momentum",
            "weight_decay": "world_model.optimizer_kwargs.weight_decay",
            "num_epochs": "world_model.num_epochs",
            "batch_size": "world_model.batch_size",
            "backbone": "world_model.backbone",
            "activation": "world_model.activation",
            "validation_split": "world_model.validation_split",
            "logvar_diff_coef": "world_model.logvar_diff_coef",
        }
        for key, path in mapping.items():
            if key in overrides:
                OmegaConf.update(base, path, overrides[key], force_add=True)

    return base


def _validate_sweep_params(cfg: DictConfig) -> None:
    """Validate MLEDynamicsNet training knobs before launching a run."""
    # Fail fast on an unsupported solver name (resolve raises with the list).
    resolve_optax_solver(str(cfg.world_model.get("optimizer", "adamw")))
    if float(cfg.world_model.lr) <= 0.0:
        raise ValueError(f"lr must be positive (got {cfg.world_model.lr})")
    if int(cfg.world_model.batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1 (got {cfg.world_model.batch_size})")
    if int(cfg.world_model.num_epochs) < 1:
        raise ValueError(f"num_epochs must be >= 1 (got {cfg.world_model.num_epochs})")


def sweep_run() -> None:
    """Execute a single W&B sweep trial for MLEDynamicsNet training."""
    wandb.init()
    sweep_params = dict(wandb.config)

    cfg = compose_config(overrides=sweep_params)
    _validate_sweep_params(cfg)

    if wandb.run is not None:
        existing = set(wandb.run.tags or ())
        new_tags = set(auto_tags(cfg))
        wandb.run.tags = tuple(sorted(existing | new_tags))

        run_id = wandb.run.id
        wm_group = wandb.run.group or f"mle-dynamicsnet-sweep-{run_id}"
    else:
        run_id = "local"
        wm_group = f"mle-dynamicsnet-sweep-{run_id}"

    checkpoint_dir = Path("checkpoints/sweep/mle_dynamicsnet") / run_id
    OmegaConf.update(cfg, "checkpoint_dir", str(checkpoint_dir))

    logger = Logger.from_existing_run(cfg, wm_group=wm_group)
    try:
        experiments.world_model.run(cfg, logger)
    finally:
        logger.finish()


if __name__ == "__main__":
    sweep_run()
