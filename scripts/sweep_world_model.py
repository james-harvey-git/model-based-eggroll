"""W&B sweep entry point for EGGROLL world-model hyperparameter tuning.

Called once per SLURM task via:
    wandb agent --count 1 "model-based-eggroll/model-based-eggroll/<SWEEP_ID>"

The agent sets sweep parameters on wandb.config before invoking this script.
This file bypasses Hydra's @hydra.main decorator and composes the OmegaConf
config manually, mirroring scripts/sweep.py for MOPO policy sweeps.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import wandb

from mbrl import experiments
from mbrl.logger import Logger, auto_tags


def compose_config(overrides: dict | None = None) -> DictConfig:
    """Manually replicate Hydra's config composition."""
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
        OmegaConf.load("configs/world_model/eggroll_ensemble.yaml"),
    )
    OmegaConf.update(base, "wandb.enabled", True)
    OmegaConf.update(base, "debug", False)
    OmegaConf.update(base, "world_model.eggroll.solver", "adamw")

    if overrides:
        mapping = {
            "population_size": "world_model.eggroll.population_size",
            "group_size": "world_model.eggroll.group_size",
            "weight_decay": "world_model.eggroll.solver_kwargs.weight_decay",
            "b1": "world_model.eggroll.solver_kwargs.b1",
            "sigma": "world_model.eggroll.sigma",
            "sigma_decay_rate": "world_model.eggroll.sigma_decay_rate",
            "lr": "world_model.eggroll.lr",
        }
        for key, path in mapping.items():
            if key in overrides:
                OmegaConf.update(base, path, overrides[key], force_add=True)

    return base


def _validate_sweep_params(cfg: DictConfig) -> None:
    """Validate EGGROLL population/group constraints before launching a run."""
    population_size = int(cfg.world_model.eggroll.population_size)
    group_size = int(cfg.world_model.eggroll.group_size)
    if group_size == 0:
        return
    if group_size % 2 != 0:
        raise ValueError(f"group_size must be even when nonzero (got {group_size})")
    if population_size % group_size != 0:
        raise ValueError(
            "group_size must divide population_size "
            f"(got population_size={population_size}, group_size={group_size})"
        )


def sweep_run() -> None:
    """Execute a single W&B sweep trial for world-model training."""
    wandb.init()
    sweep_params = dict(wandb.config)

    cfg = compose_config(overrides=sweep_params)
    _validate_sweep_params(cfg)

    if wandb.run is not None:
        existing = set(wandb.run.tags or ())
        new_tags = set(auto_tags(cfg))
        wandb.run.tags = tuple(sorted(existing | new_tags))

        run_id = wandb.run.id
        wm_group = wandb.run.group or f"eggroll-wm-sweep-{run_id}"
    else:
        run_id = "local"
        wm_group = f"eggroll-wm-sweep-{run_id}"

    checkpoint_dir = Path("checkpoints/sweep/world_model") / run_id
    OmegaConf.update(cfg, "checkpoint_dir", str(checkpoint_dir))

    logger = Logger.from_existing_run(cfg, wm_group=wm_group)
    try:
        experiments.world_model.run(cfg, logger)
    finally:
        logger.finish()


if __name__ == "__main__":
    sweep_run()
