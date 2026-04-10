"""W&B sweep entry point for MOPO hyperparameter tuning.

Called once per SLURM array task via:
    wandb agent --count 1 "model-based-eggroll/<SWEEP_ID>"

The agent sets sweep parameters via wandb.config before invoking this script.
This file bypasses Hydra's @hydra.main decorator (which reads CLI args) and
instead composes the OmegaConf config manually, then overrides policy_optimizer
fields from wandb.config.

Environment variables:
    WM_CHECKPOINT: Absolute path to a pre-trained world_model.pkl. The script
                   symlinks it into cfg.checkpoint_dir so experiments/policy.py
                   finds it at the expected location.
"""

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import wandb

from mbrl import experiments
from mbrl.logger import Logger


def compose_config(overrides: dict | None = None) -> DictConfig:
    """Manually replicate Hydra's config composition.

    Loads configs/config.yaml, resolves each entry in the defaults list by
    loading the corresponding group YAML, then applies overrides from
    wandb.config on top of policy_optimizer.
    """
    raw = OmegaConf.load("configs/config.yaml")
    assert isinstance(raw, DictConfig)
    # Pop the Hydra defaults list — not valid outside of @hydra.main
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

    # Override policy_optimizer fields from sweep parameters
    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(base, f"policy_optimizer.{key}", value)

    return base


def sweep_run() -> None:
    """Execute a single W&B sweep trial (policy training only)."""
    wandb.init()
    sweep_params = dict(wandb.config)

    cfg = compose_config(overrides=sweep_params)
    OmegaConf.update(cfg, "wandb.enabled", True)
    OmegaConf.update(cfg, "stage", "policy")

    # Each sweep run writes its policy checkpoint to a run-specific subdir
    # so concurrent runs don't clobber each other.
    run_id = wandb.run.id if wandb.run is not None else "local"
    OmegaConf.update(cfg, "checkpoint_dir", f"checkpoints/sweep/{run_id}")

    # Symlink the shared pre-trained world model into this run's checkpoint dir
    wm_path = os.environ.get("WM_CHECKPOINT")
    if wm_path:
        ckpt_dir = Path(str(cfg.checkpoint_dir))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        link = ckpt_dir / "world_model.pkl"
        if not link.exists():
            link.symlink_to(Path(wm_path).resolve())

    logger = Logger.from_existing_run(cfg)
    try:
        experiments.policy.run(cfg, logger)
    finally:
        logger.finish()


if __name__ == "__main__":
    sweep_run()
