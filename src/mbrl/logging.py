"""W&B logger. Domain-specific log methods are stubbed until training loops exist."""

from typing import Any, cast

import wandb
from omegaconf import DictConfig, OmegaConf


class Logger:
    """Thin wrapper around W&B. Passed into each experiment stage."""

    def __init__(self, cfg: DictConfig) -> None:
        self.enabled: bool = cfg.get("wandb", {}).get("enabled", True)  # type: ignore[union-attr]
        if self.enabled:
            wandb.init(
                project="model-based-eggroll",
                entity="model-based-eggroll",
                config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            )

    def finish(self) -> None:
        if not self.enabled:
            return
        wandb.finish()

    def log_world_model_step(self, epoch: int, **metrics: float) -> None:
        """Log per-epoch world model training metrics (train_loss, val_mse, etc.)."""
        if not self.enabled:
            return
        wandb.log({f"world_model/{k}": v for k, v in metrics.items()}, step=epoch)

    def log_policy_step(self, step: int, **metrics: float) -> None:
        """Log policy training metrics (return, entropy, critic loss, etc.)."""
        raise NotImplementedError

    def log_eval(self, dataset_id: str, raw_score: float, normalized_score: float) -> None:
        """Log final evaluation results."""
        if not self.enabled:
            return
        wandb.log(
            {
                "eval/raw_score": raw_score,
                "eval/normalized_score": normalized_score,
                "eval/dataset_id": dataset_id,
            }
        )
