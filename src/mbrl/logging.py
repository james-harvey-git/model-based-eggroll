"""W&B logger. Domain-specific log methods are stubbed until training loops exist."""

import wandb
from omegaconf import DictConfig, OmegaConf


class Logger:
    """Thin wrapper around W&B. Passed into each experiment stage."""

    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            project="model-based-eggroll",
            entity="model-based-eggroll",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def finish(self) -> None:
        wandb.finish()

    def log_world_model_step(self, step: int, **metrics) -> None:
        """Log world model training metrics (NLL, RMSE, val loss, etc.)."""
        raise NotImplementedError

    def log_policy_step(self, step: int, **metrics) -> None:
        """Log policy training metrics (return, entropy, critic loss, etc.)."""
        raise NotImplementedError

    def log_eval(self, dataset_id: str, raw_score: float, normalized_score: float) -> None:
        """Log final evaluation results."""
        wandb.log({
            "eval/raw_score": raw_score,
            "eval/normalized_score": normalized_score,
            "eval/dataset_id": dataset_id,
        })
