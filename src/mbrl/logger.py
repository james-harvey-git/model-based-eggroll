"""W&B logger. Domain-specific log methods are stubbed until training loops exist."""

from typing import Any, cast  # noqa: I001

from omegaconf import DictConfig, OmegaConf
import wandb

# ---------------------------------------------------------------------------
# Auto-naming helpers
# ---------------------------------------------------------------------------

_STAGE_SUFFIX: dict[str, str] = {
    "world_model": "-wm",
    "policy": "-pol",
    "eval": "-eval",
    "all": "",
}


def _world_model_type(cfg: DictConfig) -> str:
    """Extract a short world model identifier from _target_, e.g. 'mle' or 'eggroll'."""
    target: str = cfg.world_model.get("_target_", "unknown")
    class_name = target.split(".")[-1]  # e.g. "MLEEnsemble"
    return class_name.replace("Ensemble", "").lower()  # e.g. "mle"


def _algorithm_type(cfg: DictConfig) -> str:
    """Extract a short algorithm identifier from _target_, e.g. 'mopo'."""
    target: str = cfg.policy_optimizer.get("_target_", "unknown")
    # target is like "mbrl.policy_optimizers.mopo.train" — take the module name
    parts = target.split(".")
    return parts[-2] if len(parts) >= 2 else parts[-1]


def _dataset_short(cfg: DictConfig) -> str:
    """Shorten a Minari dataset ID, e.g. 'mujoco/halfcheetah/medium-v0' -> 'halfcheetah-medium'."""
    dataset_id: str = cfg.dataset.get("name", "unknown")
    parts = dataset_id.split("/")
    env = parts[1] if len(parts) > 1 else dataset_id
    split = parts[2].rsplit("-v", 1)[0] if len(parts) > 2 else ""
    return f"{env}-{split}" if split else env


def _auto_group(cfg: DictConfig) -> str:
    return f"{_world_model_type(cfg)}-{_algorithm_type(cfg)}-{_dataset_short(cfg)}"


def _auto_name(cfg: DictConfig) -> str:
    stage_suffix = _STAGE_SUFFIX.get(cfg.get("stage", "all"), "")
    return f"{_auto_group(cfg)}{stage_suffix}-s{cfg.seed}"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class Logger:
    """Thin wrapper around W&B. Passed into each experiment stage."""

    def __init__(self, cfg: DictConfig) -> None:
        wandb_cfg = cfg.get("wandb", {})
        self.enabled: bool = wandb_cfg.get("enabled", False)  # type: ignore[union-attr]
        if self.enabled:
            group = wandb_cfg.get("group", None) or _auto_group(cfg)  # type: ignore[union-attr]
            name = wandb_cfg.get("name", None) or _auto_name(cfg)  # type: ignore[union-attr]
            wandb.init(
                project="model-based-eggroll",
                entity=wandb_cfg.get("entity", "model-based-eggroll"),  # type: ignore[union-attr]
                group=group,
                job_type=cfg.get("stage", "all"),
                name=name,
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
