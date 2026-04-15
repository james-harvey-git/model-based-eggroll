"""W&B logger. Domain-specific log methods are stubbed until training loops exist."""

from datetime import datetime
import os
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


def make_wm_group(cfg: DictConfig, timestamp: str) -> str:
    """Generate the world-model group name: used as W&B group and checkpoint subdir.

    Format: {wm}-{dataset}-s{seed}-{YYYYMMDD-HHMMSS}
    """
    return f"{_world_model_type(cfg)}-{_dataset_short(cfg)}-s{cfg.seed}-{timestamp}"


def _auto_name(cfg: DictConfig, timestamp: str) -> str:
    stage = cfg.get("stage", "all")
    stage_suffix = _STAGE_SUFFIX.get(stage, "")
    wm = _world_model_type(cfg)
    dataset = _dataset_short(cfg)
    seed = cfg.seed
    if stage == "world_model":
        return f"{wm}-{dataset}-s{seed}-{timestamp}{stage_suffix}"
    algo = _algorithm_type(cfg)
    return f"{wm}-{algo}-{dataset}-s{seed}-{timestamp}{stage_suffix}"


def auto_tags(cfg: DictConfig) -> list[str]:
    """Build the full tag list: sorted auto tags first, then manual tags in config order."""
    stage = cfg.get("stage", "all")

    auto: set[str] = set()
    auto.add(_world_model_type(cfg))
    auto.add(_dataset_short(cfg))
    if stage != "world_model":
        auto.add(_algorithm_type(cfg))
    if _is_sweep_run():
        auto.add("sweep")
    if os.environ.get("SLURM_JOB_ID"):
        auto.add("cluster")
    if cfg.get("debug", False):
        auto.add("debug")

    manual = [t.lower() for t in cfg.get("wandb", {}).get("tags", [])]
    return sorted(auto) + [t for t in manual if t not in auto]


def _is_sweep_run() -> bool:
    """Return True when running under a W&B sweep agent."""
    if os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID"):
        return True
    return wandb.run is not None and getattr(wandb.run, "sweep_id", None) is not None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class Logger:
    """Thin wrapper around W&B. Passed into each experiment stage."""

    def __init__(
        self,
        cfg: DictConfig,
        wm_group: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.wm_group: str = wm_group or make_wm_group(cfg, ts)

        wandb_cfg = cfg.get("wandb", {})
        self.enabled: bool = wandb_cfg.get("enabled", False)  # type: ignore[union-attr]
        if self.enabled:
            name = wandb_cfg.get("name", None) or _auto_name(cfg, ts)  # type: ignore[union-attr]
            wandb.init(
                project="model-based-eggroll",
                entity=wandb_cfg.get("entity", "model-based-eggroll"),  # type: ignore[union-attr]
                group=self.wm_group,
                job_type=cfg.get("stage", "all"),
                name=name,
                tags=auto_tags(cfg),
                config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
            )

    @classmethod
    def from_existing_run(cls, _cfg: DictConfig, wm_group: str | None = None) -> "Logger":
        """Create a Logger that attaches to an already-initialised W&B run.

        Used by sweep scripts where wandb.init() is called before Logger creation.
        """
        instance = cls.__new__(cls)
        instance.enabled = True
        instance.wm_group = wm_group or ""
        return instance

    def finish(self) -> None:
        if not self.enabled:
            return
        wandb.finish()

    def set_crashed_tag(self) -> None:
        """Add 'crashed' tag to the current W&B run. No-op when disabled. Idempotent."""
        if not self.enabled or wandb.run is None:
            return
        tags = tuple(wandb.run.tags or ())
        if "crashed" not in tags:
            wandb.run.tags = (*tags, "crashed")

    def log_world_model_step(self, epoch: int, **metrics: float) -> None:
        """Log per-epoch world model metrics (losses plus optional work counters)."""
        if not self.enabled:
            return
        wandb.log({f"world_model/{k}": v for k, v in metrics.items()}, step=epoch)

    def log_policy_step(self, step: int, **metrics: float) -> None:
        """Log policy training metrics (return, entropy, critic loss, etc.)."""
        if not self.enabled:
            return
        wandb.log({f"policy/{k}": v for k, v in metrics.items()}, step=step)

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
