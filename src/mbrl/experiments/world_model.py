"""World model training experiment."""

from omegaconf import DictConfig

from mbrl.logging import Logger


def run(cfg: DictConfig, logger: Logger) -> None:
    """Train a world model and save a checkpoint.

    Configured by cfg.world_model. Checkpoints saved to cfg.checkpoint_dir.
    Full implementation deferred to Phase 2 (MLE) and Phase 5 (EGGROLL).
    """
    raise NotImplementedError
