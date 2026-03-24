"""Policy training experiment."""

from omegaconf import DictConfig

from mbrl.logging import Logger


def run(cfg: DictConfig, logger: Logger) -> None:
    """Load a world model checkpoint, train a policy, and save a checkpoint.

    Configured by cfg.algorithm. Checkpoints loaded from / saved to cfg.checkpoint_dir.
    Full implementation deferred to Phase 3 (MOPO) and Phase 6 (EGGROLL policy).
    """
    raise NotImplementedError
