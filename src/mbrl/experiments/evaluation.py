"""Evaluation experiment: roll out a policy in the real environment and log the score."""

from omegaconf import DictConfig

from mbrl.evaluation import compute_normalized_score, evaluate_policy
from mbrl.logging import Logger


def run(cfg: DictConfig, logger: Logger) -> None:
    """Load a policy checkpoint, evaluate in the real environment, and log the score.

    Wraps mbrl.evaluation. Checkpoint loading deferred to Phase 3.
    """
    raise NotImplementedError
