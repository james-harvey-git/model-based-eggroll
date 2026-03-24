"""Single Hydra entry point for all experiments.

Run a specific stage or the full pipeline via cfg.stage:
    uv run python src/mbrl/main.py stage=world_model
    uv run python src/mbrl/main.py stage=policy
    uv run python src/mbrl/main.py stage=eval
    uv run python src/mbrl/main.py stage=all  (default)

Combine with config group overrides:
    uv run python src/mbrl/main.py world_model=mle algorithm=mopo
    uv run python src/mbrl/main.py world_model=eggroll_ensemble algorithm=eggroll
"""

import hydra
from omegaconf import DictConfig

from mbrl import experiments
from mbrl.logging import Logger


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = Logger(cfg)
    try:
        if cfg.stage in ("world_model", "all"):
            experiments.world_model.run(cfg, logger)
        if cfg.stage in ("policy", "all"):
            experiments.policy.run(cfg, logger)
        if cfg.stage in ("eval", "all"):
            experiments.evaluation.run(cfg, logger)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
