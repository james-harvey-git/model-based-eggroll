"""Single Hydra entry point for all experiments.

Compose world_model= and algorithm= config groups to run any combination, e.g.:
    uv run python src/mbrl/train.py world_model=mle algorithm=mopo
    uv run python src/mbrl/train.py world_model=eggroll_ensemble algorithm=eggroll
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
