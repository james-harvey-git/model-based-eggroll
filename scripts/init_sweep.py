"""Register a W&B sweep and print the sweep ID.

Usage (run once from the repo root before submitting the SLURM array):
    uv run python scripts/init_sweep.py
    uv run python scripts/init_sweep.py configs/sweep/mopo_halfcheetah.yaml
"""

import sys

import yaml

import wandb


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/sweep/mopo_halfcheetah.yaml"
    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, entity="model-based-eggroll", project="model-based-eggroll")
    print(f"Sweep ID: {sweep_id}")
    print(f"View at: https://wandb.ai/model-based-eggroll/model-based-eggroll/sweeps/{sweep_id}")


if __name__ == "__main__":
    main()
