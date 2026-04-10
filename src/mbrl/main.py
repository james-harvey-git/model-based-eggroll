"""Single Hydra entry point for all experiments.

Run a specific stage or the full pipeline via cfg.stage:
    uv run python src/mbrl/main.py stage=world_model
    uv run python src/mbrl/main.py stage=policy
    uv run python src/mbrl/main.py stage=eval
    uv run python src/mbrl/main.py stage=all  (default)

Combine with config group overrides:
    uv run python src/mbrl/main.py world_model=mle policy_optimizer=mopo
    uv run python src/mbrl/main.py world_model=eggroll_ensemble policy_optimizer=eggroll
"""

from datetime import datetime
import os
from pathlib import Path
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf

from mbrl import experiments
from mbrl.logger import Logger, _algorithm_type, make_wm_group


def _update_latest_symlink(base_dir: Path, wm_group: str) -> None:
    """Atomically update {base_dir}/latest -> {wm_group}/."""
    tmp = base_dir / "latest.tmp"
    link = base_dir / "latest"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(wm_group)
    os.replace(tmp, link)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_checkpoint_dir = Path(cfg.checkpoint_dir)
    stage = cfg.get("stage", "all")

    if stage in ("world_model", "all"):
        wm_group = make_wm_group(cfg, timestamp)
        OmegaConf.update(cfg, "checkpoint_dir", str(base_checkpoint_dir / wm_group))
    else:  # policy or eval — read wm_group from existing checkpoint
        wm_ckpt = base_checkpoint_dir / "world_model.pkl"
        if not wm_ckpt.exists():
            # Fall back to latest symlink for the common split-stage workflow where
            # checkpoint_dir is left at its default and latest/ points to the last WM run.
            base_checkpoint_dir = (base_checkpoint_dir / "latest").resolve()
            OmegaConf.update(cfg, "checkpoint_dir", str(base_checkpoint_dir))
            wm_ckpt = base_checkpoint_dir / "world_model.pkl"
        with open(wm_ckpt, "rb") as f:
            ckpt = pickle.load(f)
        ckpt_dataset = ckpt.get("dataset_id", "<unknown>")
        if ckpt_dataset != cfg.dataset.name:
            raise ValueError(
                f"Dataset mismatch: config specifies '{cfg.dataset.name}' but "
                f"world model checkpoint was trained on '{ckpt_dataset}'. "
                f"Pass checkpoint_dir= pointing to the correct run, or re-train "
                f"the world model with dataset={cfg.dataset.name.split('/')[-2]}."
            )
        wm_group = ckpt["wm_group"]

    logger = Logger(cfg, wm_group=wm_group, timestamp=timestamp)
    try:
        if stage in ("world_model", "all"):
            experiments.world_model.run(cfg, logger)
            _update_latest_symlink(base_checkpoint_dir, wm_group)
        if stage in ("policy", "all"):
            policy_dir = (
                Path(cfg.checkpoint_dir)
                / "policies"
                / f"{_algorithm_type(cfg)}-s{cfg.seed}-{timestamp}"
            )
            OmegaConf.update(cfg, "policy_checkpoint_dir", str(policy_dir))
            experiments.policy.run(cfg, logger)
        if stage in ("eval", "all"):
            if cfg.get("policy_checkpoint_dir") is None:
                # Auto-resolve: pick the most recently modified policy run subdir.
                policies_dir = Path(cfg.checkpoint_dir) / "policies"
                runs = sorted(policies_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                if not runs:
                    raise FileNotFoundError(f"No policy checkpoints found under {policies_dir}")
                OmegaConf.update(cfg, "policy_checkpoint_dir", str(runs[-1]))
            experiments.evaluate.run(cfg, logger)
    except Exception:
        logger.set_crashed_tag()
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
