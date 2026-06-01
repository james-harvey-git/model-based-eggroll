"""World-model validation MSE on an arbitrary dataset.

Loads a trained world-model checkpoint, loads any Minari dataset, and reports a
single scalar matching that world model's training-time ``val_mse`` (or
``val_mse_elite`` for ensembles). Used to probe OOD robustness by evaluating a
checkpoint trained on one behavioural-policy dataset against another (e.g. a
``halfcheetah-medium`` checkpoint vs ``halfcheetah-expert`` data).
"""

from pathlib import Path
import pickle

from omegaconf import DictConfig

from mbrl.data import load_dataset
from mbrl.logger import Logger
from mbrl.world_models.base import load_world_model_from_checkpoint


def run(cfg: DictConfig, logger: Logger) -> None:
    """Compute and log validation MSE for a checkpoint against ``cfg.dataset``."""
    ckpt_path = Path(cfg.checkpoint_dir) / "world_model.pkl"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No world model checkpoint at '{ckpt_path}'. "
            f"Pass checkpoint_dir= pointing to a world-model run directory."
        )
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    dataset, info = load_dataset(cfg.dataset.name)

    # Fail fast on cross-environment misuse (e.g. halfcheetah ckpt vs walker dataset)
    # so the user sees a clear message before any JAX broadcast error.
    if ckpt["obs_dim"] != info.obs_dim or ckpt["act_dim"] != info.act_dim:
        raise ValueError(
            f"Shape mismatch: checkpoint obs/act=({ckpt['obs_dim']}, {ckpt['act_dim']}) "
            f"trained on '{ckpt['dataset_id']}' vs eval dataset "
            f"obs/act=({info.obs_dim}, {info.act_dim}) for '{info.dataset_id}'."
        )

    world_model = load_world_model_from_checkpoint(str(ckpt_path))
    val_mse = float(world_model.compute_val_mse(dataset))

    print(
        f"wm_eval: train={ckpt['dataset_id']} eval={info.dataset_id} "
        f"val_mse={val_mse:.6f}"
    )
    logger.log_wm_eval(ckpt["dataset_id"], info.dataset_id, val_mse)
