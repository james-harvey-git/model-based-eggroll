"""World-model validation MSE on an arbitrary dataset.

Loads a trained world-model checkpoint, loads any Minari dataset, and reports a
single scalar matching that world model's training-time ``val_mse`` (or
``val_mse_elite`` for ensembles). Used to probe OOD robustness by evaluating a
checkpoint trained on one behavioural-policy dataset against another (e.g. a
``halfcheetah-medium`` checkpoint vs ``halfcheetah-expert`` data).

When ``cfg.wm_eval.traj_horizon > 0`` (and the model class supports it) the open-loop
rollout MSE and its per-step compounding-error curve are also reported — one-step MSE
correlates only weakly with rollout usefulness, so this is the headline model-quality
number for MBRL.
"""

from pathlib import Path
import pickle

from omegaconf import DictConfig

from mbrl.data import load_dataset, load_episodes
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

    # Open-loop rollout MSE over the full eval dataset (when evaluating the training
    # dataset itself this includes its training split — fine for cross-dataset probing,
    # but not a held-out number).
    traj_kwargs: dict = {}
    traj_horizon = int(cfg.get("wm_eval", {}).get("traj_horizon", 0) or 0)
    if traj_horizon > 0:
        compute_grounding = getattr(world_model, "compute_traj_grounding", None)
        if compute_grounding is not None:
            episodes, _ = load_episodes(cfg.dataset.name)
            include_persistence = bool(
                cfg.get("wm_eval", {}).get("log_persistence_baseline", False)
            )
            traj_mse, traj_mse_elite, curve, nmse, figures = compute_grounding(
                episodes, traj_horizon, info.dataset_id, include_persistence
            )
            traj_kwargs = dict(
                traj_mse=float(traj_mse),
                traj_mse_elite=float(traj_mse_elite),
                traj_mse_curve=[float(v) for v in curve],
                traj_nmse_curve=nmse,
                figures=figures,
            )
            print(
                f"wm_eval: traj_mse@h{traj_horizon}={traj_kwargs['traj_mse']:.6f} "
                f"(elite {traj_kwargs['traj_mse_elite']:.6f})"
            )
        else:
            print(
                "wm_eval: world-model class has no compute_traj_grounding; "
                "skipping rollout evaluation."
            )

    logger.log_wm_eval(ckpt["dataset_id"], info.dataset_id, val_mse, **traj_kwargs)
