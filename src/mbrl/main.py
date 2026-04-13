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
from typing import Any
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from mbrl import experiments
from mbrl.logger import Logger, _algorithm_type, make_wm_group


def _load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _update_latest_symlink(base_dir: Path, target_name: str) -> None:
    """Atomically update {base_dir}/latest -> {target_name}/."""
    tmp = base_dir / "latest.tmp"
    link = base_dir / "latest"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target_name)
    os.replace(tmp, link)


def _load_policy_checkpoint(policy_dir: Path) -> dict[str, Any]:
    """Load ``policy.pkl`` from *policy_dir* with a clear error if missing."""
    checkpoint_path = policy_dir / "policy.pkl"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No policy checkpoint found at '{checkpoint_path}'. "
            f"Pass policy_checkpoint_dir= pointing to a directory containing policy.pkl."
        )
    return _load_pickle(checkpoint_path)


def _infer_wm_checkpoint_from_policy_dir(policy_dir: Path) -> Path | None:
    """Infer the sibling ``world_model.pkl`` from a standard policy run layout."""
    candidate = policy_dir.parent.parent / "world_model.pkl"
    return candidate if candidate.exists() else None


def _find_latest_wm_for_dataset(base_dir: Path, dataset_name: str) -> Path:
    """Return the most recent world_model.pkl under base_dir trained on dataset_name.

    Prefers the explicit ``{base_dir}/latest`` symlink when it points to a
    checkpoint for the requested dataset. Falls back to modification time only
    when no usable explicit pointer is available.
    """
    latest_ckpt = base_dir / "latest" / "world_model.pkl"
    if latest_ckpt.exists():
        latest = _load_pickle(latest_ckpt)
        if latest.get("dataset_id") == dataset_name:
            return latest_ckpt
        warnings.warn(
            f"Ignoring explicit latest world-model pointer at '{latest_ckpt}' because it "
            f"targets dataset '{latest.get('dataset_id', '<unknown>')}', not '{dataset_name}'. "
            "Falling back to modification-time resolution among matching checkpoints.",
            stacklevel=2,
        )

    parts = dataset_name.split("/")
    env = parts[1] if len(parts) > 1 else dataset_name
    split = parts[2].rsplit("-v", 1)[0] if len(parts) > 2 else ""
    dataset_short = f"{env}-{split}" if split else env

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory '{base_dir}' does not exist. "
            f"Train a world model first:\n"
            f"  python src/mbrl/main.py stage=world_model"
        )

    candidates = []
    for d in base_dir.iterdir():
        if not d.is_dir() or dataset_short not in d.name:
            continue
        pkl = d / "world_model.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            ckpt = pickle.load(f)
        if ckpt.get("dataset_id") == dataset_name:
            candidates.append(pkl)

    if not candidates:
        raise FileNotFoundError(
            f"No world model checkpoint found for dataset '{dataset_name}' "
            f"under '{base_dir}'.\n"
            f"Train one with: python src/mbrl/main.py stage=world_model"
        )
    warnings.warn(
        f"No explicit latest world-model pointer found for dataset '{dataset_name}' under "
        f"'{base_dir}'. Falling back to the newest matching checkpoint by modification time.",
        stacklevel=2,
    )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_latest_policy_run(policies_dir: Path) -> Path:
    """Return the preferred policy run directory under *policies_dir*.

    Prefers the explicit ``latest`` symlink. If no such pointer exists (or it is
    broken), falls back to the newest directory containing ``policy.pkl``.
    """
    if not policies_dir.exists():
        raise FileNotFoundError(
            f"Policies directory '{policies_dir}' does not exist. "
            "Train a policy first before running stage=eval."
        )

    latest_dir = policies_dir / "latest"
    if latest_dir.exists():
        resolved = latest_dir.resolve()
        if (resolved / "policy.pkl").exists():
            return resolved
        warnings.warn(
            f"Ignoring broken latest policy pointer at '{latest_dir}' because it does not "
            "point to a directory containing policy.pkl. Falling back to modification-time "
            "resolution.",
            stacklevel=2,
        )

    candidates = [
        d for d in policies_dir.iterdir()
        if d.name != "latest" and d.is_dir() and (d / "policy.pkl").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No policy checkpoints found under '{policies_dir}'. "
            "Train a policy first before running stage=eval."
        )
    warnings.warn(
        f"No explicit latest policy pointer found under '{policies_dir}'. Falling back to "
        "the newest policy checkpoint directory by modification time.",
        stacklevel=2,
    )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_eval_inputs(
    base_checkpoint_dir: Path,
    dataset_name: str,
    policy_checkpoint_dir: str | None,
    allow_lineage_mismatch: bool = False,
) -> tuple[Path | None, Path, dict[str, Any], str]:
    """Resolve eval inputs and return ``(wm_ckpt, policy_dir, policy_ckpt, wm_group)``.

    When a policy checkpoint is explicitly provided, the policy artifact becomes
    the source of truth for experiment lineage. If an explicit world-model run
    directory is also provided via ``checkpoint_dir`` and its lineage disagrees
    with the policy checkpoint, raise unless ``allow_lineage_mismatch`` is true.
    """
    resolved_wm_ckpt: Path | None = None
    direct_wm_ckpt = base_checkpoint_dir / "world_model.pkl"
    if direct_wm_ckpt.exists():
        resolved_wm_ckpt = direct_wm_ckpt

    if policy_checkpoint_dir is None:
        if resolved_wm_ckpt is None:
            resolved_wm_ckpt = _find_latest_wm_for_dataset(base_checkpoint_dir, dataset_name)
        policy_dir = _find_latest_policy_run(resolved_wm_ckpt.parent / "policies")
    else:
        policy_dir = Path(policy_checkpoint_dir)

    policy_ckpt = _load_policy_checkpoint(policy_dir)
    policy_wm_group = policy_ckpt.get("wm_group")

    inferred_wm_ckpt = _infer_wm_checkpoint_from_policy_dir(policy_dir)
    if resolved_wm_ckpt is None and inferred_wm_ckpt is not None:
        resolved_wm_ckpt = inferred_wm_ckpt

    resolved_wm_group: str | None = None
    if resolved_wm_ckpt is not None:
        resolved_wm_group = _load_pickle(resolved_wm_ckpt).get("wm_group")

    if policy_wm_group is None:
        if resolved_wm_group is None:
            raise ValueError(
                f"Policy checkpoint '{policy_dir / 'policy.pkl'}' is missing wm_group provenance "
                "and no matching world-model checkpoint could be inferred. Re-train the policy "
                "with the new checkpoint format or pass checkpoint_dir= pointing to the original "
                "world-model run."
            )
        policy_wm_group = resolved_wm_group
    elif resolved_wm_group is not None and policy_wm_group != resolved_wm_group:
        message = (
            f"Lineage mismatch: policy checkpoint '{policy_dir / 'policy.pkl'}' records "
            f"wm_group='{policy_wm_group}', but checkpoint_dir points to world model "
            f"wm_group='{resolved_wm_group}' at '{resolved_wm_ckpt}'."
        )
        if allow_lineage_mismatch:
            warnings.warn(
                message + " Proceeding with the policy checkpoint lineage because "
                "allow_lineage_mismatch=true.",
                stacklevel=2,
            )
        else:
            raise ValueError(
                message
                + " Pass checkpoint_dir= for the matching world-model run, or set "
                "allow_lineage_mismatch=true to override."
            )

    return resolved_wm_ckpt, policy_dir, policy_ckpt, policy_wm_group


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_checkpoint_dir = Path(cfg.checkpoint_dir)
    stage = cfg.get("stage", "all")

    if stage in ("world_model", "all"):
        wm_group = make_wm_group(cfg, timestamp)
        OmegaConf.update(cfg, "checkpoint_dir", str(base_checkpoint_dir / wm_group))
    elif stage == "policy":
        wm_ckpt = base_checkpoint_dir / "world_model.pkl"
        if not wm_ckpt.exists():
            # Auto-find the most recent checkpoint for this dataset so the user
            # doesn't need to specify checkpoint_dir for the common split-stage workflow.
            wm_ckpt = _find_latest_wm_for_dataset(base_checkpoint_dir, cfg.dataset.name)
            base_checkpoint_dir = wm_ckpt.parent
            OmegaConf.update(cfg, "checkpoint_dir", str(base_checkpoint_dir))
        ckpt = _load_pickle(wm_ckpt)
        ckpt_dataset = ckpt.get("dataset_id", "<unknown>")
        if ckpt_dataset != cfg.dataset.name:
            raise ValueError(
                f"Dataset mismatch: config specifies '{cfg.dataset.name}' but "
                f"world model checkpoint was trained on '{ckpt_dataset}'. "
                f"Pass checkpoint_dir= pointing to the correct run, or re-train "
                f"the world model with dataset={cfg.dataset.name.split('/')[-2]}."
            )
        wm_group = ckpt["wm_group"]
    else:  # eval
        resolved_wm_ckpt, policy_dir, policy_ckpt, wm_group = _resolve_eval_inputs(
            base_checkpoint_dir,
            cfg.dataset.name,
            cfg.get("policy_checkpoint_dir"),
            allow_lineage_mismatch=bool(cfg.get("allow_lineage_mismatch", False)),
        )
        if resolved_wm_ckpt is not None:
            base_checkpoint_dir = resolved_wm_ckpt.parent
            OmegaConf.update(cfg, "checkpoint_dir", str(base_checkpoint_dir))
        else:
            inferred_wm_ckpt = _infer_wm_checkpoint_from_policy_dir(policy_dir)
            if inferred_wm_ckpt is not None:
                OmegaConf.update(cfg, "checkpoint_dir", str(inferred_wm_ckpt.parent))
        OmegaConf.update(cfg, "policy_checkpoint_dir", str(policy_dir))
        if "dataset_id" in policy_ckpt:
            OmegaConf.update(cfg, "dataset.name", policy_ckpt["dataset_id"])

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
            _update_latest_symlink(policy_dir.parent, policy_dir.name)
        if stage in ("eval", "all"):
            if cfg.get("policy_checkpoint_dir") is None:
                policies_dir = Path(cfg.checkpoint_dir) / "policies"
                OmegaConf.update(
                    cfg,
                    "policy_checkpoint_dir",
                    str(_find_latest_policy_run(policies_dir)),
                )
            experiments.evaluate.run(cfg, logger)
    except Exception:
        logger.set_crashed_tag()
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
