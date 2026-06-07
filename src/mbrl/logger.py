"""W&B logger. Domain-specific log methods are stubbed until training loops exist."""

from datetime import datetime
import os
import pickle
from typing import Any, cast  # noqa: I001

from omegaconf import DictConfig, OmegaConf
import wandb

# ---------------------------------------------------------------------------
# Auto-naming helpers
# ---------------------------------------------------------------------------

_STAGE_SUFFIX: dict[str, str] = {
    "world_model": "-wm",
    "policy": "-pol",
    "eval": "-eval",
    "wm_eval": "-wmeval",
    "all": "",
}


def _world_model_type(cfg: DictConfig) -> str:
    """Short world-model identifier: the training algorithm (``eggroll`` | ``backprop``)
    for ``EnsembleMLP``, or ``unifloral`` for the ported Flax baseline.

    Used in run/group names, tags, and the checkpoint subdir so the backprop-vs-eggroll
    comparison the project exists to make is legible at a glance.
    """
    target: str = cfg.world_model.get("_target_", "unknown")
    class_name = target.split(".")[-1]
    if class_name == "EnsembleMLP":
        return str(cfg.world_model.get("trainer", "unknown"))
    if class_name == "UnifloralEnsembleMLP":
        return "unifloral"
    return class_name.lower()


def _algorithm_type(cfg: DictConfig) -> str:
    """Extract a short algorithm identifier from _target_, e.g. 'mopo'."""
    target: str = cfg.policy_optimizer.get("_target_", "unknown")
    # target is like "mbrl.policy_optimizers.mopo.train" — take the module name
    parts = target.split(".")
    return parts[-2] if len(parts) >= 2 else parts[-1]


def _dataset_short(cfg: DictConfig) -> str:
    """Shorten a Minari dataset ID, e.g. 'mujoco/halfcheetah/medium-v0' -> 'halfcheetah-medium'."""
    dataset_id: str = cfg.dataset.get("name", "unknown")
    parts = dataset_id.split("/")
    env = parts[1] if len(parts) > 1 else dataset_id
    split = parts[2].rsplit("-v", 1)[0] if len(parts) > 2 else ""
    return f"{env}-{split}" if split else env


def make_wm_group(cfg: DictConfig, timestamp: str) -> str:
    """Generate the world-model group name: used as W&B group and checkpoint subdir.

    Format: {wm}-{dataset}-s{seed}-{YYYYMMDD-HHMMSS}
    """
    return f"{_world_model_type(cfg)}-{_dataset_short(cfg)}-s{cfg.seed}-{timestamp}"


def _auto_name(cfg: DictConfig, timestamp: str) -> str:
    stage = cfg.get("stage", "all")
    stage_suffix = _STAGE_SUFFIX.get(stage, "")
    wm = _world_model_type(cfg)
    dataset = _dataset_short(cfg)
    seed = cfg.seed
    if stage in ("world_model", "wm_eval"):
        return f"{wm}-{dataset}-s{seed}-{timestamp}{stage_suffix}"
    algo = _algorithm_type(cfg)
    return f"{wm}-{algo}-{dataset}-s{seed}-{timestamp}{stage_suffix}"


def auto_tags(cfg: DictConfig) -> list[str]:
    """Build the full tag list: sorted auto tags first, then manual tags in config order."""
    stage = cfg.get("stage", "all")

    auto: set[str] = set()
    auto.add(_world_model_type(cfg))
    auto.add(_dataset_short(cfg))
    if stage in ("policy", "eval", "all"):
        auto.add(_algorithm_type(cfg))
    if stage == "wm_eval":
        auto.add("wm_eval")
    if _is_sweep_run():
        auto.add("sweep")
    if os.environ.get("SLURM_JOB_ID"):
        auto.add("cluster")
    if cfg.get("debug", False):
        auto.add("debug")

    wm = cfg.get("world_model", None)
    if wm is not None:
        if wm.get("init_checkpoint", None):
            auto.add("finetune")
        if wm.get("trainer") == "eggroll":
            auto.add("shared-pert" if wm.get("use_shared_perturbations", False) else "indep-pert")

    manual = [t.lower() for t in cfg.get("wandb", {}).get("tags", [])]
    return sorted(auto) + [t for t in manual if t not in auto]


def _legend_fields(cfg: DictConfig) -> dict:
    """Flat top-level keys promoted into the W&B run config for legend templates and
    runs-table columns.

    ``trainer`` is the training algorithm (eggroll | backprop) — named to avoid clashing
    with ``optimizer``/``solver``, which always means the optax solver. Shared
    hyperparameters are always included; trainer-specific ones (population size, sigma,
    ... for eggroll; batch size for backprop) appear only on the runs that have them —
    W&B leaves the column blank for runs that don't, which is the desired behaviour.
    """
    if "world_model" not in cfg:
        return {}
    wm = cfg.world_model
    wm_type = _world_model_type(cfg)
    fields: dict = {
        "trainer": "backprop" if wm_type == "unifloral" else wm_type,
        "backbone": str(wm.get("backbone", "mlp")),
    }
    for k in ("num_ensemble", "num_elites"):
        if k in wm:
            fields[k] = wm[k]
    if wm.get("trainer") == "eggroll" and "eggroll" in wm:
        eg = wm.eggroll
        fields["lr"] = eg.get("lr")
        fields["population_size"] = eg.get("population_size")
        fields["sigma"] = eg.get("sigma")
        fields["group_size"] = eg.get("group_size")
        fields["use_shared_perturbations"] = bool(wm.get("use_shared_perturbations", False))
    else:
        if "lr" in wm:
            fields["lr"] = wm.lr
        if "batch_size" in wm:
            fields["batch_size"] = wm.batch_size
    return fields


def _finetune_provenance(cfg: DictConfig) -> dict:
    """Manifest fields recording the checkpoint a fine-tune run started from.

    Recorded in the run config (not the run name) so backprop->eggroll, eggroll->backprop
    and eggroll->eggroll lineages are filterable. ``finetune_lineage`` chains across
    repeated fine-tunes by carrying the parent's recorded lineage forward.
    """
    wm = cfg.get("world_model", None)
    path = wm.get("init_checkpoint", None) if wm else None
    if not path:
        return {}
    try:
        with open(path, "rb") as f:
            src = pickle.load(f)
    except (FileNotFoundError, OSError, pickle.UnpicklingError):
        return {"is_finetune": True}
    src_cfg = src.get("world_model_cfg", {})
    src_class = str(src_cfg.get("_target_", "")).split(".")[-1]
    if src_class == "EnsembleMLP":
        src_trainer = str(src_cfg.get("trainer", "unknown"))
    elif src_class == "UnifloralEnsembleMLP":
        src_trainer = "unifloral"
    else:
        src_trainer = src_class.lower() or "unknown"
    parent_lineage = src.get("finetune_lineage") or src_trainer
    return {
        "is_finetune": True,
        "finetuned_from_trainer": src_trainer,
        "finetuned_from_group": src.get("wm_group"),
        "finetuned_from_seed": src.get("seed"),
        "finetuned_from_steps": src.get("update_steps_completed"),
        "finetune_lineage": f"{parent_lineage}->{_world_model_type(cfg)}",
    }


def _define_step_metrics() -> None:
    """Give each stage its own x-axis. In a ``stage=all`` run the world-model and
    policy stages share one W&B run but their step counts differ by orders of
    magnitude (eggroll update steps reach millions; policy updates are far fewer).
    Custom step metrics keep each stage on its own axis so neither forces W&B's
    single global step — without which the later, lower-stepped stage's data is
    rejected as non-monotonic and silently dropped.

    World-model loss defaults to transitions seen (data efficiency) so backprop and
    eggroll runs overlay on a comparable axis rather than the raw update step.
    """
    wandb.define_metric("world_model/transitions_seen")
    for metric in ("world_model/val_mse", "world_model/val_mse_elite", "world_model/train_loss"):
        wandb.define_metric(metric, step_metric="world_model/transitions_seen")
    # Phase-2 (trajectory fine-tuning) gets its own generation axis: in a combined
    # backprop->trajectory run both phases share the W&B run, and Phase-2's generation
    # counter restarts near zero, which would be dropped against the Phase-1 axis.
    wandb.define_metric("world_model_ft/generation")
    wandb.define_metric("world_model_ft/*", step_metric="world_model_ft/generation")
    wandb.define_metric("policy/step")
    wandb.define_metric("policy/*", step_metric="policy/step")


def _is_sweep_run() -> bool:
    """Return True when running under a W&B sweep agent."""
    if os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID"):
        return True
    return wandb.run is not None and getattr(wandb.run, "sweep_id", None) is not None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class Logger:
    """Thin wrapper around W&B. Passed into each experiment stage."""

    def __init__(
        self,
        cfg: DictConfig,
        wm_group: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.wm_group: str = wm_group or make_wm_group(cfg, ts)

        # Fine-tune lineage (e.g. "backprop->eggroll"); recorded in the saved checkpoint
        # so a subsequent fine-tune can chain it. None for from-scratch runs.
        provenance = _finetune_provenance(cfg)
        self.finetune_lineage: str | None = provenance.get("finetune_lineage")

        wandb_cfg = cfg.get("wandb", {})
        self.enabled: bool = wandb_cfg.get("enabled", False)  # type: ignore[union-attr]
        if self.enabled:
            name = wandb_cfg.get("name", None) or _auto_name(cfg, ts)  # type: ignore[union-attr]
            config_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            config_dict.update(_legend_fields(cfg))
            config_dict.update(provenance)
            wandb.init(
                project="model-based-eggroll",
                entity=wandb_cfg.get("entity", "model-based-eggroll"),  # type: ignore[union-attr]
                group=self.wm_group,
                job_type=cfg.get("stage", "all"),
                name=name,
                tags=auto_tags(cfg),
                config=config_dict,
            )
            _define_step_metrics()

    @classmethod
    def from_existing_run(cls, cfg: DictConfig, wm_group: str | None = None) -> "Logger":
        """Create a Logger that attaches to an already-initialised W&B run.

        Used by sweep scripts where wandb.init() is called before Logger creation.
        Applies the same legend fields, fine-tune provenance, and default step
        metric as ``__init__`` so sweep and fine-tune-sweep runs are not missing
        them on the agent-initialised run.
        """
        instance = cls.__new__(cls)
        instance.enabled = True
        instance.wm_group = wm_group or ""
        provenance = _finetune_provenance(cfg)
        instance.finetune_lineage = provenance.get("finetune_lineage")
        if wandb.run is not None:
            # allow_val_change: legend keys (lr, population_size, ...) collide with
            # the flat swept-param keys the sweep agent already set on the run.
            wandb.config.update(
                {**_legend_fields(cfg), **provenance}, allow_val_change=True
            )
            _define_step_metrics()
        return instance

    def finish(self) -> None:
        if not self.enabled:
            return
        wandb.finish()

    def set_crashed_tag(self) -> None:
        """Add 'crashed' tag to the current W&B run. No-op when disabled. Idempotent."""
        if not self.enabled or wandb.run is None:
            return
        tags = tuple(wandb.run.tags or ())
        if "crashed" not in tags:
            wandb.run.tags = (*tags, "crashed")

    def log_world_model_step(self, step: int, **metrics: float) -> None:
        """Log per-update-step world model metrics (losses plus optional work counters)."""
        if not self.enabled:
            return
        wandb.log({f"world_model/{k}": v for k, v in metrics.items()}, step=step)

    def log_world_model_finetune_step(self, generation: int, **metrics: float) -> None:
        """Log Phase-2 (trajectory EGGROLL) metrics on the ``world_model_ft`` axis.

        Logs ``generation`` as the ``world_model_ft/generation`` field (its custom step
        metric) rather than W&B's global step, so in a combined backprop->trajectory run
        Phase-2's generation count (restarting near zero) is not rejected as non-monotonic
        against the Phase-1 axis.
        """
        if not self.enabled:
            return
        wandb.log(
            {f"world_model_ft/{k}": v for k, v in metrics.items()}
            | {"world_model_ft/generation": generation}
        )

    def log_policy_step(self, step: int, **metrics: float) -> None:
        """Log policy training metrics (return, entropy, critic loss, etc.).

        Logs ``step`` as the ``policy/step`` field (its custom step metric) rather
        than forcing W&B's global step — so in a ``stage=all`` run the policy step
        count (far smaller than eggroll's update step) is not rejected as
        non-monotonic against the world-model stage's elevated global step.
        """
        if not self.enabled:
            return
        wandb.log({f"policy/{k}": v for k, v in metrics.items()} | {"policy/step": step})

    def log_eval(self, dataset_id: str, raw_score: float, normalized_score: float) -> None:
        """Log final evaluation results."""
        if not self.enabled:
            return
        wandb.log(
            {
                "eval/raw_score": raw_score,
                "eval/normalized_score": normalized_score,
                "eval/dataset_id": dataset_id,
            }
        )

    def log_wm_eval(self, train_dataset_id: str, eval_dataset_id: str, val_mse: float) -> None:
        """Log a world-model validation MSE on an arbitrary eval dataset."""
        if not self.enabled:
            return
        wandb.log(
            {
                "wm_eval/val_mse": val_mse,
                "wm_eval/train_dataset_id": train_dataset_id,
                "wm_eval/eval_dataset_id": eval_dataset_id,
            }
        )
