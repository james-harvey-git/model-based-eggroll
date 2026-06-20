"""World model training experiment."""

import math
from pathlib import Path
import pickle
import time

from hydra.utils import get_class
import jax
from omegaconf import DictConfig, OmegaConf

from mbrl.data import load_dataset, load_episodes
from mbrl.logger import Logger

# Architecture fields a Phase-2 trajectory fine-tune must inherit from the Phase-1
# checkpoint it warm-starts (otherwise the param shapes can't load).
_ARCH_FIELDS = (
    "num_ensemble", "num_elites", "hidden_dims", "activation", "init_scheme",
    "backbone", "max_logvar_init", "min_logvar_init", "disable_logvar_predictions",
)


def _check_init_ckpt_dataset(init_ckpt_path, info) -> None:
    """Guard a warm-start against a dataset / shape mismatch with the loaded dataset."""
    if init_ckpt_path is None:
        return
    with open(init_ckpt_path, "rb") as f:
        meta = pickle.load(f)
    if meta["dataset_id"] != info.dataset_id:
        raise ValueError(
            f"init_checkpoint dataset_id mismatch: ckpt={meta['dataset_id']!r} "
            f"vs current dataset={info.dataset_id!r}. Refusing to finetune on a "
            "different dataset to the one used for pretraining."
        )
    if meta["obs_dim"] != info.obs_dim or meta["act_dim"] != info.act_dim:
        raise ValueError(
            f"init_checkpoint shape mismatch: ckpt obs/act=({meta['obs_dim']},"
            f" {meta['act_dim']}) vs dataset=({info.obs_dim}, {info.act_dim})."
        )


def _plumb_seed(wm_cfg: DictConfig, seed: int) -> None:
    """Set wm_cfg.seed from the top-level seed when unset (for reproducible splits)."""
    if "seed" in wm_cfg and wm_cfg.seed is None:
        OmegaConf.set_struct(wm_cfg, False)
        wm_cfg.seed = int(seed)
        OmegaConf.set_struct(wm_cfg, True)


def _make_onestep_log_fn(logger: Logger, start_time: float):
    """Positional log fn for the one-step trainers (backprop / eggroll)."""

    def log_fn(
        step: int,
        train_loss: float,
        val_mse: float,
        transitions_seen: int,
        forward_evals: int,
        epoch: int | None = None,
        val_mse_elite: float | None = None,
        lr: float | None = None,
        sigma: float | None = None,
        fitness_std: float | None = None,
    ) -> None:
        metrics: dict[str, float] = {}
        train_loss_f = float(train_loss)
        if math.isfinite(train_loss_f):
            metrics["train_loss"] = train_loss_f
        val_mse_f = float(val_mse)
        if math.isfinite(val_mse_f):
            metrics["val_mse"] = val_mse_f
        if val_mse_elite is not None and math.isfinite(val_mse_elite):
            metrics["val_mse_elite"] = float(val_mse_elite)
        if lr is not None and math.isfinite(lr):
            metrics["lr"] = float(lr)
        if sigma is not None and math.isfinite(sigma):
            metrics["sigma"] = float(sigma)
        if fitness_std is not None and math.isfinite(fitness_std):
            metrics["fitness_std"] = float(fitness_std)
        if epoch is not None:
            metrics["epoch"] = float(epoch)
        metrics["transitions_seen"] = float(transitions_seen)
        metrics["forward_evals"] = float(forward_evals)
        metrics["wall_time_sec"] = time.perf_counter() - start_time
        logger.log_world_model_step(int(step), **metrics)

    return log_fn


def _make_traj_log_fn(logger: Logger, start_time: float):
    """(generation, **metrics) log fn for Phase-2 (its own ``world_model_ft`` axis).

    Reserved keys containing ``_curve`` are routed to line-chart panels rather than
    logged as scalars: a dict value (series name -> per-step list, e.g.
    ``{train,val}_traj_mse_curve`` each overlaying init vs final) becomes one
    line_series panel; a plain list becomes a single-line panel. The reserved
    ``rollout_figures`` key (a dict name -> matplotlib Figure) is logged as images.
    """

    def log_fn(generation: int, **metrics) -> None:
        figures = metrics.pop("rollout_figures", None)
        if figures is not None:
            for name, fig in figures.items():
                logger.log_world_model_finetune_image(name, fig, int(generation))
        for key in [k for k in metrics if "_curve" in k]:
            value = metrics.pop(key)
            if isinstance(value, dict):
                logger.log_world_model_finetune_curves(key, value)
            else:
                logger.log_world_model_finetune_curve(key, value)
        metrics["wall_time_sec"] = time.perf_counter() - start_time
        logger.log_world_model_finetune_step(int(generation), **metrics)

    return log_fn


def _save_checkpoint(
    path: Path, world_model, wm_cfg: DictConfig, info, logger: Logger, lineage
) -> None:
    """Write an EnsembleDynamics checkpoint (class recovered on load from _target_)."""
    common = {
        "obs_dim": info.obs_dim,
        "act_dim": info.act_dim,
        "dataset_id": info.dataset_id,
        "world_model_cfg": OmegaConf.to_container(wm_cfg),
        "wm_group": logger.wm_group,
        "finetune_lineage": lineage,
        # W&B run identity so wm_eval can append its outputs to this training run.
        "wandb_run_id": logger.run_id,
        "wandb_entity": logger.run_entity,
        "wandb_project": logger.run_project,
    }
    checkpoint = {**common, **world_model.checkpoint_state()}
    if not bool(wm_cfg.get("save_opt_state", True)):
        # Roughly halves checkpoint size; only fine-tunes with reset_optax_state=false
        # need the optimiser state.
        checkpoint["opt_state"] = None
    # Pickle numpy leaves, not live jax.Arrays: portable across machines and JAX
    # versions, and never tied to a device buffer.
    checkpoint = jax.tree.map(
        lambda x: jax.device_get(x) if isinstance(x, jax.Array) else x, checkpoint
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def _train_model(wm_cfg: DictConfig, dataset, info, rng, log_fn, episodes=None):
    """Instantiate and train one world model (class via wm_cfg._target_).

    ``episodes`` is passed only for the trajectory trainer; other world-model classes
    (and the ABC) take no such kwarg, so omit it when None.
    """
    world_model = get_class(wm_cfg._target_)(
        info.obs_dim, info.act_dim, info.dataset_id, wm_cfg
    )
    if episodes is None:
        world_model.train(dataset, wm_cfg, rng, log_fn=log_fn)
    else:
        world_model.train(dataset, wm_cfg, rng, log_fn=log_fn, episodes=episodes)
    return world_model


def run(cfg: DictConfig, logger: Logger) -> None:
    """Train a world model and save a checkpoint to cfg.checkpoint_dir.

    Three modes, dispatched from cfg.world_model:
    - **Combined two-phase** (``world_model.finetune=true``): backprop pretrain, then
      trajectory fine-tune (the separate ``finetune_world_model`` config — EGGROLL or BPTT)
      with the Phase-1 checkpoint auto-handed off.
    - **Standalone Phase 2** (``trainer=eggroll_trajectory`` / ``bptt_trajectory``): a single
      trajectory fine-tune from an explicit ``init_checkpoint``.
    - **Single-phase** (otherwise): one backprop or eggroll training run (unchanged).
    """
    rng = jax.random.key(cfg.seed)
    dataset, info = load_dataset(cfg.dataset.name)
    ckpt_dir = Path(cfg.checkpoint_dir)

    if bool(cfg.world_model.get("finetune", False)):
        # ── Phase 1: backprop pretrain ──────────────────────────────────────────
        _plumb_seed(cfg.world_model, cfg.seed)
        _check_init_ckpt_dataset(cfg.world_model.get("init_checkpoint", None), info)
        rng, p1_rng = jax.random.split(rng)
        wm1 = _train_model(
            cfg.world_model, dataset, info, p1_rng,
            _make_onestep_log_fn(logger, time.perf_counter()),
        )
        phase1_path = ckpt_dir / "world_model_phase1.pkl"
        _save_checkpoint(phase1_path, wm1, cfg.world_model, info, logger, logger.finetune_lineage)

        # ── Phase 2: trajectory fine-tune (separate config, auto-handoff) ───────
        ft_cfg = cfg.finetune_world_model
        OmegaConf.set_struct(ft_cfg, False)
        ft_cfg.init_checkpoint = str(phase1_path)
        for k in _ARCH_FIELDS:  # arch must match the Phase-1 checkpoint
            if k in cfg.world_model:
                ft_cfg[k] = cfg.world_model[k]
        OmegaConf.set_struct(ft_cfg, True)
        _plumb_seed(ft_cfg, cfg.seed)

        episodes, _ = load_episodes(cfg.dataset.name)
        rng, p2_rng = jax.random.split(rng)
        wm2 = _train_model(
            ft_cfg, dataset, info, p2_rng,
            _make_traj_log_fn(logger, time.perf_counter()), episodes=episodes,
        )
        if ft_cfg.get("precompute_term_stats", False):
            rng, stats_rng = jax.random.split(rng)
            print("Precomputing MoReL term stats (discrepancy, min_r)...")
            wm2.precompute_term_stats(dataset, stats_rng)
            print(f"  discrepancy={wm2.discrepancy:.6f}  min_r={wm2.min_r:.6f}")
        lineage = f"{cfg.world_model.trainer}->{ft_cfg.trainer}"
        _save_checkpoint(ckpt_dir / "world_model.pkl", wm2, ft_cfg, info, logger, lineage)
        return

    # ── Single-phase (backprop, eggroll, or standalone eggroll_trajectory) ──────
    _plumb_seed(cfg.world_model, cfg.seed)
    _check_init_ckpt_dataset(cfg.world_model.get("init_checkpoint", None), info)
    is_traj = str(cfg.world_model.get("trainer", "")) in (
        "eggroll_trajectory", "bptt_trajectory"
    )
    rng, train_rng = jax.random.split(rng)
    if is_traj:
        episodes, _ = load_episodes(cfg.dataset.name)
        log_fn = _make_traj_log_fn(logger, time.perf_counter())
    else:
        episodes, log_fn = None, _make_onestep_log_fn(logger, time.perf_counter())
    world_model = _train_model(cfg.world_model, dataset, info, train_rng, log_fn, episodes)

    # Precompute MoReL's halt-penalty statistics (discrepancy, min_r) if requested.
    # Opt-in and expensive (O(N^2)); only needed when this model feeds MoReL.
    if cfg.world_model.get("precompute_term_stats", False):
        rng, stats_rng = jax.random.split(rng)
        print("Precomputing MoReL term stats (discrepancy, min_r)...")
        world_model.precompute_term_stats(dataset, stats_rng)
        print(f"  discrepancy={world_model.discrepancy:.6f}  min_r={world_model.min_r:.6f}")

    _save_checkpoint(
        ckpt_dir / "world_model.pkl", world_model, cfg.world_model, info, logger,
        logger.finetune_lineage,
    )
