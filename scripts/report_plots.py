"""Report figures for trajectory world-model fine-tuning.

Recomputes everything from saved checkpoints (it does NOT scrape W&B), so the same
script produces every panel reproducibly and on any dataset — including the
cross-policy case (a model fine-tuned on *medium* evaluated on *expert*), which the
training runs never logged.

It reuses the existing eval methods on ``EnsembleMLP``:
- ``compute_traj_mse(episodes, horizon) -> (all_mean, elite_mean, curve)`` for the
  per-step compounding-error curve (the cheap, curve-only path),
- ``compute_traj_grounding(...)`` / ``build_rollout_figures(...)`` for the heatmap,
  phase-portrait and time-series figures.

By default (``eval_on: own_val``) each checkpoint is scored on its *own* recorded
held-out trajectory split, so independent seeds use different val episodes (the
generalization error bar) and a fine-tuned model is never scored on episodes it trained
on. ``eval_on: val`` instead forces one common split across all models — only correct
when every run shared that split, otherwise it leaks training episodes into the eval.

Run on the cluster (where the fine-tune checkpoints and Minari datasets live):

    uv run python scripts/report_plots.py --config scripts/report_plots.yaml
    uv run python scripts/report_plots.py --config scripts/report_plots.yaml --list
    uv run python scripts/report_plots.py --config scripts/report_plots.yaml -s halfcheetah_val
    uv run python scripts/report_plots.py --config scripts/report_plots.yaml --only figures

Fill ``scripts/report_plots.yaml`` with your cluster checkpoint paths (see the
``EXAMPLE_CONFIG`` block at the bottom of this file for the schema).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from omegaconf import OmegaConf

from mbrl.data import derive_episode_train_val_split, load_episodes
from mbrl.world_models.base import load_world_model_from_checkpoint

# ── plotting style ──────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    }
)


# ── episode caches: load each Minari dataset / val-split once ────────────────────
_EP_CACHE: dict = {}


def _episodes_for(dataset_id: str):
    if dataset_id not in _EP_CACHE:
        _EP_CACHE[dataset_id] = load_episodes(dataset_id)
    return _EP_CACHE[dataset_id]


def _resolve_eval_eps(dataset_id: str, on: str, wm, val_split: float, split_seed: int):
    """The episode set ONE checkpoint is scored on (resolved per checkpoint).

    - ``on='own_val'`` (default, correct for independent seeds): each checkpoint is
      evaluated on its *own* recorded held-out trajectory split (fraction + seed read
      off the checkpoint). Different seeds → different val episodes by design, and a
      fine-tuned model is never scored on episodes it trained on (no leakage). Phase-1
      checkpoints don't record a trajectory split, so they fall back to the config
      ``val_split`` fraction with the checkpoint's own ``seed`` (reproducing the split
      their fine-tune descendant would have used).
    - ``on='val'``: a single forced common split (``val_split`` + ``split_seed``). ONLY
      correct when every model genuinely shared that split (Design A — split fixed,
      training seed varied). If the models were trained under different split seeds this
      leaks their training episodes into the eval — see build_curve_plot's warning.
    - ``on='all'``: the whole dataset (cross-policy / OOD transfer, where the eval set is
      wholly unseen relative to the trained model so a split is meaningless).
    """
    episodes, info = _episodes_for(dataset_id)
    if on == "all":
        return episodes, info
    if on == "own_val":
        frac = getattr(wm, "_trajectory_validation_split", None)
        frac = float(frac) if frac is not None else float(val_split)  # Phase-1 fallback
        seed = getattr(wm, "_seed", None)
        seed = int(seed) if seed is not None else int(split_seed)
        _, val_eps = derive_episode_train_val_split(episodes, frac, seed)
        return val_eps, info
    # on == "val": forced common split, shared across every checkpoint.
    _, val_eps = derive_episode_train_val_split(episodes, val_split, split_seed)
    return val_eps, info


# ── core: per-checkpoint curve ──────────────────────────────────────────────────
def _curve_for_checkpoint(
    ckpt_path: str, dataset_id: str, on: str, horizon: int,
    val_split: float, split_seed: int,
) -> np.ndarray:
    """All-member open-loop rollout MSE curve (T,) for one checkpoint, evaluated on the
    episode set resolved for it by ``on`` (per-checkpoint own split by default)."""
    wm = load_world_model_from_checkpoint(ckpt_path)
    compute = getattr(wm, "compute_traj_mse", None)
    if compute is None:
        raise TypeError(
            f"{Path(ckpt_path).name}: world-model class {type(wm).__name__} has no "
            "compute_traj_mse (only EnsembleMLP supports trajectory rollout eval)."
        )
    eval_eps, info = _resolve_eval_eps(dataset_id, on, wm, val_split, split_seed)
    _assert_env_match(ckpt_path, wm, info, dataset_id)
    _, _, curve = compute(eval_eps, horizon)
    jax.block_until_ready(curve)
    return np.asarray(curve, dtype=np.float64)  # (T,)


def _assert_env_match(ckpt_path: str, wm, info, dataset_id: str) -> None:
    """Fail fast (and readably) when a checkpoint's env differs from the eval dataset's.

    Otherwise the obs+action dim mismatch surfaces as a cryptic ``dot_general`` shape
    error deep in the world-model forward. Usually means ``dataset_id`` was left pointing
    at the wrong env (e.g. a HalfCheetah id copy-pasted into a Hopper plot block).
    """
    if (wm.obs_dim, wm.act_dim) != (info.obs_dim, info.act_dim):
        raise ValueError(
            f"env mismatch for {Path(ckpt_path).name}: checkpoint is "
            f"obs/act=({wm.obs_dim},{wm.act_dim}) but dataset_id='{dataset_id}' is "
            f"({info.obs_dim},{info.act_dim}). Set dataset_id to the checkpoint's env."
        )


def _aggregate_seeds(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
    """(mean (T,), std (T,) or None for a single seed) over a list of (T,) curves."""
    stack = np.stack(curves, axis=0)  # (S, T)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else None
    return mean, std


# ── config schema ───────────────────────────────────────────────────────────────
@dataclass
class ModelGroup:
    """One line on a curve plot. ``checkpoints`` may hold several seeds (→ error band)."""

    label: str
    checkpoints: list[str]
    color: str | None = None
    linestyle: str = "-"


@dataclass
class CurvePlot:
    """One overlaid compounding-error figure."""

    name: str
    dataset_id: str  # minari id the curves are evaluated on
    horizon: int
    groups: list[ModelGroup]
    on: str = "own_val"  # 'own_val' (per-ckpt split, default) | 'val' (forced common) | 'all'
    val_split: float = 0.1  # own_val: Phase-1 fallback fraction; val: the common fraction
    split_seed: int = 0  # used only when on='val' (the forced common split seed)
    title: str | None = None
    band: str = "std"  # 'std' | 'sem' | 'none'
    yscale: str = "linear"  # matplotlib y-axis scale: 'linear' (default) | 'log' | 'symlog'


# ── plot builders ────────────────────────────────────────────────────────────────
def _draw_curves(spec: CurvePlot, ax, ylabel: bool = True, legend: bool = True) -> None:
    """Draw one CurvePlot's overlaid curves onto ``ax`` (shared by the single-figure and
    multi-panel-grid builders)."""
    n_ckpts = sum(len(g.checkpoints) for g in spec.groups)
    if spec.on == "val" and n_ckpts > 1:
        print(
            f"  WARNING [{spec.name}]: on='val' forces one common split across "
            f"{n_ckpts} checkpoints. This is only correct if every run shared that "
            "split (split seed fixed, training seed varied). If seeds used different "
            "splits, a fine-tuned model gets scored on episodes it trained on "
            "(leakage) — use on='own_val' instead."
        )
    for g in spec.groups:
        curves = [
            _curve_for_checkpoint(
                p, spec.dataset_id, spec.on, spec.horizon, spec.val_split, spec.split_seed
            )
            for p in g.checkpoints
        ]
        mean, std = _aggregate_seeds(curves)
        # Exact final-horizon-step numbers for the report (mean ± std over seeds; just the
        # value for a single-seed curve, where std is undefined).
        t_final = mean.shape[0]
        if std is not None:
            print(
                f"    [{spec.name}] {g.label}: final-step (t={t_final}) MSE = "
                f"{mean[-1]:.6g} ± {std[-1]:.6g}  (std over n={len(curves)} seeds)"
            )
        else:
            print(
                f"    [{spec.name}] {g.label}: final-step (t={t_final}) MSE = "
                f"{mean[-1]:.6g}  (single seed)"
            )
        steps = np.arange(1, mean.shape[0] + 1)
        (line,) = ax.plot(
            steps, mean, label=g.label, color=g.color, linestyle=g.linestyle, lw=2
        )
        if std is not None and spec.band != "none":
            band = std / np.sqrt(len(curves)) if spec.band == "sem" else std
            ax.fill_between(
                steps, mean - band, mean + band, color=line.get_color(), alpha=0.2, lw=0
            )
    ax.set_xlabel("rollout step $t$")
    if ylabel:
        ax.set_ylabel("trajectory MSE")
    # Rollout steps are integers; suppress matplotlib's fractional (1.5, 2.5, …) ticks.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale(spec.yscale)  # 'linear' by default; 'log' for wide-range compounding
    ax.set_title(spec.title or spec.name)
    if legend:
        ax.legend()


def build_curve_plot(spec: CurvePlot, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    _draw_curves(spec, ax)
    _save(fig, outdir, spec.name)


def build_curve_grid(name: str, panels: list[CurvePlot], outdir: Path) -> None:
    """Horizontally-stacked 1×N figure: one CurvePlot panel per column (e.g. one env each).

    Panels keep independent y-axes (envs differ in MSE scale); the 'trajectory MSE' label
    and legend are drawn once (leftmost panel) since the groups are shared across panels.
    """
    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(4.8 * n, 4.0), squeeze=False, constrained_layout=True
    )
    for i, panel in enumerate(panels):
        _draw_curves(panel, axes[0][i], ylabel=(i == 0), legend=(i == 0))
    _save(fig, outdir, name)


def dump_rollout_figures(
    ckpt_path: str, dataset_id: str, horizon: int, outdir: Path, on: str = "own_val",
    val_split: float = 0.1, split_seed: int = 0, prefix: str | None = None,
) -> None:
    """Heatmap / phase-portrait / time-series figures for one checkpoint (built by the
    world model itself via compute_traj_grounding) — saved to ``outdir``."""
    wm = load_world_model_from_checkpoint(ckpt_path)
    compute = getattr(wm, "compute_traj_grounding", None)
    if compute is None:
        print(f"  skip {Path(ckpt_path).name}: no compute_traj_grounding")
        return
    eval_eps, info = _resolve_eval_eps(dataset_id, on, wm, val_split, split_seed)
    _assert_env_match(ckpt_path, wm, info, dataset_id)
    *_, figures = compute(eval_eps, horizon, info.dataset_id)
    tag = prefix or Path(ckpt_path).parent.name
    for name, figure in figures.items():
        _save(figure, outdir, f"{tag}__{name}")
        plt.close(figure)


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):  # pdf for LaTeX (vector), png for quick preview
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir / name}.pdf")


# ── config loading ───────────────────────────────────────────────────────────────
def _parse_curve_spec(c: dict, default_name: str = "panel") -> CurvePlot:
    return CurvePlot(
        name=c.get("name", c.get("title", default_name)),
        dataset_id=c["dataset_id"],
        horizon=int(c["horizon"]),
        # 'eval_on' not 'on': YAML 1.1 parses a bare 'on' key as boolean True.
        on=c.get("eval_on", "own_val"),
        val_split=float(c.get("val_split", 0.1)),
        split_seed=int(c.get("split_seed", 0)),
        title=c.get("title"),
        band=c.get("band", "std"),
        yscale=c.get("yscale", "linear"),
        groups=[ModelGroup(**g) for g in c["groups"]],
    )


def _load_config(
    path: str,
) -> tuple[list[CurvePlot], list[tuple[str, list[CurvePlot]]], list[dict], Path]:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(raw, dict)
    outdir = Path(raw.get("outdir", "report_figures"))
    curve_specs = [_parse_curve_spec(c) for c in raw.get("curve_plots", [])]
    # Each grid is (name, [panel CurvePlot]); panels render side by side in one figure.
    grid_specs = [
        (
            g["name"],
            [_parse_curve_spec(p, f"{g['name']}_{i}") for i, p in enumerate(g["panels"])],
        )
        for g in raw.get("curve_grids", [])
    ]
    figure_specs = list(raw.get("rollout_figures", []))
    return curve_specs, grid_specs, figure_specs, outdir


def _figure_name(spec: dict) -> str:
    """Identifier used to ``--select`` a rollout-figure entry (its prefix, else the
    checkpoint's parent dir)."""
    return spec.get("prefix") or Path(spec["checkpoint"]).parent.name


def _selected(name: str, patterns: list[str] | None) -> bool:
    """True if no filter is given, or ``name`` contains any of the substring patterns."""
    return patterns is None or any(p in name for p in patterns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML config (see EXAMPLE_CONFIG)")
    ap.add_argument(
        "--only", choices=["validation", "figures"], default=None,
        help="run only curve plots, or only rollout-figure dumps",
    )
    ap.add_argument(
        "--select", "-s", default=None,
        help="comma-separated name substrings; only matching curve plots (by name) / "
        "rollout figures (by prefix) are run. e.g. -s halfcheetah_val,transfer",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="list the curve-plot and rollout-figure names in the config, then exit",
    )
    args = ap.parse_args()

    curve_specs, grid_specs, figure_specs, outdir = _load_config(args.config)

    if args.list:
        print("curve_plots:")
        for spec in curve_specs:
            print(f"  {spec.name}")
        print("curve_grids:")
        for gname, panels in grid_specs:
            print(f"  {gname}  ({', '.join(p.dataset_id for p in panels)})")
        print("rollout_figures:")
        for spec in figure_specs:
            print(f"  {_figure_name(spec)}")
        return

    patterns = [p.strip() for p in args.select.split(",")] if args.select else None
    ran = 0

    if args.only != "figures":
        for spec in curve_specs:
            if not _selected(spec.name, patterns):
                continue
            print(f"[curve] {spec.name}  ({spec.dataset_id}, h={spec.horizon}, on={spec.on})")
            build_curve_plot(spec, outdir)
            ran += 1
        for gname, panels in grid_specs:
            if not _selected(gname, patterns):
                continue
            print(f"[grid] {gname}  ({len(panels)} panels)")
            build_curve_grid(gname, panels, outdir)
            ran += 1

    if args.only != "validation":
        for spec in figure_specs:
            if not _selected(_figure_name(spec), patterns):
                continue
            print(f"[figures] {spec['checkpoint']}")
            dump_rollout_figures(
                spec["checkpoint"], spec["dataset_id"], int(spec["horizon"]),
                outdir, on=spec.get("eval_on", "own_val"),
                val_split=float(spec.get("val_split", 0.1)),
                split_seed=int(spec.get("split_seed", 0)),
                prefix=spec.get("prefix"),
            )
            ran += 1

    if ran == 0:
        print(
            f"Nothing matched (select={args.select!r}, only={args.only!r}). "
            "Run with --list to see available names."
        )


# ── example config (copy to scripts/report_plots.yaml and fill in cluster paths) ──
EXAMPLE_CONFIG = """
outdir: report_figures

curve_plots:
  # Plot 1: validation trajectory MSE — shared unfine-tuned checkpoint + fine-tuned models.
  - name: halfcheetah_val_traj_mse
    dataset_id: mujoco/halfcheetah/medium-v0
    horizon: 5                 # eggroll only has T=5 ckpts → compare BPTT at T=5
    eval_on: own_val                # each ckpt on its OWN held-out split (independent seeds)
    val_split: 0.1             # only the Phase-1 fallback fraction (Phase-1 records no traj split)
    title: HalfCheetah medium — validation rollout MSE
    band: std
    groups:
      - label: unfine-tuned (Phase 1)      # several seeds → error band
        checkpoints:
          - /cluster/.../phase1_s0/world_model.pkl
          - /cluster/.../phase1_s1/world_model.pkl
          - /cluster/.../phase1_s2/world_model.pkl
        color: gray
        linestyle: ":"
      - label: BPTT (T=5)                  # several seeds → error band
        checkpoints:
          - /cluster/.../bptt_t5_s0/world_model.pkl
          - /cluster/.../bptt_t5_s1/world_model.pkl
          - /cluster/.../bptt_t5_s2/world_model.pkl
        color: C0
      - label: EGGROLL (T=5)               # single seed → no band
        checkpoints:
          - /cluster/.../eggroll_t5_s0/world_model.pkl
        color: C1

  # Plot 2: cross-policy transfer — medium-fine-tuned models evaluated on EXPERT data.
  - name: halfcheetah_transfer_to_expert
    dataset_id: mujoco/halfcheetah/expert-v0
    horizon: 5
    eval_on: all                    # expert is OOD wrt the medium-trained model → whole set
    title: HalfCheetah — transfer to expert (rollout MSE)
    groups:
      - label: BPTT (T=5)
        checkpoints: [/cluster/.../bptt_t5_s0/world_model.pkl]
        color: C0
      - label: EGGROLL (T=5)
        checkpoints: [/cluster/.../eggroll_t5_s0/world_model.pkl]
        color: C1

rollout_figures:               # heatmap / phase-portrait / time-series per checkpoint
  - checkpoint: /cluster/.../eggroll_t5_s0/world_model.pkl
    dataset_id: mujoco/halfcheetah/medium-v0
    horizon: 5
    eval_on: val
    val_split: 0.1
    split_seed: 0
    prefix: eggroll_t5
"""

if __name__ == "__main__":
    main()
