"""Training-telemetry figures from W&B (grad norm, clip fraction, wall-clock).

These quantities are *training-time* dynamics, not recomputable from a checkpoint
(unlike scripts/report_plots.py). They are read from W&B, where the trajectory
fine-tuner logs them on the ``world_model_ft`` axis:
  - ``world_model_ft/grad_norm``     — BPTT pre-clip global grad norm (per epoch)
  - ``world_model_ft/clip_fraction`` — fraction of the epoch's steps that clipped
  - ``world_model_ft/wall_time_sec`` — cumulative fine-tune wall-clock (logged from
                                       Phase-2 start, so its max excludes Phase-1).

Plots:
  - grad_norm : mean ± std band over BPTT seeds, log-y (grad norms span orders of mag)
  - clip_frac : mean over BPTT seeds (no band — it sits at ~1.0)
  - walltime  : Phase-2 wall-clock bar, one bar per configured group (e.g. BPTT T=5,
                BPTT T=50, EGGROLL T=5); multi-seed groups get an error bar

Run where W&B auth is available (the cluster, or anywhere with WANDB_API_KEY):
    uv run python scripts/training_plots.py --config scripts/training_plots.yaml
    uv run python scripts/training_plots.py --config scripts/training_plots.yaml --list-runs bptt
    uv run python scripts/training_plots.py --config scripts/training_plots.yaml --only walltime
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from omegaconf import OmegaConf

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

_FT_GEN = "world_model_ft/generation"
_WALL = "world_model_ft/wall_time_sec"


# ── W&B fetch (not unit-tested; needs auth) ──────────────────────────────────────
def _api():
    import wandb  # imported lazily so --list-runs help works without auth issues

    return wandb.Api()


def _run(entity: str, project: str, run_id: str):
    return _api().run(f"{entity}/{project}/{run_id}")


def _ft_history(run, metric: str, samples: int) -> list[dict]:
    """Sampled ``world_model_ft`` rows for one run, sorted by generation.

    Uses W&B's *server-side* sampled ``history(keys=..., samples=...)`` rather than
    ``scan_history`` (which streams the whole run — ~1.25M rows here, since Phase-1
    dominates — and is far too slow). Requesting the ft keys returns only ft-phase rows.
    """
    rows = run.history(keys=[_FT_GEN, metric], samples=samples, pandas=False)
    rows = [r for r in rows if r.get(_FT_GEN) is not None and r.get(metric) is not None]
    rows.sort(key=lambda r: r[_FT_GEN])
    return rows


def fetch_curve(run, metric: str, samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """(generation re-zeroed to start at 0, metric value) for one run's ft phase."""
    rows = _ft_history(run, metric, samples)
    gen = np.array([r[_FT_GEN] for r in rows], dtype=np.float64)
    val = np.array([r[metric] for r in rows], dtype=np.float64)
    return gen - gen.min(), val  # re-zero: a combined run offsets gen by Phase-1 steps


def fetch_walltime_seconds(run) -> float:
    """Phase-2 fine-tune wall-clock = final cumulative ``wall_time_sec`` (excludes Phase-1).

    Read from the run summary (the last logged value — instant); falls back to the max
    over a small sampled history if the summary lacks it.
    """
    v = run.summary.get(_WALL)
    if v is not None:
        return float(v)
    rows = _ft_history(run, _WALL, samples=500)
    return float(max(r[_WALL] for r in rows))


# ── pure aggregation (unit-tested) ───────────────────────────────────────────────
def aggregate_curves(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Align per-seed (gen, val) curves onto a common generation grid and reduce.

    Seeds share a config so their generation grids match up to sampling; each curve is
    linearly interpolated onto the first curve's grid, then reduced to (gen, mean, std).
    ``std`` is None for a single seed.
    """
    ref_gen = curves[0][0]
    stack = np.stack([np.interp(ref_gen, g, v) for g, v in curves], axis=0)  # (S, T)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else None
    return ref_gen, mean, std


# ── plot builders ────────────────────────────────────────────────────────────────
def plot_grad_norm(groups: list[dict], outdir: Path, yscale: str = "log") -> None:
    """Overlaid grad-norm curves. ``groups`` = list of ``{label, color, curves}`` where
    ``curves`` is a list of per-seed ``(gen, val)``; each group → mean line + std band."""
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    for g in groups:
        gen, mean, std = aggregate_curves(g["curves"])
        (line,) = ax.plot(gen, mean, lw=2, color=g.get("color"), label=g["label"])
        if std is not None:
            lo = mean - std
            if yscale == "log":  # keep the band strictly positive for a log axis
                lo = np.maximum(lo, mean * 1e-3)
            ax.fill_between(gen, lo, mean + std, color=line.get_color(), alpha=0.2, lw=0)
    ax.set_yscale(yscale)
    ax.set_xlabel("BPTT epoch")
    ax.set_ylabel("pre-clip grad norm")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("BPTT gradient norm")
    ax.legend()
    _save(fig, outdir, "bptt_grad_norm")


def plot_clip_fraction(curves, outdir: Path) -> None:
    gen, mean, _ = aggregate_curves(curves)
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    ax.plot(gen, mean, lw=2, color="C3", label="BPTT")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("BPTT epoch")
    ax.set_ylabel("clip fraction")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("BPTT gradient-clipping fraction")
    ax.legend()
    _save(fig, outdir, "bptt_clip_fraction")


def plot_walltime_bar(
    groups: list[dict], outdir: Path, units: str = "min", yscale: str = "linear"
) -> None:
    """One bar per group. ``groups`` = list of ``{label, secs: [float], color}``; a group
    with >1 seed gets an error bar (std), a single-seed group gets none."""
    div, unit_label = (60.0, "minutes") if units in ("min", "minutes") else (1.0, "seconds")
    labels = [g["label"] for g in groups]
    arrs = [np.array(g["secs"], dtype=np.float64) / div for g in groups]
    means = [float(a.mean()) for a in arrs]
    errs = [float(a.std(ddof=1)) if a.size > 1 else 0.0 for a in arrs]
    colors = [g.get("color") for g in groups]
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.bar(labels, means, yerr=errs, capsize=6, color=colors, alpha=0.85)
    ax.set_yscale(yscale)  # 'log' makes all bars readable despite large ratios
    ax.set_ylabel(f"Phase-2 fine-tune wall-clock ({unit_label})")
    ax.set_title("Fine-tuning wall-clock time")
    for i, m in enumerate(means):  # annotate each bar with its value
        ax.text(i, m, f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    _save(fig, outdir, "walltime_finetuning")


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir / name}.pdf")


# ── driver ───────────────────────────────────────────────────────────────────────
def _load_config(path: str) -> dict:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(raw, dict)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--only", choices=["gradnorm", "clipfrac", "walltime"], default=None,
        help="run only one of the three figures",
    )
    ap.add_argument(
        "--list-runs", metavar="SUBSTR", default=None,
        help="list project runs whose name contains SUBSTR (to find run ids), then exit",
    )
    args = ap.parse_args()
    cfg = _load_config(args.config)
    entity, project = cfg["wandb"]["entity"], cfg["wandb"]["project"]
    outdir = Path(cfg.get("outdir", "report_figures"))

    if args.list_runs is not None:
        for r in _api().runs(f"{entity}/{project}"):
            if args.list_runs.lower() in r.name.lower() or args.list_runs.lower() in r.id:
                print(f"  {r.id}  {r.name}  [{r.state}]")
        return

    if args.only in (None, "gradnorm"):
        # grad norm overlays one or more configs (e.g. BPTT T=5 vs T=50) to show how the
        # explosion scales with the BPTT horizon.
        groups = []
        for g in cfg.get("grad_norm_groups", []):
            runs = [_run(entity, project, rid) for rid in g["runs"]]
            print(f"[gradnorm] {g['label']}: fetching {len(runs)} runs …")
            curves = [fetch_curve(r, "world_model_ft/grad_norm") for r in runs]
            groups.append({"label": g["label"], "color": g.get("color"), "curves": curves})
        plot_grad_norm(groups, outdir, yscale=cfg.get("grad_norm_yscale", "log"))

    if args.only in (None, "clipfrac"):
        # clip fraction is ~1.0 for every config, so one group (BPTT T=5) suffices.
        clip_ids = list(cfg.get("clip_fraction_runs", []))
        print(f"[clipfrac] fetching {len(clip_ids)} BPTT(T=5) runs …")
        clip_runs = [_run(entity, project, rid) for rid in clip_ids]
        curves = [fetch_curve(r, "world_model_ft/clip_fraction") for r in clip_runs]
        plot_clip_fraction(curves, outdir)

    if args.only in (None, "walltime"):
        bars = list(cfg.get("walltime_bars", []))
        print(f"[walltime] {len(bars)} groups …")
        groups = [
            {
                "label": b["label"],
                "color": b.get("color"),
                "secs": [fetch_walltime_seconds(_run(entity, project, rid)) for rid in b["runs"]],
            }
            for b in bars
        ]
        plot_walltime_bar(
            groups, outdir,
            units=cfg.get("walltime_units", "min"),
            yscale=cfg.get("walltime_yscale", "linear"),
        )


if __name__ == "__main__":
    main()
