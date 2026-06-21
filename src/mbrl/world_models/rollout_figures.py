"""Host-side matplotlib figures for inspecting open-loop world-model rollouts.

These ground the aggregate trajectory-MSE curves with a *visual* check of how a predicted
rollout diverges from the real trajectory: per-feature time-series (does it drift, blow up,
or stay calibrated within the ensemble spread?), a per-feature normalized-error heatmap
(which dimensions drive the compounding error, and when), and — for envs with a registered
layout (HalfCheetah, Hopper, Adroit pen) — genuine phase portraits (position/angle vs velocity).

All functions are pure: they take numpy arrays and return a ``matplotlib.figure.Figure``
(or ``None`` when the plot does not apply), and never touch W&B or global pyplot state.
"""

import math

import matplotlib

matplotlib.use("Agg")  # headless: no display, render straight to image buffers
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Per-environment phase-portrait layouts: (position obs index, velocity obs index, label)
# for the genuine angle/height-vs-velocity planes. Indices follow each env's Gymnasium
# observation layout with exclude_current_positions_from_observation=True, so the excluded
# root x leaves the forward velocity (qvel[0]) with no position counterpart, unpaired.
#
# HalfCheetah: obs = qpos[1:] (8) then qvel (9); a position at index i pairs with its
#   velocity at i + 9. Only the six actuated joints are plotted (root z / angle omitted).
# Hopper: obs = qpos[1:] (5) then qvel (6); a position at index i pairs with its velocity at
#   i + 6. Height (obs[0]) and torso angle (obs[1]) govern falling/termination, so they are
#   plotted alongside the three actuated joints (thigh, leg, foot).
_PHASE_PORTRAIT_LAYOUTS: dict[str, list[tuple[int, int, str]]] = {
    "halfcheetah": [
        (2, 11, "bthigh"),
        (3, 12, "bshin"),
        (4, 13, "bfoot"),
        (5, 14, "fthigh"),
        (6, 15, "fshin"),
        (7, 16, "ffoot"),
    ],
    "hopper": [
        (0, 6, "height"),
        (1, 7, "torso ang"),
        (2, 8, "thigh"),
        (3, 9, "leg"),
        (4, 10, "foot"),
    ],
    # Adroit pen: obs = 24 hand-joint positions (no hand velocities observed), then the pen
    # object — pos obs[24:27], linear vel obs[27:30], angular vel obs[30:33], orientation
    # obs[33:36]. So the only genuine phase planes are the pen's pos-vs-linear-vel and
    # orientation-vs-angular-vel; the latter is the task-relevant one (the goal is to reorient
    # the pen). Keyed on "pen/" (not "pen") so it cannot match invertedpendulum ids.
    "pen/": [
        (24, 27, "pen x"),
        (25, 28, "pen y"),
        (26, 29, "pen z"),
        (33, 30, "orien x"),
        (34, 31, "orien y"),
        (35, 32, "orien z"),
    ],
}


def _feature_labels(n_features: int) -> list[str]:
    """Generic feature names: the last column is the reward, the rest are state dims."""
    return [f"obs[{i}]" for i in range(n_features - 1)] + ["reward"]


def select_window_indices(per_window_mse: np.ndarray) -> dict[str, int]:
    """Best / median / worst window indices by per-window rollout MSE (deterministic)."""
    order = np.argsort(np.asarray(per_window_mse))
    return {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }


def plot_rollout_timeseries(
    true_traj: np.ndarray,
    pred_mean: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    title: str = "rollout",
) -> Figure:
    """Per-feature true-vs-predicted time-series for one window, with the ensemble band.

    ``true_traj``/``pred_mean``/``pred_lo``/``pred_hi`` are ``(T, D)``. One subplot per
    feature: the real trajectory, the ensemble-mean prediction, and the shaded ensemble
    min/max spread (whether the truth stays inside it is the calibration check).
    """
    horizon, n_features = pred_mean.shape
    steps = np.arange(1, horizon + 1)
    labels = _feature_labels(n_features)
    ncols = min(4, n_features)
    nrows = math.ceil(n_features / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), squeeze=False)
    for f in range(n_features):
        ax = axes[f // ncols][f % ncols]
        ax.fill_between(steps, pred_lo[:, f], pred_hi[:, f], alpha=0.25, color="tab:blue",
                        label="ensemble spread")
        ax.plot(steps, pred_mean[:, f], color="tab:blue", label="predicted")
        ax.plot(steps, true_traj[:, f], color="tab:orange", ls="--", label="true")
        ax.set_title(labels[f], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # integer rollout steps only
    for k in range(n_features, nrows * ncols):  # hide unused cells
        axes[k // ncols][k % ncols].axis("off")
    axes[0][0].legend(fontsize=6, loc="best")
    fig.suptitle(title)
    fig.supxlabel("rollout step")
    fig.tight_layout()
    return fig


def plot_error_heatmap(per_feature_nmse: np.ndarray, title: str = "rollout NMSE") -> Figure:
    """Per-feature × horizon normalized-error heatmap.

    ``per_feature_nmse`` is ``(T, D)`` — per-step, per-feature MSE divided by that feature's
    target variance (1.0 = no better than predicting that feature's mean). Rows = features,
    columns = rollout step.
    """
    horizon, n_features = per_feature_nmse.shape
    labels = _feature_labels(n_features)
    fig, ax = plt.subplots(figsize=(0.5 * horizon + 2.5, 0.32 * n_features + 1.5))
    im = ax.imshow(per_feature_nmse.T, aspect="auto", origin="lower", cmap="magma",
                   vmin=0.0, vmax=max(1.0, float(np.nanmax(per_feature_nmse))))
    ax.set_xticks(range(horizon))
    ax.set_xticklabels([str(i) for i in range(1, horizon + 1)], fontsize=7)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("rollout step")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="NMSE (per-feature)")
    fig.tight_layout()
    return fig


def plot_joint_phase_portraits(
    true_traj: np.ndarray, pred_mean: np.ndarray, dataset_id: str, title: str = "phase portraits"
) -> Figure | None:
    """Angle/height-vs-velocity phase portraits for one window.

    Looks up the env-specific obs-index pairing in ``_PHASE_PORTRAIT_LAYOUTS`` (keyed by a
    substring of ``dataset_id``). Returns ``None`` (and prints a one-line note) for any dataset
    without a registered layout, since the pairing is environment-specific.
    ``true_traj``/``pred_mean`` are ``(T, D)``.
    """
    pairs = next(
        (p for k, p in _PHASE_PORTRAIT_LAYOUTS.items() if k in dataset_id.lower()), None
    )
    if pairs is None:
        print(f"rollout_figures: phase portraits skipped (no layout for): {dataset_id}")
        return None
    needed = max(max(a, v) for a, v, _ in pairs)
    if true_traj.shape[1] <= needed:  # obs smaller than the assumed halfcheetah layout
        print(
            f"rollout_figures: phase portraits skipped (obs has {true_traj.shape[1]} "
            f"features, need > {needed}): {dataset_id}"
        )
        return None
    ncols = 3
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    for i, (ang, vel, name) in enumerate(pairs):
        ax = axes[i // ncols][i % ncols]
        ax.plot(true_traj[:, ang], true_traj[:, vel], color="tab:orange", ls="--",
                marker="o", ms=2, label="true")
        ax.plot(pred_mean[:, ang], pred_mean[:, vel], color="tab:blue",
                marker="o", ms=2, label="predicted")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("angle", fontsize=7)
        ax.set_ylabel("ang. vel.", fontsize=7)
        ax.tick_params(labelsize=7)
    for k in range(len(pairs), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    axes[0][0].legend(fontsize=6, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
