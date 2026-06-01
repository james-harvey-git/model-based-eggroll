#!/bin/bash
# SLURM array job for the backprop EnsembleMLP SGD learning-rate W&B sweep on the
# EIT CDT cluster. Each trial trains a backprop ensemble from scratch with plain
# SGD and sweeps only the learning rate; all other knobs are pinned to the
# reference config.
#
# Default sweep config:
#   configs/sweep/ensemble_mlp_backprop_halfcheetah_sgd_lr.yaml
#   32 trials (run_cap), single lr axis, no early termination
#
# Usage:
#   # 1. Register the sweep once from the repo root
#   uv run python scripts/init_sweep.py \
#       configs/sweep/ensemble_mlp_backprop_halfcheetah_sgd_lr.yaml
#   # → prints: Sweep ID: <SWEEP_ID>
#
#   # 2. Submit one array task per GPU you want to use total.
#   #    Example: 4 GPUs total, 12 runs packed onto each GPU.
#   RUNS_PER_GPU=12 SWEEP_ID=<SWEEP_ID> \
#       sbatch --array=0-3 scripts/run_ensemble_mlp_backprop_sgd_lr_sweep_eit.slurm

# ── Job metadata ───────────────────────────────────────────────────────────────
#SBATCH --job-name=ensemble-mlp-backprop-sgd-lr-sweep
#SBATCH --partition=compute
#SBATCH --time=08:00:00

# ── Resources ──────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# ── Logging ────────────────────────────────────────────────────────────────────
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

RUNS_PER_GPU="${RUNS_PER_GPU:-12}"

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="${HOME}/model-based-eggroll"
MINARI_CACHE="${HOME}/minari_cache"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${MINARI_CACHE}"

cd "${REPO_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"
export MINARI_DATASETS_PATH="${MINARI_CACHE}"

: "${SWEEP_ID:?SWEEP_ID must be set — run init_sweep.py to get it}"
export SWEEP_ID

# ── JAX memory settings ────────────────────────────────────────────────────────
# Disable JAX preallocation so multiple processes can share the single GPU
# allocated to this array task. Cap each process to an equal slice of ~90%.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$(awk -v runs="${RUNS_PER_GPU}" 'BEGIN { printf "%.4f", 0.90 / runs }')"

# ── Diagnostics ────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job:      ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:     $(hostname)"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Sweep:    ${SWEEP_ID}"
echo "Runs/GPU: ${RUNS_PER_GPU}"
echo "Dir:      $(pwd)"
nvidia-smi
echo "============================================"

# ── Launch parallel sweep trials ───────────────────────────────────────────────
# Launch RUNS_PER_GPU agents on the single GPU allocated to this array task.
# Across the array, total concurrent runs = RUNS_PER_GPU × number of array tasks.
for i in $(seq 1 "${RUNS_PER_GPU}"); do
    uv run wandb agent "model-based-eggroll/model-based-eggroll/${SWEEP_ID}" &
done

wait
echo "All ${RUNS_PER_GPU} sweep agents on task ${SLURM_ARRAY_TASK_ID:-0} finished."
