#!/bin/bash
# SLURM array job for running the MOPO W&B hyperparameter sweep on the EIT CDT cluster.
#
# Usage:
#   # 1. Train world model (one-off, if not already done)
#   sbatch scripts/run_pipeline_eit.slurm stage=world_model
#
#   # 2. Register the sweep with W&B (run once from login node)
#   uv run python scripts/init_sweep.py
#   # → prints: Sweep ID: <SWEEP_ID>
#
#   # 3. Submit the sweep array
#   SWEEP_ID=<SWEEP_ID> WM_CHECKPOINT=$HOME/model-based-eggroll/checkpoints/world_model.pkl \
#       sbatch scripts/run_sweep_eit.slurm
#
# Each SLURM task claims one sweep config from W&B and runs it to completion.
# Submit from the model-based-eggroll/ directory.

# ── Job metadata ───────────────────────────────────────────────────────────────
#SBATCH --job-name=mopo-sweep
#SBATCH --partition=compute
#SBATCH --time=08:00:00

# ── Resources ──────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       # JAX data loading + gymnasium vectorized eval workers
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# ── Array ──────────────────────────────────────────────────────────────────────
#SBATCH --array=0-35             # 36 tasks = 4 penalty_coeff × 3 lr × 3 rollout_length

# ── Logging ────────────────────────────────────────────────────────────────────
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="${HOME}/model-based-eggroll"
MINARI_CACHE="${HOME}/minari_cache"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${MINARI_CACHE}"

cd "${REPO_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"
export MINARI_DATASETS_PATH="${MINARI_CACHE}"

# Fail early with a clear message if required env vars are not set
: "${SWEEP_ID:?SWEEP_ID must be set — run init_sweep.py to get it}"
: "${WM_CHECKPOINT:?WM_CHECKPOINT must be set to the path of world_model.pkl}"
export WM_CHECKPOINT

# ── Diagnostics ────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job:      ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node:     $(hostname)"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Sweep:    ${SWEEP_ID}"
echo "WM ckpt:  ${WM_CHECKPOINT}"
echo "Dir:      $(pwd)"
nvidia-smi
echo "============================================"

# ── Run one sweep trial ────────────────────────────────────────────────────────
# Each task claims exactly one config from the W&B sweep queue and runs it.
uv run wandb agent --count 1 "model-based-eggroll/${SWEEP_ID}"
