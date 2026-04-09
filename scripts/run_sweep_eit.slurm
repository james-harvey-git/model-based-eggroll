#!/bin/bash
# SLURM array job for running the MOPO W&B hyperparameter sweep on the EIT CDT cluster.
#
# Runs RUNS_PER_GPU sweep trials in parallel on each GPU. With 80GB H100s and
# ~8-10GB per MOPO run, 6 parallel processes per GPU is safe.
#
# Usage:
#   # 1. Train world model (one-off, if not already done)
#   sbatch scripts/run_pipeline_eit.slurm stage=world_model
#
#   # 2. Register the sweep with W&B (run once from login node)
#   uv run python scripts/init_sweep.py
#   # → prints: Sweep ID: <SWEEP_ID>
#
#   # 3. Submit the sweep array (2 GPUs, 6 runs each = 12 concurrent)
#   SWEEP_ID=<SWEEP_ID> WM_CHECKPOINT=$HOME/model-based-eggroll/checkpoints/world_model.pkl \
#       sbatch scripts/run_sweep_eit.slurm
#
# Submit from the model-based-eggroll/ directory.

# ── Job metadata ───────────────────────────────────────────────────────────────
#SBATCH --job-name=mopo-sweep
#SBATCH --partition=compute
#SBATCH --time=08:00:00

# ── Resources ──────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24      # 6 runs × ~4 CPUs each (data loading + eval workers)
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# ── Array ──────────────────────────────────────────────────────────────────────
#SBATCH --array=0-1             # 2 GPU nodes, 6 parallel runs each = 12 total

# ── Logging ────────────────────────────────────────────────────────────────────
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

RUNS_PER_GPU=6

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

# ── JAX memory settings ────────────────────────────────────────────────────────
# Disable JAX's default 75% pre-allocation so multiple processes can share the GPU.
# Each process is capped at 15% of 80GB ≈ 12GB, comfortably above MOPO's ~8-10GB.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

# ── Diagnostics ────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job:      ${SLURM_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node:     $(hostname)"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Sweep:    ${SWEEP_ID}"
echo "WM ckpt:  ${WM_CHECKPOINT}"
echo "Runs/GPU: ${RUNS_PER_GPU}"
echo "Dir:      $(pwd)"
nvidia-smi
echo "============================================"

# ── Launch parallel sweep trials ───────────────────────────────────────────────
# Spawn RUNS_PER_GPU wandb agents in the background, each claiming one config.
# The sweep queue ensures no two agents pick the same config.
# 'wait' holds until all background processes finish.
for i in $(seq 1 "${RUNS_PER_GPU}"); do
    uv run wandb agent --count 1 "model-based-eggroll/model-based-eggroll/${SWEEP_ID}" &
done

wait
echo "All ${RUNS_PER_GPU} trials on task ${SLURM_ARRAY_TASK_ID} finished."
