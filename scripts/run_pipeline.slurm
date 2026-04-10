#!/bin/bash
# SLURM script for running the model-based-eggroll pipeline on the Oxford ARC HTC cluster.
# https://arc-user-guide.readthedocs.io
#
# Usage:
#   sbatch scripts/run_pipeline.slurm [hydra-overrides...]
#
# Examples:
#   sbatch scripts/run_pipeline.slurm stage=all
#   sbatch scripts/run_pipeline.slurm stage=world_model dataset=halfcheetah_medium seed=1
#   sbatch scripts/run_pipeline.slurm stage=all world_model=mle policy_optimizer=mopo seed=42
#
# Any arguments are forwarded directly to Hydra as CLI overrides.
# Submit from the model-based-eggroll/ directory.
#
# ── One-time setup (run once on a compute node before first submission) ────────
#
#   # 1. Install uv
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   export PATH="$HOME/.local/bin:$PATH"
#
#   # 2. Install dependencies on a compute node (not the login node)
#   #    srun --clusters=htc -p interactive --pty /bin/bash
#   cd $DATA/model-based-eggroll
#   uv sync --group cuda
#
#   # 3. Log in to WandB
#   uv run wandb login
#
#   # 4. Download Minari datasets (requires internet — do on login node)
#   uv run python -c "import minari; minari.load_dataset('mujoco/halfcheetah/medium-v0', download=True)"
#
#   # 5. Create the log directory (SLURM needs this before the job starts)
#   mkdir -p logs/slurm
#
# ──────────────────────────────────────────────────────────────────────────────

# ── Job metadata ───────────────────────────────────────────────────────────────
#SBATCH --job-name=eggroll
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=08:00:00

# ── Resources ──────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       # JAX data loading + gymnasium vectorized eval workers
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# #SBATCH --constraint='gpu_sku:A100'  # optional: A100, L40S, RTXA6000, H100, ...

# ── Notifications ──────────────────────────────────────────────────────────────
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=<sso>@ox.ac.uk

# ── Logging ────────────────────────────────────────────────────────────────────
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
# Assumes the repo is cloned under $DATA. Adjust REPO_ROOT if needed.
REPO_ROOT="${DATA}/model-based-eggroll"
MINARI_CACHE="${DATA}/minari_cache"    # shared dataset cache across runs

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${MINARI_CACHE}"

cd "${REPO_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"
export MINARI_DATASETS_PATH="${MINARI_CACHE}"

# ── Diagnostics ────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Dir:      $(pwd)"
nvidia-smi
echo "============================================"

# ── Run pipeline ───────────────────────────────────────────────────────────────
uv run python src/mbrl/main.py \
    wandb.enabled=true \
    "$@"
