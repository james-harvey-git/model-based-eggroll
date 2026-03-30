#!/bin/bash
# SLURM script for running the model-based-eggroll pipeline on the EIT CDT cluster.
#
# Usage:
#   sbatch scripts/run_pipeline_eit.slurm [hydra-overrides...]
#
# Examples:
#   sbatch scripts/run_pipeline_eit.slurm stage=all
#   sbatch scripts/run_pipeline_eit.slurm stage=world_model dataset=halfcheetah_medium seed=1
#   sbatch scripts/run_pipeline_eit.slurm stage=all world_model=mle policy_optimizer=mopo seed=42
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
#   cd $HOME/model-based-eggroll
#   uv sync --group cuda
#
#   # 3. Log in to WandB
#   uv run wandb login
#
#   # 4. Download Minari datasets (requires internet — do on login node)
#   #    Set MINARI_DATASETS_PATH so datasets land in the same cache the job uses.
#   MINARI_DATASETS_PATH=$HOME/minari_cache uv run python -c "import minari; minari.load_dataset('mujoco/halfcheetah/medium-v0', download=True)"
#
#   # 5. Create the log directory (SLURM needs this before the job starts)
#   mkdir -p logs/slurm
#
# ──────────────────────────────────────────────────────────────────────────────

# ── Job metadata ───────────────────────────────────────────────────────────────
#SBATCH --job-name=eggroll
#SBATCH --partition=compute
#SBATCH --time=08:00:00

# ── Resources ──────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       # JAX data loading + gymnasium vectorized eval workers
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# ── Notifications ──────────────────────────────────────────────────────────────
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=<sso>@ox.ac.uk

# ── Logging ────────────────────────────────────────────────────────────────────
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
# Assumes the repo is cloned under $HOME. Adjust REPO_ROOT if needed.
REPO_ROOT="${HOME}/model-based-eggroll"
MINARI_CACHE="${HOME}/minari_cache"    # shared dataset cache across runs

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
