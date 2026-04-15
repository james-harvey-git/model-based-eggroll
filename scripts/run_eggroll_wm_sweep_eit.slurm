#!/bin/bash
# SLURM job for running the EGGROLL world-model W&B sweep on the EIT CDT cluster.
#
# Default sweep config:
#   configs/sweep/eggroll_wm_halfcheetah.yaml
#   1296 total trials
#
# Usage:
#   # 1. Register the sweep once from the repo root
#   uv run python scripts/init_sweep.py configs/sweep/eggroll_wm_halfcheetah.yaml
#   # → prints: Sweep ID: <SWEEP_ID>
#
#   # 2. Submit a job with however many GPUs you want to use total.
#   #    The script launches RUNS_PER_GPU W&B agents on each allocated GPU.
#   SWEEP_ID=<SWEEP_ID> sbatch --gres=gpu:2 scripts/run_eggroll_wm_sweep_eit.slurm
#
#   # 3. Optionally increase packing:
#   RUNS_PER_GPU=16 SWEEP_ID=<SWEEP_ID> sbatch --gres=gpu:2 scripts/run_eggroll_wm_sweep_eit.slurm

# ── Job metadata ───────────────────────────────────────────────────────────────
#SBATCH --job-name=eggroll-wm-sweep
#SBATCH --partition=compute
#SBATCH --time=08:00:00

# ── Resources ──────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# ── Logging ────────────────────────────────────────────────────────────────────
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

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

# Derive the exact GPUs assigned to this job. We then pin RUNS_PER_GPU agents
# onto each one so the total GPU usage matches the allocation the user requested.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

NUM_GPUS="${#GPU_IDS[@]}"
if [[ "${NUM_GPUS}" -eq 0 ]]; then
    echo "ERROR: no GPUs visible to the job allocation" >&2
    exit 1
fi

# ── JAX memory settings ────────────────────────────────────────────────────────
# Disable JAX preallocation so multiple processes can share each GPU. Since
# each agent is pinned to one GPU, the per-process cap depends only on
# RUNS_PER_GPU, not the total number of GPUs allocated to the job.
PER_PROCESS_MEM_FRACTION="$(awk -v runs="${RUNS_PER_GPU}" 'BEGIN { printf "%.4f", 0.90 / runs }')"

# ── Diagnostics ────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Num GPUs: ${NUM_GPUS}"
echo "Sweep:    ${SWEEP_ID}"
echo "Runs/GPU: ${RUNS_PER_GPU}"
echo "Dir:      $(pwd)"
nvidia-smi
echo "============================================"

# ── Launch parallel sweep trials ───────────────────────────────────────────────
# Launch RUNS_PER_GPU agents per allocated GPU, pinning each subprocess to a
# single device. Agents keep claiming work until the sweep queue is exhausted.
for gpu_id in "${GPU_IDS[@]}"; do
    for i in $(seq 1 "${RUNS_PER_GPU}"); do
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        XLA_PYTHON_CLIENT_MEM_FRACTION="${PER_PROCESS_MEM_FRACTION}" \
        uv run wandb agent "model-based-eggroll/model-based-eggroll/${SWEEP_ID}" &
    done
done

wait
echo "All sweep agents on job ${SLURM_JOB_ID} finished."
