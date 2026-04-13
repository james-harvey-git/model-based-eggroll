#!/bin/bash
# SLURM job for reproducibility checks on the EIT CDT cluster.
#
# Launches multiple identical full-pipeline runs in parallel on a single GPU so
# we can verify that `mle + mopo + halfcheetah_medium` is reproducible under the
# exact same algorithm config and seed.
#
# Each replica gets its own checkpoint root to avoid collisions when the runs
# start in the same second and therefore generate identical timestamp-based run
# names internally.
#
# Usage:
#   # Default: 4 identical stage=all runs of +experiment=halfcheetah_medium_mopo seed=0
#   sbatch scripts/run_repro_eit.slurm
#
#   # 6 replicas on one H100, still using the same seed/config for every replica
#   RUNS_PER_GPU=6 sbatch scripts/run_repro_eit.slurm
#
#   # Change the seed, keeping all replicas identical
#   SEED=7 sbatch scripts/run_repro_eit.slurm
#
#   # Enable W&B for all replicas (disabled by default for cleaner repro checks)
#   WANDB_ENABLED=true sbatch scripts/run_repro_eit.slurm
#
#   # Apply the same extra Hydra overrides to every replica
#   sbatch scripts/run_repro_eit.slurm eval.num_episodes=5 debug=false
#
# Submit from the model-based-eggroll/ directory.

# ── Job metadata ───────────────────────────────────────────────────────────────
#SBATCH --job-name=mopo-repro
#SBATCH --partition=compute
#SBATCH --time=08:00:00

# ── Resources ──────────────────────────────────────────────────────────────────
# Default sizing assumes up to 6 parallel replicas, each roughly matching the
# single-run CPU footprint from run_pipeline_eit.slurm.
# If you raise RUNS_PER_GPU above 6, also increase cpus-per-task and memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# ── Logging ────────────────────────────────────────────────────────────────────
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

RUNS_PER_GPU="${RUNS_PER_GPU:-4}"
SEED="${SEED:-0}"
EXPERIMENT="${EXPERIMENT:-halfcheetah_medium_mopo}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="${HOME}/model-based-eggroll"
MINARI_CACHE="${HOME}/minari_cache"
RUN_ROOT="${REPO_ROOT}/repro_runs/${SLURM_JOB_ID}"
REPLICA_LOG_ROOT="${REPO_ROOT}/logs/repro/${SLURM_JOB_ID}"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${MINARI_CACHE}"
mkdir -p "${RUN_ROOT}"
mkdir -p "${REPLICA_LOG_ROOT}"

cd "${REPO_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"
export MINARI_DATASETS_PATH="${MINARI_CACHE}"

# ── JAX memory settings ────────────────────────────────────────────────────────
# Multiple JAX processes cannot safely share one GPU if each eagerly grabs most
# of the device memory. Disable preallocation and cap each process to an even
# slice of ~90% of the GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$(
    awk -v runs="${RUNS_PER_GPU}" 'BEGIN { printf "%.3f", 0.90 / runs }'
)"

# ── Diagnostics ────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job:            ${SLURM_JOB_ID}"
echo "Node:           $(hostname)"
echo "GPU(s):         ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Experiment:     ${EXPERIMENT}"
echo "Seed:           ${SEED}"
echo "Runs/GPU:       ${RUNS_PER_GPU}"
echo "W&B enabled:    ${WANDB_ENABLED}"
echo "Run root:       ${RUN_ROOT}"
echo "Replica logs:   ${REPLICA_LOG_ROOT}"
echo "Mem fraction:   ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "Dir:            $(pwd)"
nvidia-smi
echo "============================================"

# Persist the exact launcher settings used for this repro batch.
cat > "${RUN_ROOT}/README.txt" <<EOF
SLURM_JOB_ID=${SLURM_JOB_ID}
EXPERIMENT=${EXPERIMENT}
SEED=${SEED}
RUNS_PER_GPU=${RUNS_PER_GPU}
WANDB_ENABLED=${WANDB_ENABLED}
XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}
EXTRA_HYDRA_OVERRIDES=$*
EOF

# ── Launch identical replicas ──────────────────────────────────────────────────
pids=()

for replica_idx in $(seq 1 "${RUNS_PER_GPU}"); do
    replica_name=$(printf "replica_%02d" "${replica_idx}")
    replica_root="${RUN_ROOT}/${replica_name}"
    replica_stdout="${REPLICA_LOG_ROOT}/${replica_name}.out"
    replica_stderr="${REPLICA_LOG_ROOT}/${replica_name}.err"

    mkdir -p "${replica_root}"

    (
        echo "[${replica_name}] starting at $(date --iso-8601=seconds)"
        echo "[${replica_name}] checkpoint root: ${replica_root}"
        uv run python src/mbrl/main.py \
            +experiment="${EXPERIMENT}" \
            stage=all \
            seed="${SEED}" \
            wandb.enabled="${WANDB_ENABLED}" \
            checkpoint_dir="${replica_root}" \
            "$@"
        echo "[${replica_name}] finished at $(date --iso-8601=seconds)"
    ) >"${replica_stdout}" 2>"${replica_stderr}" &

    pids+=("$!")
done

failed=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        replica_name=$(printf "replica_%02d" "$((idx + 1))")
        echo "Replica ${replica_name} failed. See ${REPLICA_LOG_ROOT}/${replica_name}.err" >&2
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "One or more reproducibility replicas failed." >&2
    exit 1
fi

echo "All ${RUNS_PER_GPU} reproducibility replicas finished successfully."
echo "Outputs are under ${RUN_ROOT}"
