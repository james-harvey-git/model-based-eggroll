#!/bin/bash
# SLURM array job for grid-searching EGGROLL world-model hyperparameters on the
# EIT CDT cluster.
#
# The script enumerates a Cartesian product over the selected hyperparameter
# lists, filters invalid (population_size, group_size) pairs, and launches up to
# RUNS_PER_GPU trials in parallel on each 80GB H100.
#
# Default grid (override any axis via env vars listed below):
#   population_size   ∈ {2048, 4096}
#   group_size        ∈ {0, 2, 4, 8}
#   weight_decay      ∈ {0, 1e-5, 1e-4}
#   adamw b1          ∈ {0.9, 0.95}
#   sigma             ∈ {0.005, 0.01, 0.02}
#   sigma_decay_rate  ∈ {0.99997, 0.999985, 0.999995}
#   lr                ∈ {1e-4, 3e-4, 1e-3}
#
# Validity constraints:
#   - group_size may be 0 (ungrouped mode), or
#   - group_size must be even and divide population_size
#
# Typical usage:
#   # Preview the grid locally / on a login node
#   DRY_RUN=true bash scripts/run_eggroll_wm_grid_eit.slurm
#
#   # Submit using the default 12 runs per GPU
#   sbatch --array=0-107 scripts/run_eggroll_wm_grid_eit.slurm
#
#   # Override the grid axes and parallelism
#   RUNS_PER_GPU=16 \
#   POPULATION_SIZES_STR="2048 4096" \
#   GROUP_SIZES_STR="0 2 4" \
#   SIGMAS_STR="0.005 0.01" \
#   sbatch --array=0-35 scripts/run_eggroll_wm_grid_eit.slurm
#
# Any extra CLI arguments are forwarded directly to Hydra for every trial.

# ── Job metadata ───────────────────────────────────────────────────────────────
#SBATCH --job-name=eggroll-wm-grid
#SBATCH --partition=compute
#SBATCH --time=06:00:00

# ── Resources ──────────────────────────────────────────────────────────────────
# Default sizing assumes 12 concurrent world-model runs on one H100.
# If you raise RUNS_PER_GPU substantially, also increase cpus-per-task / memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --gres=gpu:1

# ── Array ──────────────────────────────────────────────────────────────────────
# Override with sbatch --array=0-<n> based on the printed total grid size.
#SBATCH --array=0-0

# ── Logging ────────────────────────────────────────────────────────────────────
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

RUNS_PER_GPU="${RUNS_PER_GPU:-12}"
SEED="${SEED:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
DATASET_CFG="${DATASET_CFG:-halfcheetah_medium}"
WORLD_MODEL_CFG="${WORLD_MODEL_CFG:-eggroll_ensemble}"
DRY_RUN="${DRY_RUN:-false}"

POPULATION_SIZES_STR="${POPULATION_SIZES_STR:-2048 4096}"
GROUP_SIZES_STR="${GROUP_SIZES_STR:-0 2 4 8}"
WEIGHT_DECAYS_STR="${WEIGHT_DECAYS_STR:-0 1e-5 1e-4}"
ADAMW_B1S_STR="${ADAMW_B1S_STR:-0.9 0.95}"
SIGMAS_STR="${SIGMAS_STR:-0.005 0.01 0.02}"
SIGMA_DECAY_RATES_STR="${SIGMA_DECAY_RATES_STR:-0.99997 0.999985 0.999995}"
LRS_STR="${LRS_STR:-1e-4 3e-4 1e-3}"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)"
REPO_ROOT="${REPO_ROOT:-$(
    cd "${SCRIPT_DIR}/.." && pwd
)}"
MINARI_CACHE="${HOME}/minari_cache"
JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_ROOT="${REPO_ROOT}/sweep_runs/eggroll_wm_grid/${JOB_ID}"
TASK_LOG_ROOT="${REPO_ROOT}/logs/eggroll_wm_grid/${JOB_ID}/task_${ARRAY_TASK_ID}"
GRID_FILE="${RUN_ROOT}/grid.tsv"

mkdir -p "${REPO_ROOT}/logs/slurm"
mkdir -p "${MINARI_CACHE}"
mkdir -p "${RUN_ROOT}"
mkdir -p "${TASK_LOG_ROOT}"

cd "${REPO_ROOT}"

export PATH="${HOME}/.local/bin:${PATH}"
export MINARI_DATASETS_PATH="${MINARI_CACHE}"

# ── JAX memory settings ────────────────────────────────────────────────────────
# Disable eager 75% GPU preallocation so multiple JAX processes can share one
# H100. Cap each process to an even slice of ~90% of the device.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$(
    awk -v runs="${RUNS_PER_GPU}" 'BEGIN { printf "%.4f", 0.90 / runs }'
)"

export POPULATION_SIZES_STR
export GROUP_SIZES_STR
export WEIGHT_DECAYS_STR
export ADAMW_B1S_STR
export SIGMAS_STR
export SIGMA_DECAY_RATES_STR
export LRS_STR

# ── Build valid grid ───────────────────────────────────────────────────────────
uv run python - <<'PY' > "${GRID_FILE}"
import itertools
import os


def parse_ints(name: str) -> list[int]:
    return [int(v) for v in os.environ[name].split()]


def parse_strs(name: str) -> list[str]:
    return os.environ[name].split()


population_sizes = parse_ints("POPULATION_SIZES_STR")
group_sizes = parse_ints("GROUP_SIZES_STR")
weight_decays = parse_strs("WEIGHT_DECAYS_STR")
adamw_b1s = parse_strs("ADAMW_B1S_STR")
sigmas = parse_strs("SIGMAS_STR")
sigma_decay_rates = parse_strs("SIGMA_DECAY_RATES_STR")
lrs = parse_strs("LRS_STR")

trial_idx = 0
for population_size in population_sizes:
    for group_size in group_sizes:
        if group_size != 0 and (group_size % 2 != 0 or population_size % group_size != 0):
            continue

        for weight_decay, b1, sigma, sigma_decay_rate, lr in itertools.product(
            weight_decays,
            adamw_b1s,
            sigmas,
            sigma_decay_rates,
            lrs,
        ):
            print(
                trial_idx,
                population_size,
                group_size,
                weight_decay,
                b1,
                sigma,
                sigma_decay_rate,
                lr,
                sep="\t",
            )
            trial_idx += 1
PY

TOTAL_TRIALS=$(wc -l < "${GRID_FILE}" | tr -d ' ')
if [[ "${TOTAL_TRIALS}" -eq 0 ]]; then
    echo "ERROR: Hyperparameter grid is empty after filtering invalid combinations." >&2
    exit 1
fi

TOTAL_TASKS=$(( (TOTAL_TRIALS + RUNS_PER_GPU - 1) / RUNS_PER_GPU ))
START_TRIAL=$(( ARRAY_TASK_ID * RUNS_PER_GPU ))
END_TRIAL=$(( START_TRIAL + RUNS_PER_GPU - 1 ))
if (( END_TRIAL >= TOTAL_TRIALS )); then
    END_TRIAL=$(( TOTAL_TRIALS - 1 ))
fi

# ── Diagnostics ────────────────────────────────────────────────────────────────
echo "============================================"
echo "Job:              ${JOB_ID}"
echo "Task:             ${ARRAY_TASK_ID}"
echo "Node:             $(hostname)"
echo "GPU(s):           ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Runs/GPU:         ${RUNS_PER_GPU}"
echo "Dataset cfg:      ${DATASET_CFG}"
echo "WM cfg:           ${WORLD_MODEL_CFG}"
echo "Seed:             ${SEED}"
echo "W&B enabled:      ${WANDB_ENABLED}"
echo "Grid file:        ${GRID_FILE}"
echo "Total trials:     ${TOTAL_TRIALS}"
echo "Suggested array:  --array=0-$((TOTAL_TASKS - 1))"
echo "Assigned trials:  ${START_TRIAL}-${END_TRIAL}"
echo "Mem fraction:     ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "Run root:         ${RUN_ROOT}"
echo "Task logs:        ${TASK_LOG_ROOT}"
echo "Dir:              $(pwd)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi
echo "============================================"

echo "Grid axes:"
echo "  population_size:  ${POPULATION_SIZES_STR}"
echo "  group_size:       ${GROUP_SIZES_STR}"
echo "  weight_decay:     ${WEIGHT_DECAYS_STR}"
echo "  adamw b1:         ${ADAMW_B1S_STR}"
echo "  sigma:            ${SIGMAS_STR}"
echo "  sigma_decay_rate: ${SIGMA_DECAY_RATES_STR}"
echo "  lr:               ${LRS_STR}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo
    echo "Dry run enabled. First 10 trials:"
    head -n 10 "${GRID_FILE}"
    exit 0
fi

if (( START_TRIAL >= TOTAL_TRIALS )); then
    echo "Task ${ARRAY_TASK_ID} has no assigned trials (grid exhausted)." >&2
    exit 0
fi

cat > "${RUN_ROOT}/README.txt" <<EOF
SLURM_JOB_ID=${JOB_ID}
RUNS_PER_GPU=${RUNS_PER_GPU}
SEED=${SEED}
DATASET_CFG=${DATASET_CFG}
WORLD_MODEL_CFG=${WORLD_MODEL_CFG}
WANDB_ENABLED=${WANDB_ENABLED}
XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}
POPULATION_SIZES_STR=${POPULATION_SIZES_STR}
GROUP_SIZES_STR=${GROUP_SIZES_STR}
WEIGHT_DECAYS_STR=${WEIGHT_DECAYS_STR}
ADAMW_B1S_STR=${ADAMW_B1S_STR}
SIGMAS_STR=${SIGMAS_STR}
SIGMA_DECAY_RATES_STR=${SIGMA_DECAY_RATES_STR}
LRS_STR=${LRS_STR}
TOTAL_TRIALS=${TOTAL_TRIALS}
TOTAL_TASKS=${TOTAL_TASKS}
EXTRA_HYDRA_OVERRIDES=$*
EOF

mapfile -t ASSIGNED_TRIALS < <(sed -n "$((START_TRIAL + 1)),$((END_TRIAL + 1))p" "${GRID_FILE}")

# ── Launch assigned trials in parallel ────────────────────────────────────────
pids=()

for trial in "${ASSIGNED_TRIALS[@]}"; do
    IFS=$'\t' read -r trial_idx population_size group_size weight_decay b1 sigma sigma_decay_rate lr <<< "${trial}"

    trial_name=$(printf "trial_%04d" "${trial_idx}")
    trial_root="${RUN_ROOT}/${trial_name}"
    trial_stdout="${TASK_LOG_ROOT}/${trial_name}.out"
    trial_stderr="${TASK_LOG_ROOT}/${trial_name}.err"
    mkdir -p "${trial_root}"

    cat > "${trial_root}/config.txt" <<EOF
trial_idx=${trial_idx}
population_size=${population_size}
group_size=${group_size}
weight_decay=${weight_decay}
adamw_b1=${b1}
sigma=${sigma}
sigma_decay_rate=${sigma_decay_rate}
lr=${lr}
EOF

    (
        echo "[${trial_name}] starting at $(date --iso-8601=seconds)"
        echo "[${trial_name}] population_size=${population_size} group_size=${group_size} weight_decay=${weight_decay} b1=${b1} sigma=${sigma} sigma_decay_rate=${sigma_decay_rate} lr=${lr}"
        echo "[${trial_name}] checkpoint root: ${trial_root}"
        uv run python src/mbrl/main.py \
            stage=world_model \
            dataset="${DATASET_CFG}" \
            world_model="${WORLD_MODEL_CFG}" \
            debug=false \
            seed="${SEED}" \
            wandb.enabled="${WANDB_ENABLED}" \
            checkpoint_dir="${trial_root}" \
            world_model.eggroll.solver=adamw \
            world_model.eggroll.population_size="${population_size}" \
            world_model.eggroll.group_size="${group_size}" \
            world_model.eggroll.solver_kwargs.weight_decay="${weight_decay}" \
            +world_model.eggroll.solver_kwargs.b1="${b1}" \
            world_model.eggroll.sigma="${sigma}" \
            world_model.eggroll.sigma_decay_rate="${sigma_decay_rate}" \
            world_model.eggroll.lr="${lr}" \
            "$@"
        echo "[${trial_name}] finished at $(date --iso-8601=seconds)"
    ) >"${trial_stdout}" 2>"${trial_stderr}" &

    pids+=("$!")
done

failed=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        failed_trial="${ASSIGNED_TRIALS[$idx]}"
        IFS=$'\t' read -r trial_idx _ <<< "${failed_trial}"
        trial_name=$(printf "trial_%04d" "${trial_idx}")
        echo "Trial ${trial_name} failed. See ${TASK_LOG_ROOT}/${trial_name}.err" >&2
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "One or more EGGROLL world-model sweep trials failed." >&2
    exit 1
fi

echo "Task ${ARRAY_TASK_ID} finished ${#ASSIGNED_TRIALS[@]} trial(s) successfully."
