#!/bin/bash
#SBATCH --job-name=report_plots
#SBATCH --partition=cpu            # <-- CPU partition; NO GPU (forward-only eval)
#SBATCH --account=REPLACE_ACCOUNT  # <-- your slurm account, if required
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=report_plots_%j.log

# Report figures for trajectory world-model fine-tuning. No GPU: this is unperturbed
# forward-only rollout eval on small MLPs + matplotlib. Submit from the repo root:
#     sbatch scripts/report_plots.slurm
#     sbatch scripts/report_plots.slurm --only validation
#     sbatch scripts/report_plots.slurm -s halfcheetah_val
# Any extra args ($@) are forwarded to report_plots.py.

set -euo pipefail

# Force CPU and keep JAX/BLAS from oversubscribing beyond the allocation.
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-8}"

uv run python scripts/report_plots.py --config scripts/report_plots.yaml "$@"
