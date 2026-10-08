#!/bin/bash
#SBATCH --job-name=agentcbo
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
exec "${PYTHON_BIN:-python}" -m experiments.run_comparison "$@"
