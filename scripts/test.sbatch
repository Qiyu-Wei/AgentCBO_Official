#!/bin/bash
#SBATCH --job-name=agentcbo-tests
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Constructor-only LLM tests use an inert local endpoint and make no paid calls.
export USE_VLLM=true VLLM_MODEL=unit-test-model VLLM_BASE_URL=http://127.0.0.1:9/v1
"${PYTHON_BIN:-python}" -m compileall -q agentcbo experiments
"${PYTHON_BIN:-python}" -m experiments.run_comparison --help
"${PYTHON_BIN:-python}" -m pytest -q tests "$@"
