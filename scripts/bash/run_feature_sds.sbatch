#!/bin/bash
#SBATCH --job-name=mario_feature_sds
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Required env vars: SUBJECT, SESSIONS, TEST_SESSIONS, EXPERIMENT_ID

module load miniconda3
conda activate mario

# Cap BLAS / OpenMP thread counts for consistent host-memory behaviour.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home1/ozvar/Git/cneuromod/mario_neural_encoding

python scripts/python/compute_feature_sds.py \
    --subject ${SUBJECT} \
    --sessions ${SESSIONS} \
    --test-sessions ${TEST_SESSIONS} \
    --experiment-id ${EXPERIMENT_ID}
