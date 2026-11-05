#!/bin/bash
#SBATCH --job-name=mario_vp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Required env vars: SUBJECT, SESSIONS, TEST_SESSIONS

module load miniconda3
module load cuda/12.1
conda activate mario

# Cap BLAS / OpenMP thread counts so per-thread scratch buffers don't scale
# with the node's CPU count and produce inconsistent host-memory peaks.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home1/ozvar/Git/cneuromod/mario_neural_encoding

python scripts/python/fit_variance_partitioning.py \
    --subject ${SUBJECT} \
    --sessions ${SESSIONS} \
    --test-sessions ${TEST_SESSIONS} \
    --backend torch_cuda \
    --cv-scheme loso \
    --compute-product-measure
