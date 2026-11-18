#!/bin/bash
#SBATCH --job-name=mario_pme_reliability
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --chdir=/home1/ozvar/Git/cneuromod/mario_neural_encoding
#SBATCH --output=/home1/ozvar/Git/cneuromod/mario_neural_encoding/logs/pme_reliability_%x_%j.out
#SBATCH --error=/home1/ozvar/Git/cneuromod/mario_neural_encoding/logs/pme_reliability_%x_%j.err

set -euo pipefail

module load miniconda3
conda activate mario

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python scripts/python/audit_pme_reliability.py \
    --group GROUP_N5_PRIMARY \
    --n-permutations 1000

python scripts/python/plot_pme_reliability.py \
    --table-dir results/pme_reliability \
    --output-dir results/figures/pme_reliability
