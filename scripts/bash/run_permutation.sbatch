#!/bin/bash
#SBATCH --job-name=mario_vp_perm
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-perm-%j.out
#SBATCH --error=logs/slurm-perm-%j.err

# Required env vars: GROUP
# Optional env vars: N_PERM, SEED

if [ -z "${GROUP}" ]; then
    echo "ERROR: GROUP env var required (e.g., GROUP=GROUP_N5_PRIMARY sbatch ...)"
    exit 1
fi

module load miniconda3
module load cuda/12.1
conda activate mario

cd /home1/ozvar/Git/cneuromod/mario_neural_encoding

EXTRA_FLAGS=""
if [ -n "${N_PERM}" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --n-perm ${N_PERM}"
fi
if [ -n "${SEED}" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --seed ${SEED}"
fi

python scripts/python/compute_vp_significance_permutation.py \
    --group ${GROUP} \
    $EXTRA_FLAGS
