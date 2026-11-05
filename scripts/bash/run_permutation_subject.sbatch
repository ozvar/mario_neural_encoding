#!/bin/bash
#SBATCH --job-name=mario_vp_perm_sub
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-perm-%j.out
#SBATCH --error=logs/slurm-perm-%j.err

# Per-subject block-permutation significance (CPU; the permutation loop is NumPy).
# Required env vars: SUBJECT, SESSIONS, TEST_SESSIONS, EXPERIMENT_ID
# Optional env vars: N_PERM, SEED

for v in SUBJECT SESSIONS TEST_SESSIONS EXPERIMENT_ID; do
    if [ -z "${!v}" ]; then
        echo "ERROR: ${v} env var required"
        exit 1
    fi
done

module load miniconda3
conda activate mario

# Cap BLAS / OpenMP thread counts for consistent host-memory behaviour.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /home1/ozvar/Git/cneuromod/mario_neural_encoding

EXTRA_FLAGS=""
if [ -n "${N_PERM}" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --n-perm ${N_PERM}"
fi
if [ -n "${SEED}" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --seed ${SEED}"
fi

python scripts/python/compute_vp_significance_permutation.py \
    --subject ${SUBJECT} \
    --train-sessions ${SESSIONS} \
    --test-sessions ${TEST_SESSIONS} \
    --experiment-id ${EXPERIMENT_ID} \
    $EXTRA_FLAGS
