#!/bin/bash
#SBATCH --job-name=mario_neurosynth
#SBATCH --partition=blade
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output=slurm-neurosynth-%j.out
#SBATCH --error=slurm-neurosynth-%j.err

module load miniconda3
conda activate mario

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /home1/ozvar/Git/cneuromod/mario_neural_encoding

#python scripts/python/prepare_neurosynth_map.py
python scripts/python/neurosynth_decoding.py
