#!/bin/bash
#SBATCH --job-name=mario_post_vp
#SBATCH --partition=blade
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_post-modelling-%j.out
#SBATCH --error=logs/run_post-modelling-%j.err

# =============================================================================
# CONFIGURATION -- update GROUP before submitting
# =============================================================================
GROUP=GROUP_N5_PRIMARY

module load miniconda3
conda activate mario

cd /home1/ozvar/Git/cneuromod/mario_neural_encoding

# =============================================================================
# PCA WEIGHT PREPARATION
# Rescales VP model weights for group PCA. Session configs and experiment IDs
# are read from group_configs.py -- update that file, not this script.
# =============================================================================

python scripts/python/prepare_pca_weights.py --group ${GROUP}

# =============================================================================
# GROUP PCA
# =============================================================================

python scripts/python/fit_pca.py --group ${GROUP}

# =============================================================================
# CIFTI GENERATION
# =============================================================================

python scripts/python/create_vp_model_cifti.py --group ${GROUP}

echo "Post-VP pipeline complete for ${GROUP}!"
