# Mario Neural Encoding Model

Voxelwise encoding model analysis of fMRI data collected during Super Mario Bros gameplay (CNeuroMod dataset).

## Project Overview

This project fits voxelwise banded ridge encoding models that map behavioural and visual features of naturalistic gameplay onto brain activity, then decomposes the held-out explained variance between the two feature spaces. The approach follows Dupré la Tour et al. (2025) and uses the Gallant Lab's [voxelwise tutorials](https://gallantlab.org/voxelwise_tutorials) stack.

**Dataset:** [Courtois NeuroMod Mario dataset](https://docs.cneuromod.ca/), available via DataLad
**Task:** Super Mario Bros (NES) gameplay across multiple levels
**Subjects:** sub-01, 02, 03, 05, 06 (sub-04 has no data); 9 to 19 training sessions each, plus one held-out test session
**Acquisition:** TR = 1.49 s, CIFTI in fsLR_32k (91,282 grayordinates)
**fMRI preprocessing:** HCP pipeline (MSMAll-registered, high-pass filtered, ICA-FIX cleaned)

### Experimental Design

- **Discovery phase:** players repeatedly attempt each level until they succeed at least once (learning effects present)
- **Practice phase:** players play randomly selected levels they have already mastered (stable behaviour)

Models are trained and tested exclusively on practice-phase data, so brain-behaviour mappings are not confounded by learning.

---

## Documentation

The README is the only tracked Markdown file. Detailed methods, pipeline and design documents are kept locally under `docs/` and are intentionally not included in the code repository.

---

## Repository Structure

```
mario_neural_encoding/
├── docs/                            # Local documentation (gitignored)
├── inputs/                          # Processed behavioural data (gitignored)
├── results/                         # Model outputs, figures, logs (gitignored; most live on the cluster)
├── scripts/
│   ├── python/                      # Pipeline, analysis, diagnostics, figures
│   └── bash/                        # SLURM submission scripts
├── src/python/mario_encoding/       # Core package (config, data loading, model fitting)
├── environment.yml
└── README.md
```

Script families in `scripts/python/`:

- `process_*`, `merge_*`, `downsample_*`, `create_practice_phase_metadata.py`: behavioural preprocessing (pipeline steps 1-5)
- `fit_*`, `submit_*`: model fitting and job submission
- `compute_*`: significance testing, group statistics, derived measures
- `check_*`, `diagnose_*`, `inspect_*`: data and pipeline checks
- `plot_*`, `create_*_cifti.py`, `export_*`: figures and brain maps
- `*neurosynth*`: Neurosynth follow-up analyses

### External Data Repositories (sibling directories)

- `mario.replays/`: raw behavioural data (gamelogs) and extracted replay variables
- `mario.scenes/`: scene-level perceptual features and metadata
- `mario.annotations/`: BIDS-format event annotations with temporal alignment

---

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd mario_neural_encoding
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate mario
```

### 3. Install CUDA-enabled PyTorch
**Required for GPU acceleration via himalaya's torch_cuda backend:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install the mario_encoding package
```bash
pip install -e .
```

### 5. Configure paths
Edit `src/python/mario_encoding/config.py` to point at your data directories if they differ from the default sibling-directory layout.

### Computational Requirements

- **GPU:** CUDA-enabled GPU required for practical runtimes
- **Memory:** 16-32 GB GPU memory recommended for 91k grayordinates
- **Backend:** himalaya with the `torch_cuda` backend
- **Batching:** fits run in 2000-target batches (`n_targets_batch` in `config.py`) to bound GPU memory

---

## Pipeline

Full commands are in `docs/pipeline.md`. In outline:

```
Raw replay JSONs (60 fps)
  [1] process_replay_sessions.py                 -> session-level replay variables
  [2] merge_bids_events_and_repvars.py           -> run-level data with BIDS events
  [3] merge_cleaned_behav_and_scene_features.py  -> engineered, pruned features
  [4] downsample_behav_data_to_TR.py             -> one row per TR (1.49 s)
  [5] create_practice_phase_metadata.py          -> practice-phase run and level onsets
  [6] fit_variance_partitioning.py               -> per-subject model fits
  [7-9] group average, per-subject permutation significance, group significance
  [10-11] PCA on model weights (deferred; see docs/methods_full.md)
  [12-13] CIFTI maps and figures
```

Steps 6, 8 and 10-12 run on a SLURM cluster via the wrappers in `scripts/bash/`.

### Model

Banded ridge regression (himalaya `GroupRidgeCV`) with a separate regularisation parameter per feature space, so the 6-feature motor space and the ~38-feature scene space each get a penalty on their own scale. FIR delays of 1 to 5 TRs act as a flexible HRF basis. Hyperparameters are selected by random search (100 draws) under leave-one-session-out cross-validation on the training sessions; prediction accuracy is evaluated on a genuinely held-out session. Held-out variance is decomposed per feature space with the product measure of Dupré la Tour et al. (2022), which needs only the single joint fit.

Preprocessing order inside the fit is load-bearing: features are mean-centred (not z-scored), FIR delays are applied per run, and only then are inter-trial-interval TRs dropped. `docs/methods_full.md` gives the reasoning and the consequences of getting the order wrong.

### Significance

Per-voxel block-permutation test on held-out predictions, following LeBel et al. (2023), with BH-FDR per statistic. At group level the criterion is a partial conjunction: a voxel counts if it is individually significant in at least 4 of the 5 subjects. Fixed-effects (Stouffer) pooling is deprecated; see `docs/methods_full.md`.

---

## Feature Engineering Details

### Categorical Encodings
- **Powerstate:** one-hot encoded (big, fire, star; small is the reference)
- **Scrolling:** binarised (screen scrolling vs static)
- **Powerup visibility:** binarised (visible vs not)
- **Jump state:** binarised (airborne vs grounded)
- **Movement direction:** one-hot encoded (left, right; stationary is the reference)

### Event Duration Extensions
Events are extended by fixed frame counts so they span a plausible neural response:
- Coin collected: 30 frames (~0.5 s)
- Brick smashed: 30 frames (~0.5 s)
- Powerup collected: 114 frames (~1.9 s)
- Hit, powerup lost: 35 frames (~0.6 s)
- Hit, life lost: 180 frames (~3.0 s)

### Kill Events
Derived from RAM variables (`enemy_kill30-34`):
- `event_kill_stomp` (value 4)
- `event_kill_impact` (value 34)
- `event_kill_kick` (value 132)

### Scene Features
Merged from the scenes mastersheet (`mario.scenes/sourcedata/scenes_info/scenes_mastersheet.csv`): perceptual complexity metrics, spatial layout properties and object density statistics.

Zero-variance filtering during fitting drops a few rare scene features per subject, so the surviving scene space is 37-38 features and is not identical across subjects. Any cross-subject analysis over individual features must align them by name.

---

## Configuration

All paths, feature-space definitions and model hyperparameters live in `src/python/mario_encoding/config.py`; subject and session groupings live in `src/python/mario_encoding/group_configs.py`. Treat those files as authoritative rather than copying values into documentation.

---

## Data Quality Checks

Validation scripts in `scripts/python/`, for example:
- `check_behav_data_files_exist.py`: behavioural data integrity
- `check_fmri_data_files_exist.py`: fMRI data availability
- `check_behav_fmri_alignment.py`: temporal alignment
- `check_feature_correlations.py`: cross-feature and cross-space correlation structure

Logs are written to `results/logs/`.

---

## References

**Methodology:**
- Dupré la Tour, T., Visconti di Oleggio Castello, M., & Gallant, J.L. (2025). The Voxelwise Encoding Model framework: A tutorial introduction to fitting encoding models to fMRI data. *Imaging Neuroscience*, 3. https://doi.org/10.1162/imag_a_00575
- Dupré la Tour, T., Eickenberg, M., Nunez-Elizalde, A.O., & Gallant, J.L. (2022). Feature-space selection with banded ridge regression. *NeuroImage*, 264, 119728.
- Huth, A.G., Nishimoto, S., Vu, A.T., & Gallant, J.L. (2012). A continuous semantic space describes the representation of thousands of object and action categories across the human brain. *Neuron*, 76(6), 1210-1224.
- LeBel, A., Wagner, L., Jain, S., et al. (2023). A natural language fMRI dataset for voxelwise encoding models. *Scientific Data*, 10, 555.

**Dataset:**
- Boyle, J.A., Pinsard, B., Boukhdhir, A., et al. (2020). The Courtois project on neural modelling. *arXiv*. https://arxiv.org/abs/2009.13276

**Software:**
- Gallant Lab voxelwise tutorials: https://gallantlab.org/voxelwise_tutorials
- Himalaya (ridge regression): https://github.com/gallantlab/himalaya

---

## Citation

If you use this code or analysis approach, please cite:
- The CNeuroMod dataset (see https://docs.cneuromod.ca/)
- Dupré la Tour et al. (2025) for the VEM framework
- This repository: [Add DOI/URL when available]

---

## Contact

[Add your contact information or lab/institution details]

---

## License

[Specify license - MIT, Apache 2.0, etc.]

See `LICENSE` file for details.
