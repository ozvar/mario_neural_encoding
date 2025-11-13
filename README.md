# Mario Neural Encoding Model

Voxelwise encoding model analysis of fMRI data collected during Super Mario Bros gameplay (CNeuroMod dataset).

## Project Overview

This project implements a voxelwise encoding model (VEM) framework to map behavioral and visual features during naturalistic gameplay to brain activity measured with fMRI. The analysis follows the methodology described in Dupré la Tour et al. (2025) and uses tools from the Gallant Lab's [voxelwise tutorials](https://gallantlab.org/voxelwise_tutorials).

**Dataset:** [Courtois NeuroMod Mario dataset](https://docs.cneuromod.ca/) - publicly available via DataLad  
**Task:** Super Mario Bros (NES) gameplay across multiple levels  
**Subjects:** 3 participants (sub-01, sub-03, sub-05)  
**Sessions:** 16-21 sessions per subject  
**Acquisition:** TR = 1.49s, CIFTI format (fsLR 91k surface space)

### Experimental Design

- **Discovery phase:** Players repeatedly attempt each level until succeeding at least once (learning effects present)
- **Practice phase:** Players play randomly selected levels they have already mastered (stable behavior, no learning effects)

**Analysis strategy:** Train and test models exclusively on practice phase data to ensure stable brain-behavior mappings without confounds from learning.

---

## Repository Structure

```
mario_neural_encoding/
├── inputs/                          # Processed behavioral data (gitignored)
│   ├── per_session_combined_repvars_tsvs/
│   ├── per_run_combined_repvars_and_bids_events/
│   ├── per_run_pruned_features/
│   ├── per_run_downsampled_to_TR/
│   └── practice_phase_metadata/
├── results/
│   ├── models/                      # Fitted encoding models (.pkl)
│   ├── cv_scores/                   # Cross-validation R² scores
│   ├── figures/                     # Output visualizations
│   └── logs/                        # Processing logs
├── scripts/python/                  # Preprocessing and analysis scripts
├── src/python/mario_encoding/       # Core utilities package
├── notebooks/                       # Analysis notebooks
├── environment.yml                  # Conda environment specification
├── config.py                        # Paths and parameters configuration
└── README.md                        # This file
```

### External Data Repositories (sibling directories)

- `mario.replays/` - Raw behavioral data (gamelogs) and extracted replay variables
- `mario.scenes/` - Scene-level perceptual features and metadata
- `mario.annotations/` - BIDS-format event annotations with temporal alignment
- `mario.fmriprep/` - Preprocessed fMRI data (CIFTI format)

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
pip install -e src/python/
```

### 5. Configure paths
Edit `src/python/mario_encoding/config.py` to point to your data directories if they differ from the default sibling directory structure.

### Computational Requirements
- **GPU:** CUDA-enabled GPU required for practical runtimes
- **Memory:** ~16-32GB GPU memory recommended for 91k grayordinates
- **Backend:** himalaya with `torch_cuda` backend for batched processing
- **Solver choice:** Use `KernelRidgeCV` (not `RidgeCV`) to access batch size arguments for GPU memory management

---

## Preprocessing Pipeline

The preprocessing pipeline transforms raw frame-by-frame behavioral data into features aligned with fMRI TRs, ready for encoding model analysis.

### Pipeline Overview

```
Raw replay JSONs (60 fps)
    ↓
[1] process_replay_sessions.py → Session-level combined replay variables
    ↓
[2] merge_bids_events_and_repvars.py → Run-level data with BIDS events merged
    ↓
[3] process_pruned_features.py → Pruned feature set with scene features and engineered variables
    ↓
[4] downsample_to_TR.py → Downsampled to TR (1.49s)
    ↓
[5] create_practice_phase_metadata.py → CV metadata for practice phase runs
    ↓
[6] fit_encoding_model.py → Fit ridge regression models
    ↓
[Post-fitting analysis scripts] → Significance testing, weight extraction, CIFTI creation
```

### Step-by-Step Execution

All scripts are located in `scripts/python/` and should be run from the project root:

#### Step 1: Merge replay variables per session
```bash
python scripts/python/process_replay_sessions.py
```
**Input:** Frame-by-frame replay variables (JSONs) for each level repetition  
**Output:** `inputs/per_session_combined_repvars_tsvs/`  
**Purpose:** Consolidate all replay data for each session into single TSV files

#### Step 2: Merge BIDS events with replay variables
```bash
python scripts/python/merge_bids_events_and_repvars.py
```
**Input:** Session-level replay variables + BIDS event annotations  
**Output:** `inputs/per_run_combined_repvars_and_bids_events/`  
**Purpose:** Align behavioral data with fMRI runs and add event markers (coins, deaths, etc.)

#### Step 3: Engineer and prune features
```bash
python scripts/python/process_pruned_features.py
```
**Input:** Run-level combined data + scene perceptual features  
**Output:** `inputs/per_run_pruned_features/`  
**Purpose:** 
- Merge scene-level perceptual features from mastersheet
- One-hot encode categorical variables (powerstate, movement direction)
- Extend event durations to match neural response timescales
- Create kill event features from RAM variables
- Drop redundant/unused columns

#### Step 4: Downsample to TR
```bash
python scripts/python/downsample_to_TR.py
```
**Input:** Frame-level features (~60 fps)  
**Output:** `inputs/per_run_downsampled_to_TR/`  
**Purpose:** Aggregate framewise data into TR windows (1.49s)
- Binary features (events, buttons, states) → averaged (proportion active)
- Continuous features (score, coins, enemies) → averaged within TR
- Adds `TR_index` and `TR_time` columns for alignment
- Fills gaps (ITI, post-task baseline) with zeros to match fMRI acquisition length

#### Step 5: Generate CV metadata for practice phase
```bash
python scripts/python/create_practice_phase_metadata.py
```
**Input:** BIDS events (to identify practice vs discovery phase) + downsampled features  
**Output:** `inputs/practice_phase_metadata/`  
**Purpose:** Create metadata for cross-validation
- Identify runs in practice phase (stable behavior, no learning effects)
- Compute `run_onsets` arrays for leave-one-run-out CV
- Compute `level_onsets` arrays for within-level z-scoring (if needed)
- Store run-level statistics (n_TRs, n_samples)

---

## Encoding Model Fitting

### Overview

The encoding model fitting pipeline trains voxelwise ridge regression models with FIR delays to predict fMRI responses from behavioral features. Models are trained on multiple sessions using leave-one-run-out cross-validation and evaluated on held-out test sessions.

### Preprocessing Steps (Applied in Order)

The fitting pipeline applies several preprocessing steps in a specific order to ensure valid cross-validation:

1. **Baseline period filtering** (optional, currently applied)
   - Removes TRs where all behavioral features are near-zero (`sum(|features|) < threshold`)
   - Rationale: Avoids gameplay-vs-rest confounds by excluding ITI/baseline periods
   - Applied separately to train and test sets

2. **Zero-variance feature filtering**
   - Identifies features with zero variance across entire training set
   - Removes these features from both train and test sets
   - Rationale: Zero-variance features provide no information for prediction

3. **Zero-variance voxel filtering**
   - Identifies voxels with zero variance in ANY training run
   - Removes these voxels from both train and test sets
   - Rationale: Voxels with zero variance in any run will cause issues during within-run z-scoring

4. **fMRI z-scoring (within runs)**
   - Z-scores fMRI responses (Y) separately within each run
   - Rationale: Removes run-level drift and offset while preserving within-run variability (Gallant Lab standard)

5. **Feature mean-centering only**
   - Features (X) are mean-centered via `StandardScaler(with_mean=True, with_std=False)`
   - **NOT z-scored**: Preserves meaningful relative scales between features
   - Rationale: Rare events shouldn't be artificially amplified; relative feature magnitudes are informative

6. **FIR delay application**
   - Applied via `Delayer` within sklearn pipeline (after StandardScaler, before ridge regression)
   - Creates lagged copies of features at delays [1, 2, 3, 4] TRs (~1.5-6 seconds)
   - Rationale: Captures hemodynamic response function (HRF) lag

**Critical:** Delayer is applied WITHIN the pipeline, not before creating the CV object, to avoid data leakage between folds.

### Model Architecture

- **Model type:** Ridge regression with L2 regularization
- **FIR delays:** [1, 2, 3, 4] TRs (captures HRF lag of ~1.5-6 seconds post-stimulus)
- **Regularization:** 30 alpha values on log scale from 10^0 to 10^6
- **Cross-validation:** Leave-one-run-out within training sessions
- **Solver:** Himalaya's `RidgeCV` with `torch_cuda` backend for GPU acceleration
  - Note: `KernelRidgeCV` provides batch size control for GPU memory management
  - `RidgeCV` is optimal when n_samples > n_features but lacks batch arguments
  - For large datasets, use `Y_in_cpu` and `force_cpu` arguments to manage memory

### Train/Test Session Split

Models are trained on multiple sessions and evaluated on held-out sessions:
- **Training sessions:** e.g., sessions 6-10 (concatenated)
- **Test sessions:** e.g., sessions 11-12 (held-out)
- **Within training:** Leave-one-run-out CV to select regularization per voxel
- **Generalization test:** Final evaluation on held-out sessions

This design tests whether brain-behavior mappings generalize to new data from different days.

### Usage

```bash
# Example: Train on sessions 6-10, test on session 11
python scripts/python/fit_encoding_model.py \
    --subject 1 \
    --session-range 6 10 \
    --test-sessions 11 \
    --backend torch_cuda

# Example: Train on specific sessions, test on multiple sessions
python scripts/python/fit_encoding_model.py \
    --subject 1 \
    --sessions 6 7 8 \
    --test-sessions 9 10 \
    --backend torch_cuda
```

### Outputs

All outputs are saved to `results/models/` and `results/cv_scores/`:

- `sub-{XX}_train-ses-{XXX}-{XXX}_test-ses-{XXX}_model.pkl`
  - Fitted sklearn pipeline (StandardScaler → Delayer → RidgeCV)
  
- `sub-{XX}_train-ses-{XXX}-{XXX}_test-ses-{XXX}_cv_scores.npy`
  - Cross-validation R² scores (shape: n_voxels)
  - Performance on training data using leave-one-run-out CV
  
- `sub-{XX}_train-ses-{XXX}-{XXX}_test-ses-{XXX}_test_scores.npy`
  - Test set R² scores (shape: n_voxels)
  - Performance on held-out test sessions
  
- `sub-{XX}_train-ses-{XXX}-{XXX}_test-ses-{XXX}_best_alphas.npy`
  - Selected regularization parameter per voxel (shape: n_voxels)
  
- `sub-{XX}_train-ses-{XXX}-{XXX}_test-ses-{XXX}_voxel_mask.npy`
  - Boolean mask (shape: 91282) indicating which grayordinates were kept after zero-variance filtering
  
- `sub-{XX}_train-ses-{XXX}-{XXX}_test-ses-{XXX}_feature_mask.npy`
  - Boolean mask indicating which features were kept after zero-variance filtering
  
- `sub-{XX}_train-ses-{XXX}-{XXX}_test-ses-{XXX}_feature_names.json`
  - Original feature names and which features survived filtering

---

## Post-Fitting Analysis (WIP)

### Significance Testing

**Script:** `compute_voxel_significance.py`  
**Status:** Work in progress

Computes p-values for each voxel to test whether prediction accuracy is significantly above chance. Two methods available:
- **Parametric test:** Uses Fisher's z-transform on correlations (fast)
- **Permutation test:** Builds null distribution by permuting (robust, slow)

Following Huth et al. (2012), voxels with p < 0.05 (uncorrected) are typically selected for further analysis.

### Weight Extraction

**Script:** `extract_model_weights.py`  
**Status:** Work in progress

Extracts regression weights from fitted models for interpretation. Weights reveal which features drive predictions for each voxel.

### Feature PCA

**Script:** `compute_feature_pca.py`  
**Status:** Work in progress

Performs PCA on regression weights across voxels to identify principal components of feature tuning. Useful for discovering interpretable feature dimensions.

### CIFTI Creation

**Script:** `create_model_cifti.py`  
**Status:** Work in progress

Converts model results (R² scores, weights) into CIFTI brain maps for visualization on cortical surface.

**Key steps:**
1. Load saved results (CV scores, test scores, voxel mask)
2. Create full-size arrays (91282 grayordinates) with zeros for filtered voxels
3. Insert valid voxel data using voxel mask
4. Load template CIFTI file to get header/geometry
5. Create new CIFTI with result data and template geometry
6. Save as `.dscalar.nii` (scalar data) or `.dtseries.nii` (timeseries)

**Usage:**
```bash
python scripts/python/create_model_cifti.py \
    --subject 1 \
    --train-sessions 6 7 8 9 10 \
    --test-sessions 11 \
    --metric cv_scores
```

**Outputs:** CIFTI files viewable in Connectome Workbench or other neuroimaging software

---

## Feature Engineering Details

### Categorical Encodings
- **Powerstate:** One-hot encoded (big, fire, star; small is reference)
- **Scrolling:** Binarized (screen scrolling vs. static)
- **Powerup visibility:** Binarized (visible vs. not visible)
- **Jump state:** Binarized (airborne vs. grounded)
- **Movement direction:** One-hot encoded (left, right; stationary is reference)

### Event Duration Extensions
Events are extended by specified frame counts to capture neural response timescales:
- Coin collected: 30 frames (~0.5s)
- Brick smashed: 30 frames (~0.5s)
- Powerup collected: 114 frames (~1.9s)
- Hit/powerup lost: 35 frames (~0.6s)
- Hit/life lost: 180 frames (~3.0s)

### Kill Events
Created from RAM variables (`enemy_kill30-34`):
- `event_kill_stomp` (value = 4)
- `event_kill_impact` (value = 34)
- `event_kill_kick` (value = 132)

### Scene Features
Merged from scenes mastersheet (`mario.scenes/sourcedata/scenes_info/scenes_mastersheet.csv`):
- Perceptual complexity metrics
- Spatial layout properties
- Object density statistics

---

## Methodological Choices

### Why not z-score behavioral features?

Features are mean-centered but NOT z-scored (standardized). 

**Rationale:** Preserving relative feature scales is important for interpretation. Z-scoring would artificially amplify rare events (e.g., collecting a star powerup) to have the same variance as common events (e.g., moving right), potentially distorting their true relationship to neural responses. The Gallant Lab standard approach is to z-score only fMRI responses (Y) within runs, while leaving features (X) raw or mean-centered only.

### Why z-score fMRI within runs?

fMRI responses (Y) are z-scored separately within each run.

**Rationale:** Removes run-level drift and baseline offset while preserving within-run variability. This is the Gallant Lab standard preprocessing approach and ensures that model fitting focuses on within-run response patterns rather than run-to-run baseline differences.

### Why filter baseline periods?

TRs where behavioral features are near-zero (ITI, post-task rest) are optionally removed.

**Rationale:** Including baseline periods could introduce gameplay-vs-rest confounds rather than capturing feature-specific selectivity. This step ensures the model learns from active gameplay data. Note: This is currently applied but could be made optional in future versions.

### Why filter zero-variance features and voxels?

Features with zero variance across training data and voxels with zero variance in any training run are removed.

**Rationale:** Zero-variance features provide no information for prediction. Zero-variance voxels in any run will cause numerical issues during within-run z-scoring. These filtering steps are applied to training data and the resulting masks are applied to test data to maintain valid train/test separation.

### Why use separate train and test sessions?

Models are trained on multiple sessions and evaluated on completely held-out sessions from different days.

**Rationale:** This tests whether brain-behavior mappings generalize across time and provides a more rigorous validation than leave-one-run-out CV alone. It mimics real prediction scenarios where models must generalize to new data.

---

## Configuration

Key parameters in `src/python/mario_encoding/config.py`:

```python
PARAMETERS = {
    'TR': 1.49,                                    # fMRI repetition time (seconds)
    'frame_rate': 60.099826520671044,              # Game frame rate (fps)
    'event_frame_duration_dict': {                 # Event extension durations
        'event_coin_collected': 30,
        'event_brick_smashed': 30,
        'event_powerup_collected': 114,
        'event_hit_powerup_lost': 35,
        'event_hit_life_lost': 180
    },
    'encoding_model': {
        'delays': [1, 2, 3, 4],                   # FIR delays (TRs)
        'alpha_min': 0,                            # log10(alpha_min)
        'alpha_max': 6,                            # log10(alpha_max)
        'n_alphas': 30,                            # Number of alpha values
        'solver_params': {
            'n_iter': 1000,
            'tol': 1e-4
        }
    }
}
```

---

## Data Quality Checks

Validation scripts in `scripts/python/`:
- `check_behav_data_files_exist.py` - Verify behavioral data integrity
- `check_fmri_data_files_exist.py` - Verify fMRI data availability
- `check_behav_fmri_alignment.py` - Validate temporal alignment
- `check_fmri_preprocessed_behav_files.py` - Check run-level file matching

Logs are saved to `results/logs/`.

---

## References

**Methodology:**
- Dupré la Tour, T., Visconti di Oleggio Castello, M., & Gallant, J.L. (2025). The Voxelwise Encoding Model framework: A tutorial introduction to fitting encoding models to fMRI data. *Imaging Neuroscience*, 3. https://doi.org/10.1162/imag_a_00575
- Huth, A.G., Nishimoto, S., Vu, A.T., & Gallant, J.L. (2012). A continuous semantic space describes the representation of thousands of object and action categories across the human brain. *Neuron*, 76(6), 1210-1224.

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
