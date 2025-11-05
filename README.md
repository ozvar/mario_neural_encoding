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
│   ├── figures/                     # Output visualizations
│   └── logs/                        # Processing logs
├── scripts/python/                  # Preprocessing pipeline scripts
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

### 3. Install the mario_encoding package
```bash
pip install -e src/python/
```

### 4. Configure paths
Edit `src/python/mario_encoding/config.py` to point to your data directories if they differ from the default sibling directory structure.

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
[Encoding model fitting] (in development)
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

#### Step 5: Generate CV metadata for practice phase
```bash
python scripts/python/create_practice_phase_metadata.py
```
**Input:** BIDS events (to identify practice vs discovery phase) + downsampled features  
**Output:** `inputs/practice_phase_metadata/`  
**Purpose:** Create metadata for cross-validation
- Identify runs in practice phase (stable behavior, no learning effects)
- Compute `run_onsets` arrays for leave-one-run-out CV
- Store run-level statistics (n_TRs, n_samples)

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

## Next Steps (In Development)

### Encoding Model Pipeline
6. **Load and concatenate data** - Stack practice phase runs per session
7. **Z-score within runs** - Normalize features and BOLD separately per run
8. **Apply FIR delays** - Create delayed features (1-4 TRs) to capture HRF
9. **Fit voxelwise ridge regression** - Leave-one-run-out cross-validation
10. **Evaluate predictions** - Compute R² on held-out test runs
11. **Visualize on cortical surface** - Project prediction accuracy to fsLR surface
12. **Interpret model weights** - PCA on regression weights to identify feature tunings

---

## References

**Methodology:**
- Dupré la Tour, T., Visconti di Oleggio Castello, M., & Gallant, J.L. (2025). The Voxelwise Encoding Model framework: A tutorial introduction to fitting encoding models to fMRI data. *Imaging Neuroscience*, 3. https://doi.org/10.1162/imag_a_00575

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
