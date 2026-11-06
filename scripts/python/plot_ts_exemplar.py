"""
Plot empirical BOLD time series for exemplar vertices.

Saves one PNG and one SVG per panel to results/figures/BOLD_time_series/.

A single contiguous active window (--min-window TRs) is identified in the
display subject's test session and used for all panels, enabling direct
comparison across regions. The display subject is the subject with the highest
individual R2 at the 6a vertex, determined from saved R2 files.

Four panels: MMP areas 6a, VIP, MST (well-predicted) and one near-zero R2 vertex.

Usage:
    python scripts/python/plot_ts_exemplar.py --group GROUP_N5_PRIMARY
"""
import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY
from mario_encoding.utils import sns_styleset, save_figure, add_clean_end_ticks
from mario_encoding.utils.data_loading import (
    concatenate_sessions_with_names,
    select_and_validate_features,
    preprocess_data,
    compute_baseline_mask,
    apply_delays_per_run,
)
from mario_encoding.variance_partitioning import (
    expand_feature_space_indices_for_delays,
    create_feature_space_arrays,
)

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}

TR = PARAMETERS['TR']
WELL_PREDICTED_PARCELS = ['6a', 'VIP', 'MST']
BOLD_COLOR = '#792929'
PRED_COLOR = '#1a1a1a'
FONT_SIZE  = 8
TICK_SIZE  = 7

# fsLR 32k CIFTI: grayordinates 0-59411 are cortical (L+R hemispheres);
# 59412-91281 are subcortical structures.
N_CORTICAL = 59412


# ============================================================================
# Vertex selection
# ============================================================================

def find_parcel_vertex(parcel_name: str, group_r2: np.ndarray,
                       atlas_path: str) -> tuple:
    """Grayordinate with highest group R2 within the named MMP parcel."""
    img = nib.load(atlas_path)
    atlas_data = img.get_fdata().squeeze().astype(int)
    label_map = img.header.get_axis(0).label[0]
    matching_ids = [
        key for key, lt in label_map.items()
        if parcel_name.lower() in lt[0].lower()
    ]
    if not matching_ids:
        raise ValueError(f"No MMP parcel matching '{parcel_name}' found in atlas")
    parcel_indices = np.where(np.isin(atlas_data, matching_ids))[0]
    valid = parcel_indices[~np.isnan(group_r2[parcel_indices])]
    if len(valid) == 0:
        raise ValueError(f"No valid grayordinates in parcel '{parcel_name}'")
    best = int(valid[np.argmax(group_r2[valid])])
    return best, float(group_r2[best])


def select_poor_vertex(group_avg_dir: Path, poor_r2_target: float) -> tuple:
    """Cortical grayordinate whose group R2 is closest to poor_r2_target."""
    r2 = np.load(group_avg_dir / 'R2_scores.npz')['R2_full']
    cortical_mask = np.zeros(len(r2), dtype=bool)
    cortical_mask[:N_CORTICAL] = True
    valid_mask = cortical_mask & ~np.isnan(r2) & (r2 > 0)
    valid_indices = np.where(valid_mask)[0]
    local = int(np.argmin(np.abs(r2[valid_mask] - poor_r2_target)))
    grayord = int(valid_indices[local])
    return grayord, float(r2[grayord]), r2


# ============================================================================
# Subject selection
# ============================================================================

def _grayord_to_voxel_idx(grayordinate: int, valid_voxels_mask: np.ndarray):
    if not valid_voxels_mask[grayordinate]:
        return None
    return int(valid_voxels_mask[:grayordinate + 1].sum()) - 1


def get_vp_dir(subject: int, train_sessions: list, test_sessions: list,
               experiment_id: str) -> Path:
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    return (PATHS['variance_partitioning']
            / f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
            / experiment_id)


def select_display_subject(group_cfg: dict, grayordinate: int,
                            logger: logging.Logger) -> int:
    """
    Subject with highest per-subject R2 at grayordinate.
    Reads saved R2_scores.npz and valid_voxels_mask.npy; no prediction recomputation.
    """
    best_sub, best_r2 = None, -np.inf
    for sub_cfg in group_cfg['subjects']:
        subject = sub_cfg['subject']
        vp_dir = get_vp_dir(subject, sub_cfg['train_sessions'],
                            sub_cfg['test_sessions'], sub_cfg['experiment_id'])
        try:
            mask = np.load(vp_dir / 'valid_voxels_mask.npy')
            idx = _grayord_to_voxel_idx(grayordinate, mask)
            if idx is None:
                continue
            r2_arr = np.load(vp_dir / 'R2_scores.npz')['R2_full']
            if idx < len(r2_arr) and r2_arr[idx] > best_r2:
                best_r2, best_sub = r2_arr[idx], subject
        except (FileNotFoundError, KeyError):
            continue
    if best_sub is None:
        raise RuntimeError(
            f"Could not determine best subject for grayordinate {grayordinate}"
        )
    logger.info(f"Display subject: sub-{best_sub:02d} (R2={best_r2:.4f})")
    return best_sub


# ============================================================================
# Data loading
# ============================================================================

def load_fit_artifacts(vp_dir: Path) -> dict:
    return {
        'weights':            np.load(vp_dir / 'weights_full.npz')['weights_full'],
        'train_x_mean':       np.load(vp_dir / 'train_x_mean.npy'),
        'valid_voxels_mask':  np.load(vp_dir / 'valid_voxels_mask.npy'),
        'valid_features_mask': np.load(vp_dir / 'valid_features_mask.npy'),
    }


def compute_predictions(X_delayed_active: np.ndarray, weights: np.ndarray,
                        feature_names_kept: list,
                        logger: logging.Logger) -> np.ndarray:
    """Full-model predictions via space-wise weight slicing (preserves GroupRidgeCV layout)."""
    delays = PARAMETERS['variance_partitioning']['delays']
    feature_spaces_filtered = {
        space: [f for f in feats if f in feature_names_kept]
        for space, feats in FEATURE_SPACES.items()
    }
    space_to_indices = {
        space: np.array([feature_names_kept.index(f) for f in feats], dtype=int)
        for space, feats in feature_spaces_filtered.items()
    }
    space_to_indices_delayed = expand_feature_space_indices_for_delays(
        space_to_indices, len(feature_names_kept), delays, logger,
    )
    X_list, _ = create_feature_space_arrays(
        X_delayed_active, space_to_indices_delayed, feature_spaces_filtered, logger,
    )
    Y_hat = np.zeros((X_delayed_active.shape[0], weights.shape[1]), dtype=np.float32)
    start = 0
    for X_space in X_list:
        n = X_space.shape[1]
        Y_hat += (X_space @ weights[start:start + n, :]).astype(np.float32)
        start += n
    if start != weights.shape[0]:
        raise ValueError(
            f"Weight slicing mismatch: consumed {start} of {weights.shape[0]} rows"
        )
    return Y_hat


def find_best_prediction_window(pred: np.ndarray, pre_drop: np.ndarray,
                                min_window: int, onset_offset: int) -> int:
    """
    Return the start index in active-TR space of the min_window-TR window with
    the highest prediction variance, subject to:
      - all window TRs are within a single contiguous active stretch (no ITI gaps)
      - at least onset_offset TRs have elapsed since the start of that stretch
    """
    n_active = len(pred)
    best_var, best_start = -1.0, None

    for i in range(onset_offset, n_active - min_window + 1):
        # Onset offset: i must be onset_offset TRs into a contiguous stretch
        if pre_drop[i] - pre_drop[i - onset_offset] != onset_offset:
            continue
        # Contiguity: no ITI gaps within the window
        if pre_drop[i + min_window - 1] - pre_drop[i] != min_window - 1:
            continue
        var = float(np.var(pred[i:i + min_window]))
        if var > best_var:
            best_var, best_start = var, i

    if best_start is None:
        raise ValueError(
            f"No valid {min_window}-TR window with onset_offset={onset_offset} "
            "found within a contiguous active stretch"
        )
    return best_start


def load_display_data(subject: int, sub_cfg: dict, min_window: int,
                      onset_offset: int, ref_grayord: int,
                      logger: logging.Logger) -> dict:
    """
    Run the full encoding model pipeline for the display subject's test session.
    Selects the min_window-TR window that maximises prediction variance at
    ref_grayord, skipping onset_offset TRs at the start of each active stretch.

    Returns:
        Y_window          -- (min_window, 91282) empirical BOLD, z-scored
        Y_hat_window      -- (min_window, n_valid_voxels) predicted BOLD
        valid_voxels_mask -- (91282,) maps grayordinate to column index in Y_hat_window
    """
    pipeline = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']
    vp_dir = get_vp_dir(subject, sub_cfg['train_sessions'],
                        sub_cfg['test_sessions'], sub_cfg['experiment_id'])

    artifacts = load_fit_artifacts(vp_dir)

    X, Y, run_onsets, session_onsets, level_onsets, feature_names = \
        concatenate_sessions_with_names(
            subject, sub_cfg['test_sessions'], PATHS['practice_phase_metadata'],
            PATHS['per_run_downsampled_to_TR'], fmri_path, pipeline, logger,
        )

    X, feature_names_sel, _ = select_and_validate_features(
        X, feature_names, FEATURE_SPACES, logger
    )
    X = X[:, artifacts['valid_features_mask']]
    feature_names_kept = [
        n for n, k in zip(feature_names_sel, artifacts['valid_features_mask']) if k
    ]

    X, Y = preprocess_data(X, Y, run_onsets, level_onsets, logger)
    X = X - artifacts['train_x_mean'][None, :]

    active_mask = compute_baseline_mask(X, run_onsets, logger=logger)

    delays = PARAMETERS['variance_partitioning']['delays']
    X_delayed = apply_delays_per_run(X, run_onsets, delays)
    X_delayed_active = X_delayed[active_mask]

    Y_hat_active = compute_predictions(
        X_delayed_active, artifacts['weights'], feature_names_kept, logger
    )

    # Select window by prediction variance at the reference vertex
    ref_voxel_idx = _grayord_to_voxel_idx(ref_grayord, artifacts['valid_voxels_mask'])
    if ref_voxel_idx is None:
        raise ValueError(f"Reference grayordinate {ref_grayord} not in valid voxels mask")

    pre_drop = np.where(active_mask)[0]
    active_win_start = find_best_prediction_window(
        Y_hat_active[:, ref_voxel_idx], pre_drop, min_window, onset_offset
    )
    win_start = int(pre_drop[active_win_start])
    pred_var = float(np.var(Y_hat_active[active_win_start:active_win_start + min_window,
                                         ref_voxel_idx]))
    logger.info(
        f"Display window: active TRs {active_win_start}-{active_win_start + min_window - 1}, "
        f"pre-drop TRs {win_start}-{win_start + min_window - 1} "
        f"(t={win_start * TR:.1f}s), pred var={pred_var:.4f}"
    )

    return {
        'Y_window':          Y[win_start:win_start + min_window, :].copy(),
        'Y_hat_window':      Y_hat_active[active_win_start:active_win_start + min_window, :],
        'valid_voxels_mask': artifacts['valid_voxels_mask'],
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_timeseries(bold: np.ndarray,
                    predicted: np.ndarray | None = None) -> plt.Figure:
    t = np.arange(len(bold)) * TR
    fig, ax = plt.subplots(figsize=(5, 2), constrained_layout=True)
    ax.plot(t, bold, color=BOLD_COLOR, lw=1.5)
    if predicted is not None:
        ax.plot(t, predicted, color=PRED_COLOR, lw=1.5, ls='--')
    ax.set_ylim(-2.5, 2.5)
    add_clean_end_ticks(ax)
    ax.set_xlabel('Time (s)', fontsize=FONT_SIZE)
    ax.set_ylabel('BOLD (z)', fontsize=FONT_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    return fig


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--group', default='GROUP_N5_PRIMARY',
                        choices=list(GROUP_REGISTRY.keys()))
    parser.add_argument('--group-avg-timestamp', default='20260526_201255',
                        help='Subdirectory under group_average/ with R2_scores.npz')
    parser.add_argument('--min-window', type=int, default=30,
                        help='Active TRs to display (~45s at TR=1.49s)')
    parser.add_argument('--onset-offset', type=int, default=10,
                        help='TRs to skip at the start of each active stretch '
                             'to bypass the HRF onset transient (~15s)')
    parser.add_argument('--poor-r2', type=float, default=0.0,
                        help='Target group R2 for the poorly-predicted vertex')
    parser.add_argument('--output-dir', type=Path,
                        default=PATHS['figures'] / 'BOLD_time_series')
    return parser.parse_args()


def select_vertices(group_avg_dir: Path, poor_r2_target: float) -> dict:
    """
    Return ordered panel dict {label: (grayordinate, group_r2)}:
    one entry per well-predicted parcel plus the poor-R2 control vertex.
    """
    poor_grayord, poor_r2, group_r2 = select_poor_vertex(group_avg_dir, poor_r2_target)
    panels = {}
    for parcel in WELL_PREDICTED_PARCELS:
        grayord, r2_val = find_parcel_vertex(parcel, group_r2, PATHS['Glasser360'])
        panels[parcel] = (grayord, r2_val)
    panels['poor'] = (poor_grayord, poor_r2)
    return panels


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    sns_styleset()
    matplotlib.rcParams['svg.fonttype'] = 'none'

    group_cfg = GROUP_REGISTRY[args.group]
    group_avg_dir = (PATHS['variance_partitioning'] / 'group_average'
                     / args.group_avg_timestamp)

    panels = select_vertices(group_avg_dir, args.poor_r2)
    for label, (grayord, r2_val) in panels.items():
        print(f"  {label}: grayordinate={grayord}, group R2={r2_val:.4f}")

    ref_grayord = panels[WELL_PREDICTED_PARCELS[0]][0]
    display_subject = select_display_subject(group_cfg, ref_grayord, logger)
    display_sub_cfg = next(
        s for s in group_cfg['subjects'] if s['subject'] == display_subject
    )

    data = load_display_data(
        display_subject, display_sub_cfg, args.min_window, args.onset_offset,
        ref_grayord, logger
    )

    print(f"\nSaving to {args.output_dir}")
    for label, (grayord, r2_val) in panels.items():
        bold = data['Y_window'][:, grayord]
        voxel_idx = _grayord_to_voxel_idx(grayord, data['valid_voxels_mask'])
        predicted = (data['Y_hat_window'][:, voxel_idx]
                     if voxel_idx is not None else None)
        fig = plot_timeseries(bold, predicted)
        for ext in ('svg', 'png'):
            save_figure(fig, args.output_dir, f'{label}.{ext}')
        print(f"  {label}.{{svg,png}}  (grayordinate={grayord}, group R2={r2_val:.4f})")

    print("Done.")


if __name__ == '__main__':
    main()
