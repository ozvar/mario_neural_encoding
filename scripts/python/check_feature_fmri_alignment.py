"""
Check temporal alignment between behavioral features and fMRI timeseries.

Three checks, in increasing order of data requirements:

  1. TR count consistency (local): downsampled TSV row count vs metadata
     n_trs_per_run -- a mismatch causes silent mis-indexing even when the
     total sample count passes existing validation.

  2. Feature coverage timing (local): max frame_time_in_run from combined
     repvars TSV vs expected run duration (n_trs * TR). Also reports first
     rep_onset and largest within-run rep_onset gap.

  3. Cross-correlation peak lag (cluster, requires CIFTI): normalised
     cross-correlation between summed feature activity and mean fMRI signal
     of high-R2 voxels, computed per run. Expected peak at lags 2-4 TRs
     (HRF ~3-5s at TR=1.49s). A shifted peak in one session relative to
     another is suggestive of temporal misalignment, though not conclusive
     -- see script docstring notes on confounds.

Outputs:
  - alignment_timing.csv : per-run timing summary (checks 1 and 2)
  - xcorr_plot.png       : cross-correlation curves per session (check 3)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.utils.data_loading import get_fmri_filepath

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TR           = PARAMETERS['TR']
FEATURE_COLS = [f for space in sorted(FEATURE_SPACES) for f in FEATURE_SPACES[space]]

DEFAULT_R2_CIFTI = (
    Path('results/figures/model_ciftis')
    / 'sub-01_train-ses-007-012_test-ses-014-014'
    / '20260317_124704'
    / 'sub-01_train-ses-007-012_test-ses-014-014_20260317_124704_test_r2.dscalar.nii'
)


def load_metadata(subject, session, metadata_path):
    """Load practice metadata JSON for one session."""
    path = metadata_path / f'sub-{subject:02d}_ses-{session:03d}_practice_metadata.json'
    with open(path) as f:
        return json.load(f)


def downsampled_tsv_path(subject, session, run, downsampled_path):
    """Return path to downsampled TSV for one run."""
    return (
        downsampled_path
        / f'sub-{subject:02d}'
        / f'ses-{session:03d}'
        / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv'
    )


def combined_tsv_path(subject, session, run, combined_path):
    """Return path to combined repvars TSV for one run."""
    return (
        combined_path
        / f'sub-{subject:02d}'
        / f'ses-{session:03d}'
        / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-combined.tsv'
    )


# =============================================================================
# CHECK 1 -- TR count consistency
# =============================================================================

def tr_count_record(subject, session, run, run_idx, downsampled_path, meta):
    """Compare downsampled TSV row count to metadata n_trs_per_run for one run."""
    path     = downsampled_tsv_path(subject, session, run, downsampled_path)
    expected = meta['n_trs_per_run'][run_idx]
    actual   = len(pd.read_csv(path, sep='\t', usecols=['TR_index'])) if path.exists() else None

    if actual is None:
        logger.warning(f"Missing TSV: {path}")
    elif actual != expected:
        logger.warning(
            f"sub-{subject:02d} ses-{session:03d} run-{run:02d}: "
            f"TSV rows={actual}, metadata expected={expected}"
        )

    return {
        'subject': subject, 'session': session, 'run': run,
        'expected_trs': expected, 'actual_trs': actual,
        'tr_count_match': (actual == expected) if actual is not None else False,
    }


def check_tr_counts(subject, session, downsampled_path, metadata_path):
    """Return per-run TR count records for a session."""
    meta = load_metadata(subject, session, metadata_path)
    return [
        tr_count_record(subject, session, run, i, downsampled_path, meta)
        for i, run in enumerate(meta['practice_runs'])
    ]


# =============================================================================
# CHECK 2 -- Feature coverage timing
# =============================================================================

def timing_record(subject, session, run, run_idx, combined_path, meta):
    """Return timing stats for one run from combined repvars TSV."""
    path = combined_tsv_path(subject, session, run, combined_path)
    if not path.exists():
        logger.warning(f"Missing combined TSV: {path}")
        return {'subject': subject, 'session': session, 'run': run}

    df                = pd.read_csv(path, sep='\t', usecols=['rep_onset', 'frame_time_in_run'])
    expected_duration = meta['n_trs_per_run'][run_idx] * TR
    rep_onsets        = sorted(df['rep_onset'].unique())
    gaps              = [rep_onsets[i+1] - rep_onsets[i] for i in range(len(rep_onsets) - 1)]
    coverage_surplus  = df['frame_time_in_run'].max() - expected_duration

    if abs(coverage_surplus) > TR:
        logger.warning(
            f"sub-{subject:02d} ses-{session:03d} run-{run:02d}: "
            f"coverage_surplus={coverage_surplus:.2f}s (>{TR:.2f}s threshold)"
        )

    return {
        'subject': subject, 'session': session, 'run': run,
        'first_rep_onset':   round(rep_onsets[0], 3) if rep_onsets else np.nan,
        'max_frame_time':    round(df['frame_time_in_run'].max(), 3),
        'expected_duration': round(expected_duration, 3),
        'coverage_surplus':  round(coverage_surplus, 3),
        'n_reps':            len(rep_onsets),
        'max_rep_gap':       round(max(gaps), 3) if gaps else 0.0,
    }


def check_session_timing(subject, session, combined_path, metadata_path):
    """Return per-run timing records for a session."""
    meta = load_metadata(subject, session, metadata_path)
    return [
        timing_record(subject, session, run, i, combined_path, meta)
        for i, run in enumerate(meta['practice_runs'])
    ]


# =============================================================================
# CHECK 3 -- Cross-correlation peak lag
# =============================================================================

def load_r2_voxel_mask(r2_cifti_path, top_n):
    """Return boolean mask of top_n voxels by R2 from a dscalar CIFTI."""
    r2        = nib.load(str(r2_cifti_path)).get_fdata().squeeze()
    threshold = np.sort(r2)[-top_n]
    return r2 >= threshold


def load_feature_activity(tsv_path):
    """Return summed absolute feature activity per TR as 1D array."""
    df        = pd.read_csv(tsv_path, sep='\t')
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[feat_cols].abs().sum(axis=1).values.astype(float)


def compute_run_xcorr(fmri_data, feature_activity, max_lag):
    """Normalised cross-correlation between feature activity and mean fMRI at lags -max_lag..+max_lag."""
    y    = fmri_data.mean(axis=1) - fmri_data.mean()
    x    = feature_activity - feature_activity.mean()
    norm = np.std(x) * np.std(y) * len(x)
    lags = np.arange(-max_lag, max_lag + 1)

    if norm < 1e-10:
        return np.zeros(len(lags)), lags

    xcorr = np.array([
        np.dot(x[:len(x)-abs(lag)], y[abs(lag):]) / norm if lag >= 0
        else np.dot(x[abs(lag):], y[:len(y)-abs(lag)]) / norm
        for lag in lags
    ])
    return xcorr, lags


def xcorr_record(subject, session, run, pipeline, fmri_path,
                 downsampled_path, ref_voxel_mask, max_lag):
    """Compute cross-correlation record for one run. Returns None if data missing."""
    cifti_path = get_fmri_filepath(subject, session, run, fmri_path, pipeline)
    tsv_path   = downsampled_tsv_path(subject, session, run, downsampled_path)

    for path, label in [(cifti_path, 'CIFTI'), (tsv_path, 'TSV')]:
        if not path.exists():
            logger.warning(f"Missing {label}: {path} -- skipping xcorr")
            return None

    logger.info(f"  xcorr sub-{subject:02d} ses-{session:03d} run-{run:02d}")
    fmri_data   = nib.load(str(cifti_path)).get_fdata().astype(np.float32)
    fmri_ref    = fmri_data[:, ref_voxel_mask]
    feat_act    = load_feature_activity(tsv_path)
    n_trs       = min(len(feat_act), fmri_ref.shape[0])
    xcorr, lags = compute_run_xcorr(fmri_ref[:n_trs], feat_act[:n_trs], max_lag)
    peak_lag    = int(lags[np.argmax(xcorr)])

    logger.info(f"    Peak lag: {peak_lag} TRs ({peak_lag * TR:.2f}s)")
    return {
        'subject': subject, 'session': session, 'run': run,
        'peak_lag': peak_lag, 'xcorr': xcorr, 'lags': lags,
    }


def session_xcorr_records(subject, session, pipeline, fmri_path,
                           downsampled_path, metadata_path, ref_voxel_mask, max_lag):
    """Compute per-run cross-correlation records for a session."""
    meta    = load_metadata(subject, session, metadata_path)
    records = [
        xcorr_record(subject, session, run, pipeline, fmri_path,
                     downsampled_path, ref_voxel_mask, max_lag)
        for run in meta['practice_runs']
    ]
    return [r for r in records if r is not None]


# =============================================================================
# COLLECTION WRAPPERS
# =============================================================================

def collect_timing_records(subject, sessions, downsampled_path, combined_path, metadata_path):
    """Run checks 1 and 2 for all sessions, return (tr_records, timing_records)."""
    tr_records, timing_records = [], []
    for session in sessions:
        logger.info(f"Timing checks -- ses-{session:03d}")
        tr_records.extend(check_tr_counts(subject, session, downsampled_path, metadata_path))
        timing_records.extend(check_session_timing(subject, session, combined_path, metadata_path))
    return tr_records, timing_records


def collect_xcorr_records(subject, sessions, pipeline, fmri_path,
                           downsampled_path, metadata_path, ref_voxel_mask, max_lag):
    """Run check 3 for all sessions, return list of xcorr records."""
    records = []
    for session in sessions:
        logger.info(f"Cross-correlation -- ses-{session:03d}")
        records.extend(session_xcorr_records(
            subject, session, pipeline, fmri_path,
            downsampled_path, metadata_path, ref_voxel_mask, max_lag,
        ))
    return records


# =============================================================================
# OUTPUTS
# =============================================================================

def save_timing_csv(tr_records, timing_records, output_path):
    """Merge TR count and timing records and save to CSV."""
    df = pd.DataFrame(tr_records).merge(
        pd.DataFrame(timing_records).drop(columns=['subject', 'session', 'run'], errors='ignore'),
        left_index=True, right_index=True, how='left',
    )
    df.to_csv(output_path, index=False)
    logger.info(f"Saved timing CSV : {output_path}")


def log_timing_summary(tr_records, timing_records, sessions):
    """Log per-session timing anomaly summary."""
    all_match = all(r['tr_count_match'] for r in tr_records if r['actual_trs'] is not None)
    logger.info(f"TR count match (all runs): {all_match}")
    for ses in sessions:
        t = [r for r in timing_records if r['session'] == ses and 'coverage_surplus' in r]
        if not t:
            continue
        logger.info(
            f"  ses-{ses:03d} : "
            f"coverage_surplus={[round(r['coverage_surplus'], 2) for r in t]}s  "
            f"first_rep_onset={[round(r['first_rep_onset'], 2) for r in t]}s"
        )


def log_xcorr_summary(xcorr_records):
    """Log peak lag per run."""
    logger.info("Cross-correlation peak lags:")
    for r in xcorr_records:
        logger.info(
            f"  ses-{r['session']:03d} run-{r['run']:02d} : "
            f"peak_lag={r['peak_lag']} TRs ({r['peak_lag'] * TR:.2f}s)"
        )


def make_xcorr_plot(all_records, sessions, max_lag, output_path):
    """Cross-correlation curves per run, one panel per session."""
    fig, axes = plt.subplots(1, len(sessions), figsize=(6 * len(sessions), 4),
                              sharey=True, sharex=True)
    if len(sessions) == 1:
        axes = [axes]

    for ax, ses in zip(axes, sessions):
        ses_recs = [r for r in all_records if r['session'] == ses]
        if not ses_recs:
            ax.set_title(f'ses-{ses:03d} (no data)')
            continue
        for r in ses_recs:
            ax.plot(r['lags'], r['xcorr'], alpha=0.7,
                    label=f"run-{r['run']:02d} (peak={r['peak_lag']})")
        ax.axvspan(2, 4, alpha=0.12, color='green', label='expected lag')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_title(f'ses-{ses:03d}')
        ax.set_xlabel('Lag (TRs)')
        ax.legend(fontsize=7)

    axes[0].set_ylabel('Normalised cross-correlation')
    subject = all_records[0]['subject']
    fig.suptitle(
        f"sub-{subject:02d} feature-fMRI cross-correlation "
        f"(top-R2 voxels, max_lag={max_lag})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved xcorr plot : {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Check feature-fMRI temporal alignment')
    parser.add_argument('--subject',    type=int,  required=True)
    parser.add_argument('--sessions',   type=int,  nargs='+', required=True)
    parser.add_argument('--top-voxels', type=int,  default=500)
    parser.add_argument('--max-lag',    type=int,  default=8)
    parser.add_argument('--r2-cifti',   type=Path, default=DEFAULT_R2_CIFTI)
    parser.add_argument('--output-dir', type=Path, default=PATHS['figures'] / 'alignment_check')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline  = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']
    prefix    = f'sub-{args.subject:02d}'

    logger.info("=" * 70)
    logger.info("FEATURE-fMRI TEMPORAL ALIGNMENT CHECK")
    logger.info("=" * 70)
    logger.info(f"Subject   : {args.subject}")
    logger.info(f"Sessions  : {args.sessions}")
    logger.info(f"Pipeline  : {pipeline}")

    # =========================================================================
    # CHECKS 1 & 2 -- TR counts and timing (local)
    # =========================================================================
    tr_records, timing_records = collect_timing_records(
        args.subject, args.sessions,
        PATHS['per_run_downsampled_to_TR'],
        PATHS['per_run_combined_repvars_and_bids_events'],
        PATHS['practice_phase_metadata'],
    )
    log_timing_summary(tr_records, timing_records, args.sessions)
    save_timing_csv(tr_records, timing_records,
                    args.output_dir / f'{prefix}_alignment_timing.csv')

    # =========================================================================
    # CHECK 3 -- Cross-correlation (cluster)
    # =========================================================================
    if not args.r2_cifti.exists():
        logger.warning(f"R2 CIFTI not found: {args.r2_cifti} -- skipping cross-correlation")
        logger.info("Done (timing checks only).")
        return

    logger.info(f"Loading R2 voxel mask (top {args.top_voxels} voxels)...")
    ref_voxel_mask = load_r2_voxel_mask(args.r2_cifti, args.top_voxels)

    xcorr_records = collect_xcorr_records(
        args.subject, args.sessions, pipeline, fmri_path,
        PATHS['per_run_downsampled_to_TR'],
        PATHS['practice_phase_metadata'],
        ref_voxel_mask, args.max_lag,
    )

    if xcorr_records:
        log_xcorr_summary(xcorr_records)
        make_xcorr_plot(xcorr_records, args.sessions, args.max_lag,
                        args.output_dir / f'{prefix}_xcorr_plot.png')

    logger.info("Done.")


if __name__ == '__main__':
    main()
