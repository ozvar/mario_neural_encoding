"""
Screen all practice sessions for feature-fMRI temporal misalignment.

For each run across all subjects and sessions, computes the cross-correlation
peak lag between summed feature activity and the mean BOLD signal across all
grayordinates. Flags runs whose peak lag falls outside the expected HRF range
(--lag-min to --lag-max TRs). Expected peak: 2-4 TRs (~3-6s at TR=1.49s).

Uses global mean fMRI signal as reference -- weaker than subject-specific R2
voxels but unbiased across subjects and sessions. Sufficient to detect severe
misalignment. For targeted per-subject analysis with subject-specific reference
voxels, use check_feature_fmri_alignment.py instead.

Both signals are linearly detrended and high-pass filtered (zero-phase, default
0.01 Hz) before correlating. Without this, low-frequency drift in the global mean
signal and the blocky gameplay/ITI structure of the feature signal dominate, making
the cross-correlation a broad ramp whose argmax reflects drift phase rather than the
HRF. Filtering is zero-phase (filtfilt) so it cannot itself shift the peak.

Peak height is reported alongside peak lag. An argmax always exists even when the
cross-correlation is flat noise, so lag alone cannot distinguish a misaligned run
from one with no reliable peak; compare `peak_r` against `r_at_expected` (the best
correlation inside the expected HRF range) before excluding any run.

Must be run on the cluster where HCP CIFTI files are accessible.

Outputs:
  - xcorr_all_runs.csv  : per-run peak lag, peak height, and flagged status
  - xcorr_peak_lags.png : peak lag per run (top) and peak height (bottom)
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
from scipy.signal import butter, detrend, filtfilt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.utils.data_loading import get_fmri_filepath

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TR           = PARAMETERS['TR']
FEATURE_COLS = [f for space in sorted(FEATURE_SPACES) for f in FEATURE_SPACES[space]]


# =============================================================================
# SESSION DISCOVERY
# =============================================================================

def discover_practice_sessions(metadata_path):
    """Return (subject, session) pairs where all runs are practice phase."""
    sessions = []
    for json_path in sorted(metadata_path.glob('sub-*_ses-*_practice_metadata.json')):
        parts   = json_path.stem.split('_')
        subject = int(parts[0].replace('sub-', ''))
        session = int(parts[1].replace('ses-', ''))
        with open(json_path) as f:
            meta = json.load(f)
        if any(p != 'practice' for p in meta.get('run_phases', {}).values()):
            continue
        sessions.append((subject, session))
    logger.info(f"Discovered {len(sessions)} all-practice sessions")
    return sessions


def load_metadata(subject, session, metadata_path):
    """Load practice metadata JSON for one session."""
    path = metadata_path / f'sub-{subject:02d}_ses-{session:03d}_practice_metadata.json'
    with open(path) as f:
        return json.load(f)


# =============================================================================
# SIGNAL LOADING
# =============================================================================

def downsampled_tsv_path(subject, session, run, downsampled_path):
    """Return path to downsampled feature TSV for one run."""
    return (
        downsampled_path
        / f'sub-{subject:02d}'
        / f'ses-{session:03d}'
        / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv'
    )


def load_feature_activity(tsv_path):
    """Return summed absolute feature activity per TR as 1D array."""
    df        = pd.read_csv(tsv_path, sep='\t')
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[feat_cols].abs().sum(axis=1).values.astype(float)


def load_fmri_reference(cifti_path):
    """Return mean BOLD signal across all grayordinates (n_trs,)."""
    return nib.load(str(cifti_path)).get_fdata().astype(np.float32).mean(axis=1)


def highpass_filter(signal: np.ndarray, tr: float, cutoff_hz: float,
                    order: int = 2) -> np.ndarray:
    """Linearly detrend then zero-phase high-pass filter a 1D signal.

    Zero-phase (filtfilt) is required: a causal filter would impose its own group
    delay and shift the cross-correlation peak, which is the quantity of interest.
    Returns the detrended signal unfiltered if it is too short for filtfilt padding.
    """
    x       = detrend(np.asarray(signal, dtype=float), type='linear')
    nyquist = 0.5 / tr
    wn      = cutoff_hz / nyquist

    if not 0 < wn < 1:
        logger.warning(
            f"High-pass cutoff {cutoff_hz} Hz invalid at TR={tr} -- skipping filter"
        )
        return x

    b, a   = butter(order, wn, btype='highpass')
    padlen = 3 * max(len(a), len(b))
    if len(x) <= padlen:
        logger.warning(f"Signal too short ({len(x)} TRs) to filter -- detrend only")
        return x

    return filtfilt(b, a, x)


# =============================================================================
# CROSS-CORRELATION
# =============================================================================

def compute_xcorr(feature_activity, fmri_signal, max_lag):
    """Normalised cross-correlation at lags -max_lag..+max_lag."""
    x    = feature_activity - feature_activity.mean()
    y    = fmri_signal - fmri_signal.mean()
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


def summarise_xcorr(xcorr: np.ndarray, lags: np.ndarray,
                    lag_min: int, lag_max: int) -> dict:
    """Peak lag and height, plus the best height inside the expected HRF range.

    `r_margin` is how much the global peak beats the best expected-range lag. A small
    margin means the run is not statistically distinguishable from correctly aligned,
    regardless of where the argmax happens to fall.
    """
    peak_idx     = int(np.argmax(xcorr))
    peak_lag     = int(lags[peak_idx])
    peak_r       = float(xcorr[peak_idx])
    expected     = (lags >= lag_min) & (lags <= lag_max)
    r_at_expected = float(xcorr[expected].max()) if expected.any() else np.nan

    return {
        'peak_lag':      peak_lag,
        'peak_r':        peak_r,
        'r_at_expected': r_at_expected,
        'r_margin':      peak_r - r_at_expected,
        'at_boundary':   abs(peak_lag) == int(lags.max()),
    }


def run_peak_lag(subject, session, run, pipeline, fmri_path,
                 downsampled_path, max_lag, lag_min, lag_max,
                 highpass_hz, tr):
    """Compute cross-correlation summary for one run. Returns None if data missing."""
    cifti_path = get_fmri_filepath(subject, session, run, fmri_path, pipeline)
    tsv_path   = downsampled_tsv_path(subject, session, run, downsampled_path)

    for path, label in [(cifti_path, 'CIFTI'), (tsv_path, 'TSV')]:
        if not path.exists():
            logger.warning(f"Missing {label}: {path} -- skipping")
            return None

    fmri_ref = load_fmri_reference(cifti_path)
    feat_act = load_feature_activity(tsv_path)
    n_trs    = min(len(feat_act), len(fmri_ref))
    feat_act, fmri_ref = feat_act[:n_trs], fmri_ref[:n_trs]

    if highpass_hz is not None:
        feat_act = highpass_filter(feat_act, tr, highpass_hz)
        fmri_ref = highpass_filter(fmri_ref, tr, highpass_hz)

    xcorr, lags = compute_xcorr(feat_act, fmri_ref, max_lag)
    return summarise_xcorr(xcorr, lags, lag_min, lag_max)


def session_peak_lags(subject, session, pipeline, fmri_path,
                      downsampled_path, metadata_path, max_lag,
                      lag_min, lag_max, highpass_hz, tr):
    """Return per-run cross-correlation records for a session."""
    meta    = load_metadata(subject, session, metadata_path)
    records = []
    for run in meta['practice_runs']:
        logger.info(f"  sub-{subject:02d} ses-{session:03d} run-{run:02d}")
        summary = run_peak_lag(
            subject, session, run, pipeline, fmri_path,
            downsampled_path, max_lag, lag_min, lag_max, highpass_hz, tr,
        )
        if summary is not None:
            records.append({
                'subject': subject, 'session': session, 'run': run, **summary,
            })
    return records


def collect_all_peak_lags(sessions, pipeline, fmri_path, downsampled_path,
                           metadata_path, max_lag, lag_min, lag_max,
                           highpass_hz, tr):
    """Compute cross-correlation records for all sessions."""
    all_records = []
    for subject, session in sessions:
        logger.info(f"Processing sub-{subject:02d} ses-{session:03d}...")
        all_records.extend(session_peak_lags(
            subject, session, pipeline, fmri_path,
            downsampled_path, metadata_path, max_lag,
            lag_min, lag_max, highpass_hz, tr,
        ))
    return all_records


# =============================================================================
# OUTPUTS
# =============================================================================

def flag_runs(df, lag_min, lag_max):
    """Add flagged column for runs with peak_lag outside expected range."""
    df['flagged'] = ~df['peak_lag'].between(lag_min, lag_max)
    return df


def save_csv(df, output_path):
    """Save per-run peak lag table sorted by peak_lag."""
    df.sort_values('peak_lag').to_csv(output_path, index=False)
    logger.info(f"Saved CSV : {output_path}")


def log_flagged_summary(df, lag_min, lag_max):
    """Log flagged runs and summary counts."""
    n_flagged = df['flagged'].sum()
    logger.info(
        f"Flagged runs : {n_flagged} / {len(df)} "
        f"({n_flagged / len(df) * 100:.1f}%) outside lag range [{lag_min}, {lag_max}]"
    )
    n_boundary = int(df['at_boundary'].sum())
    if n_boundary:
        logger.info(
            f"At search boundary (|lag| == max_lag) : {n_boundary} -- these have no "
            f"interior maximum and are 'no reliable peak', not a measured lag"
        )
    logger.info(
        f"Peak height : median r={df['peak_r'].median():.3f}, "
        f"median margin over expected range={df['r_margin'].median():.3f}"
    )

    flagged = df[df['flagged']].sort_values('peak_lag')
    for _, row in flagged.iterrows():
        boundary = ' [AT BOUNDARY]' if row['at_boundary'] else ''
        logger.info(
            f"  sub-{row['subject']:02d} ses-{row['session']:03d} run-{row['run']:02d} : "
            f"peak_lag={row['peak_lag']} TRs ({row['peak_lag'] * TR:.2f}s), "
            f"r={row['peak_r']:.3f}, r_expected={row['r_at_expected']:.3f}, "
            f"margin={row['r_margin']:.3f}{boundary}"
        )


def make_peak_lag_plot(df, lag_min, lag_max, max_lag, output_path):
    """Peak lag per run (top) over peak height vs expected-range height (bottom).

    The height panel is the interpretive control on the lag panel: a bar in the top
    panel is only evidence of misalignment if the bottom panel shows the peak
    actually beats the expected range by a visible margin.
    """
    labels = [
        f"s{row.subject:02d}_{row.session:03d}_r{row.run:02d}"
        for row in df.itertuples()
    ]
    colors = ['red' if row.flagged else 'steelblue' for row in df.itertuples()]
    x      = range(len(df))

    fig, (ax_lag, ax_r) = plt.subplots(
        2, 1, figsize=(max(12, len(df) * 0.18), 8), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
    )

    ax_lag.bar(x, df['peak_lag'], color=colors, edgecolor='none', width=0.8)
    ax_lag.axhspan(lag_min, lag_max, alpha=0.12, color='green',
                   label=f'expected [{lag_min},{lag_max}]')
    ax_lag.axhline(0, color='gray', linewidth=0.5)
    ax_lag.set_ylabel('Peak lag (TRs)')
    ax_lag.set_ylim(-max_lag - 1, max_lag + 1)
    ax_lag.set_title('Feature-fMRI cross-correlation peak lag and height (all runs)')
    ax_lag.legend(fontsize=8)

    ax_r.bar(x, df['peak_r'], color=colors, edgecolor='none', width=0.8,
             label='peak r')
    ax_r.plot(x, df['r_at_expected'], linestyle='none', marker='_',
              markersize=4, color='black', label=f'best r in [{lag_min},{lag_max}]')
    ax_r.axhline(0, color='gray', linewidth=0.5)
    ax_r.set_ylabel('Cross-correlation r')
    ax_r.set_xticks(list(x))
    ax_r.set_xticklabels(labels, rotation=90, fontsize=5)
    ax_r.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved plot : {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Screen all sessions for feature-fMRI temporal misalignment'
    )
    parser.add_argument('--lag-min',    type=int, default=1,
                        help='Minimum expected peak lag in TRs (default: 1)')
    parser.add_argument('--lag-max',    type=int, default=5,
                        help='Maximum expected peak lag in TRs (default: 5)')
    parser.add_argument('--max-lag',    type=int, default=20,
                        help='Max lag to compute cross-correlation over (default: 20)')
    parser.add_argument('--highpass-hz', type=float, default=0.01,
                        help='Zero-phase high-pass cutoff in Hz (default: 0.01)')
    parser.add_argument('--no-highpass', action='store_true',
                        help='Disable detrending and high-pass filtering (legacy behaviour)')
    parser.add_argument('--output-dir', type=Path,
                        default=PATHS['figures'] / 'xcorr_screening')
    args = parser.parse_args()

    highpass_hz = None if args.no_highpass else args.highpass_hz

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline  = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']

    logger.info("=" * 70)
    logger.info("CROSS-CORRELATION SCREENING -- ALL SESSIONS")
    logger.info("=" * 70)
    logger.info(f"Expected lag range : [{args.lag_min}, {args.lag_max}] TRs")
    logger.info(f"Max lag            : {args.max_lag} TRs")
    logger.info(f"fMRI reference     : global mean (all grayordinates)")
    logger.info(
        f"High-pass          : "
        + ("disabled (legacy)" if highpass_hz is None
           else f"detrend + {highpass_hz} Hz zero-phase")
    )
    logger.info(f"Output             : {args.output_dir}")

    # =========================================================================
    # SETUP
    # =========================================================================
    sessions = discover_practice_sessions(PATHS['practice_phase_metadata'])

    # =========================================================================
    # COMPUTE
    # =========================================================================
    all_records = collect_all_peak_lags(
        sessions, pipeline, fmri_path,
        PATHS['per_run_downsampled_to_TR'],
        PATHS['practice_phase_metadata'],
        args.max_lag, args.lag_min, args.lag_max,
        highpass_hz, TR,
    )

    if not all_records:
        logger.error("No records collected -- check CIFTI paths")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    df = flag_runs(df, args.lag_min, args.lag_max)

    # =========================================================================
    # OUTPUTS
    # =========================================================================
    log_flagged_summary(df, args.lag_min, args.lag_max)
    save_csv(df, args.output_dir / 'xcorr_all_runs.csv')
    make_peak_lag_plot(df, args.lag_min, args.lag_max, args.max_lag,
                       args.output_dir / 'xcorr_peak_lags.png')

    logger.info("Done.")


if __name__ == '__main__':
    main()
