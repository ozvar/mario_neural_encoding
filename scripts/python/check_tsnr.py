"""
Compare tSNR across sessions to diagnose fMRI data quality differences.

Computes per-run tSNR (mean/std over time) from raw HCP CIFTI timeseries,
both over all TRs and over ITI-only TRs (where all features are zero).
Intended for diagnosing unexpectedly low model R2 in a given test session.

Outputs:
  - tsnr_per_run.csv         : per-run mean tSNR (full and ITI)
  - tsnr_distributions.png   : violin plot comparing sessions
  - tsnr_per_run.png         : per-run mean tSNR bar chart
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

from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.utils.data_loading import get_fmri_filepath

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASELINE_THRESHOLD = 0.01
MIN_ITI_TRS        = 10


def load_practice_runs(subject, session, metadata_path):
    """Return list of practice run indices for a session."""
    json_path = metadata_path / f'sub-{subject:02d}_ses-{session:03d}_practice_metadata.json'
    with open(json_path) as f:
        metadata = json.load(f)
    return metadata['practice_runs']


def load_iti_mask(subject, session, run, downsampled_path):
    """Return boolean array (n_trs,) where True = ITI TR, derived from feature TSV."""
    tsv_path = (
        downsampled_path
        / f'sub-{subject:02d}'
        / f'ses-{session:03d}'
        / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv'
    )
    if not tsv_path.exists():
        logger.warning(f"Missing TSV: {tsv_path}")
        return None
    df           = pd.read_csv(tsv_path, sep='\t')
    feature_cols = [c for c in df.columns
                    if c not in {'TR_index', 'TR_time', 'score', 'time', 'coins'}]
    activity     = df[feature_cols].abs().sum(axis=1).values
    return activity <= BASELINE_THRESHOLD


def compute_run_tsnr(fmri_data, iti_mask):
    """Compute full-run and ITI-only tSNR from (n_trs, n_grayordinates) array."""
    mu_full  = fmri_data.mean(axis=0)
    sd_full  = fmri_data.std(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        tsnr_full = np.where(sd_full > 0, mu_full / sd_full, np.nan)

    tsnr_iti = None
    if iti_mask is not None:
        n_iti = iti_mask.sum()
        if n_iti >= MIN_ITI_TRS:
            iti_data = fmri_data[iti_mask]
            mu_iti   = iti_data.mean(axis=0)
            sd_iti   = iti_data.std(axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                tsnr_iti = np.where(sd_iti > 0, mu_iti / sd_iti, np.nan)
        else:
            logger.warning(f"Only {n_iti} ITI TRs -- skipping ITI tSNR")

    return tsnr_full, tsnr_iti


def session_tsnr_records(subject, session, pipeline, fmri_path,
                         metadata_path, downsampled_path):
    """Compute per-run tSNR records for all runs in a session."""
    runs    = load_practice_runs(subject, session, metadata_path)
    records = []

    for run in runs:
        cifti_path = get_fmri_filepath(subject, session, run, fmri_path, pipeline)
        if not cifti_path.exists():
            logger.warning(f"Missing CIFTI: {cifti_path} -- skipping")
            continue

        logger.info(f"  Loading sub-{subject:02d} ses-{session:03d} run-{run:02d}")
        fmri_data = nib.load(str(cifti_path)).get_fdata().astype(np.float32)

        iti_mask = load_iti_mask(subject, session, run, downsampled_path)
        if iti_mask is not None and len(iti_mask) != fmri_data.shape[0]:
            logger.warning(
                f"  TR count mismatch: CIFTI={fmri_data.shape[0]}, "
                f"TSV={len(iti_mask)} -- skipping ITI mask"
            )
            iti_mask = None

        tsnr_full, tsnr_iti = compute_run_tsnr(fmri_data, iti_mask)

        records.append({
            'subject':        subject,
            'session':        session,
            'run':            run,
            'n_trs':          fmri_data.shape[0],
            'n_iti_trs':      int(iti_mask.sum()) if iti_mask is not None else 0,
            'mean_tsnr_full': float(np.nanmean(tsnr_full)),
            'med_tsnr_full':  float(np.nanmedian(tsnr_full)),
            'mean_tsnr_iti':  float(np.nanmean(tsnr_iti)) if tsnr_iti is not None else np.nan,
            'med_tsnr_iti':   float(np.nanmedian(tsnr_iti)) if tsnr_iti is not None else np.nan,
            'tsnr_full':      tsnr_full,
            'tsnr_iti':       tsnr_iti,
        })

    return records


def make_distribution_plot(all_records, subject, sessions, output_path):
    """Violin plots of per-grayordinate tSNR distributions, full and ITI, per session."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, key, title in zip(
        axes,
        ['tsnr_full', 'tsnr_iti'],
        ['Full run tSNR', 'ITI-only tSNR'],
    ):
        plot_data   = []
        plot_labels = []
        for ses in sessions:
            for r in [rec for rec in all_records if rec['session'] == ses]:
                vals = r[key]
                if vals is None:
                    continue
                plot_data.append(vals[np.isfinite(vals)])
                plot_labels.append(f"ses-{ses:03d} run-{r['run']:02d}")

        if not plot_data:
            ax.set_title(f'{title} (no data)')
            continue

        ax.violinplot(plot_data, showmedians=True, showextrema=False)
        ax.set_xticks(range(1, len(plot_labels) + 1))
        ax.set_xticklabels(plot_labels, fontsize=7, rotation=45, ha='right')
        ax.set_title(title)
        ax.set_ylabel('tSNR')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    ses_str = ' vs '.join(f'ses-{s:03d}' for s in sessions)
    fig.suptitle(f"sub-{subject:02d} tSNR -- {ses_str}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved distribution plot : {output_path}")


def make_per_run_plot(df, output_path):
    """Bar chart of mean tSNR per run, full and ITI side by side."""
    labels = [f"ses-{row.session:03d}\nrun-{row.run:02d}" for row in df.itertuples()]
    x      = np.arange(len(df))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.7), 5))
    ax.bar(x - width/2, df['mean_tsnr_full'], width, label='Full run', color='steelblue')
    ax.bar(x + width/2, df['mean_tsnr_iti'],  width, label='ITI only',  color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mean tSNR')
    ax.set_title('Mean tSNR per run')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved per-run plot : {output_path}")


def save_csv(df, output_path):
    """Save per-run tSNR summary table."""
    cols = ['subject', 'session', 'run', 'n_trs', 'n_iti_trs',
            'mean_tsnr_full', 'med_tsnr_full', 'mean_tsnr_iti', 'med_tsnr_iti']
    df[cols].to_csv(output_path, index=False)
    logger.info(f"Saved CSV : {output_path}")


def log_summary(df, sessions):
    """Log per-session mean tSNR summary."""
    for ses in sessions:
        sub = df[df['session'] == ses]
        if sub.empty:
            continue
        logger.info(
            f"  ses-{ses:03d} : "
            f"mean_tsnr_full={sub['mean_tsnr_full'].mean():.2f}  "
            f"runs={sub['mean_tsnr_full'].round(2).tolist()}  "
            f"mean_tsnr_iti={sub['mean_tsnr_iti'].mean():.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description='Compare tSNR across sessions')
    parser.add_argument('--subject',    type=int,   required=True)
    parser.add_argument('--sessions',   type=int,   nargs='+', required=True,
                        help='Sessions to compare (e.g. --sessions 14 21)')
    parser.add_argument('--output-dir', type=Path,
                        default=PATHS['figures'] / 'tsnr_check')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline  = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']

    logger.info("=" * 70)
    logger.info("tSNR QUALITY CHECK")
    logger.info("=" * 70)
    logger.info(f"Subject   : {args.subject}")
    logger.info(f"Sessions  : {args.sessions}")
    logger.info(f"Pipeline  : {pipeline}")
    logger.info(f"Output    : {args.output_dir}")

    # =========================================================================
    # COMPUTE tSNR PER RUN
    # =========================================================================
    all_records = []
    for session in args.sessions:
        logger.info(f"Processing ses-{session:03d}...")
        records = session_tsnr_records(
            args.subject, session, pipeline, fmri_path,
            PATHS['practice_phase_metadata'],
            PATHS['per_run_downsampled_to_TR'],
        )
        all_records.extend(records)
        logger.info(f"  {len(records)} runs processed")

    if not all_records:
        logger.error("No records collected -- check data paths")
        sys.exit(1)

    # =========================================================================
    # SUMMARY AND OUTPUTS
    # =========================================================================
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ('tsnr_full', 'tsnr_iti')}
        for r in all_records
    ])

    logger.info("Per-session tSNR summary:")
    log_summary(df, args.sessions)

    prefix = f'sub-{args.subject:02d}'
    save_csv(df, args.output_dir / f'{prefix}_tsnr_per_run.csv')
    make_distribution_plot(all_records, args.subject, args.sessions,
                           args.output_dir / f'{prefix}_tsnr_distributions.png')
    make_per_run_plot(df, args.output_dir / f'{prefix}_tsnr_per_run.png')

    logger.info("Done.")


if __name__ == '__main__':
    main()
