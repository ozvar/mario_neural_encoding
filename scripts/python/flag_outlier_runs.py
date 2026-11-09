"""
Flag outlier runs based on per-run feature content distribution.

For each practice-only run across all subjects and sessions, computes the
mean of each feature (active TRs only). Z-scores each feature across all
runs pooled. Flags runs where any feature mean exceeds abs(z) > threshold.

Outputs:
  - outlier_runs.csv        : per-run table with z-scores and flags
  - zscore_heatmap.png      : runs x features z-score heatmap sorted by max |z|
  - feature_flag_counts.png : per-feature count of flagged runs
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from mario_encoding.config import PATHS, FEATURE_SPACES

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASELINE_THRESHOLD = 0.01


def get_all_feature_names():
    """Flat list of features from FEATURE_SPACES in sorted-space order."""
    names = []
    for space in sorted(FEATURE_SPACES.keys()):
        names.extend(FEATURE_SPACES[space])
    return names


def discover_practice_sessions(metadata_path):
    """Return (subject, session) pairs where all runs are practice phase."""
    sessions = []
    skipped = []

    for json_path in sorted(metadata_path.glob('sub-*_ses-*_practice_metadata.json')):
        parts   = json_path.stem.split('_')
        subject = int(parts[0].replace('sub-', ''))
        session = int(parts[1].replace('ses-', ''))

        with open(json_path) as f:
            metadata = json.load(f)

        if any(p != 'practice' for p in metadata.get('run_phases', {}).values()):
            skipped.append((subject, session))
            continue

        sessions.append((subject, session))

    logger.info(f"Discovered {len(sessions)} all-practice sessions")
    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} sessions with discovery runs: "
            + ', '.join(f"sub-{s:02d} ses-{e:03d}" for s, e in skipped)
        )
    return sessions


def run_feature_means(subject, session, metadata_path, downsampled_path, feature_names):
    """Load one session's runs and return per-run feature mean records."""
    json_path = metadata_path / f'sub-{subject:02d}_ses-{session:03d}_practice_metadata.json'
    with open(json_path) as f:
        metadata = json.load(f)

    records = []
    for run in metadata['practice_runs']:
        tsv_path = (
            downsampled_path
            / f'sub-{subject:02d}'
            / f'ses-{session:03d}'
            / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv'
        )
        if not tsv_path.exists():
            logger.warning(f"Missing TSV: {tsv_path} -- skipping")
            continue

        df      = pd.read_csv(tsv_path, sep='\t')
        avail   = [f for f in feature_names if f in df.columns]
        missing = set(feature_names) - set(df.columns)

        if missing:
            logger.warning(
                f"sub-{subject:02d} ses-{session:03d} run-{run:02d}: "
                f"missing features {sorted(missing)}"
            )

        feature_data = df[avail].values.astype(float)
        active_mask  = np.abs(feature_data).sum(axis=1) > BASELINE_THRESHOLD
        n_active     = active_mask.sum()

        if n_active == 0:
            logger.warning(
                f"sub-{subject:02d} ses-{session:03d} run-{run:02d}: "
                "no active TRs -- skipping"
            )
            continue

        mean_vec = feature_data[active_mask].mean(axis=0)

        record = {'subject': subject, 'session': session, 'run': run,
                  'n_active_trs': int(n_active)}
        record.update(dict(zip(avail, mean_vec)))
        record.update({f: np.nan for f in missing})
        records.append(record)

    return records


def collect_all_run_records(sessions, metadata_path, downsampled_path, feature_names):
    """Collect per-run feature mean records across all sessions."""
    all_records = []
    for subject, session in sessions:
        all_records.extend(
            run_feature_means(subject, session, metadata_path, downsampled_path, feature_names)
        )
    return all_records


def compute_zscores(df_runs, feature_names):
    """Z-score each feature pooled across all runs; drop near-zero-variance features."""
    df_z    = pd.DataFrame(index=df_runs.index)
    dropped = []

    for feat in feature_names:
        if feat not in df_runs.columns:
            continue
        sigma = df_runs[feat].std()
        if sigma < 1e-6:
            dropped.append(feat)
            continue
        df_z[feat] = (df_runs[feat] - df_runs[feat].mean()) / sigma

    if dropped:
        logger.warning(f"Dropped {len(dropped)} near-zero-variance features: {dropped}")

    return df_z


def annotate_outliers(df_runs, df_z, threshold):
    """Add max_abs_z, flagged, flagged_features, and per-feature flag columns to df_runs."""
    z_cols = list(df_z.columns)

    df_runs['max_abs_z']       = df_z.abs().max(axis=1)
    df_runs['flagged']         = df_runs['max_abs_z'] > threshold
    df_runs['flagged_features'] = df_z.apply(
        lambda row: [f for f in z_cols if abs(row[f]) > threshold], axis=1
    )
    for feat in z_cols:
        df_runs[f'flagged_{feat}'] = df_z[feat].abs() > threshold

    return df_runs


def log_flagged_summary(df_runs, threshold):
    """Log count of flagged runs and top 10 by max |z|."""
    n_flagged = df_runs['flagged'].sum()
    logger.info(
        f"Flagged runs : {n_flagged} / {len(df_runs)} "
        f"({n_flagged / len(df_runs) * 100:.1f}%) at threshold {threshold}"
    )
    flagged = df_runs[df_runs['flagged']].sort_values('max_abs_z', ascending=False)
    if len(flagged):
        logger.info("Top flagged runs:")
        for _, row in flagged.head(10).iterrows():
            logger.info(
                f"  sub-{row['subject']:02d} ses-{row['session']:03d} run-{row['run']:02d} "
                f"max_z={row['max_abs_z']:.2f}  features={row['flagged_features']}"
            )


def save_csv(df_runs, output_dir):
    """Save per-run outlier table sorted by max |z|."""
    cols = ['subject', 'session', 'run', 'n_active_trs', 'max_abs_z', 'flagged', 'flagged_features']
    path = output_dir / 'outlier_runs.csv'
    df_runs[cols].sort_values('max_abs_z', ascending=False).to_csv(path, index=False)
    logger.info(f"Saved CSV : {path}")


def make_heatmap(df_runs, df_z, threshold, output_path):
    """Z-score heatmap: rows = runs sorted by max |z|, flagged rows in red."""
    z_cols     = list(df_z.columns)
    sort_order = df_z.abs().max(axis=1).sort_values(ascending=False).index
    z_sorted   = df_z.loc[sort_order]

    run_labels = [
        f"sub-{df_runs.loc[i, 'subject']:02d} "
        f"ses-{df_runs.loc[i, 'session']:03d} "
        f"run-{df_runs.loc[i, 'run']:02d}"
        for i in sort_order
    ]

    fig, ax = plt.subplots(
        figsize=(max(12, len(z_cols) * 0.35), max(8, len(sort_order) * 0.22))
    )
    sns.heatmap(
        z_sorted.values, ax=ax,
        cmap='RdBu_r', vmin=-3, vmax=3, center=0,
        xticklabels=z_cols, yticklabels=run_labels,
        linewidths=0, cbar_kws={'label': 'Z-score'},
    )
    ax.set_title(f'Per-run feature mean Z-scores (threshold = {threshold})', fontsize=12)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Run')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=6)

    flagged_idx = set(df_runs.index[df_runs['flagged']])
    for tick, idx in zip(ax.get_yticklabels(), sort_order):
        if idx in flagged_idx:
            tick.set_color('red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap : {output_path}")


def make_feature_flag_counts(df_runs, output_path):
    """Bar chart: number of flagged runs per feature."""
    flag_cols = [c for c in df_runs.columns
                 if c.startswith('flagged_') and c != 'flagged_features']
    if not flag_cols:
        return

    counts       = df_runs[flag_cols].sum().sort_values(ascending=False)
    counts.index = [c.replace('flagged_', '') for c in counts.index]

    fig, ax = plt.subplots(figsize=(max(10, len(counts) * 0.4), 5))
    counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('Number of flagged runs per feature')
    ax.set_xlabel('Feature')
    ax.set_ylabel('N flagged runs')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved feature flag counts : {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Flag outlier runs by feature content distribution')
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='Absolute z-score threshold for flagging (default: 2.0)')
    parser.add_argument('--output-dir', type=Path, default=PATHS['figures'] / 'outlier_runs',
                        help='Output directory for CSV and figures')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RUN-LEVEL OUTLIER DETECTION")
    logger.info("=" * 70)
    logger.info(f"Z-score threshold : {args.threshold}")
    logger.info(f"Output directory  : {args.output_dir}")

    feature_names = get_all_feature_names()
    logger.info(f"Features          : {len(feature_names)} from FEATURE_SPACES")

    sessions = discover_practice_sessions(PATHS['practice_phase_metadata'])

    all_records = collect_all_run_records(
        sessions, PATHS['practice_phase_metadata'],
        PATHS['per_run_downsampled_to_TR'], feature_names,
    )
    if not all_records:
        logger.error("No run records collected -- check data paths")
        sys.exit(1)

    df_runs = pd.DataFrame(all_records).reset_index(drop=True)
    logger.info(f"Collected {len(df_runs)} runs across {len(sessions)} sessions")

    df_z    = compute_zscores(df_runs, feature_names)
    df_runs = annotate_outliers(df_runs, df_z, args.threshold)

    log_flagged_summary(df_runs, args.threshold)
    save_csv(df_runs, args.output_dir)
    make_heatmap(df_runs, df_z, args.threshold, args.output_dir / 'zscore_heatmap.png')
    make_feature_flag_counts(df_runs, args.output_dir / 'feature_flag_counts.png')

    logger.info("Done.")


if __name__ == '__main__':
    main()
