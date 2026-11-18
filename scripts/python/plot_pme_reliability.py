"""Plot the Stage B PMe reliability audit tables.

Reads the tables and attempt-pattern archives written by
audit_pme_reliability.py and writes the audit figures. No audit statistic is
recomputed; the heatmap only displays the correlation matrix of the saved
attempt patterns.
"""
import argparse
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mario_encoding.config import PATHS

OUTCOME_COLORS = {
    'completed': '#0072B2',
    'failed': '#D55E00',
    'ambiguous': '#999999',
}
OUTCOME_ORDER = ['completed', 'failed', 'ambiguous']
MIN_MEDIAN_SPLIT_HALF = 0.50  # Reference line, matches the audit rule.
RANDOM_SEED = 42

ATTEMPT_TABLE = 'attempt_reliability.csv'
SUBJECT_TABLE = 'subject_reliability_summary.csv'
PATTERN_DIR = 'patterns'


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(output_dir / 'plot.log'),
                  logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def level_sort_key(level):
    """Sort levels such as w2l1 by world and level number."""
    match = re.match(r'w(\d+)l(\d+)', str(level))
    if match:
        return int(match.group(1)), int(match.group(2)), str(level)
    return 999, 999, str(level)


def load_tables(table_dir):
    """Load and validate the Stage B audit tables."""
    table_dir = Path(table_dir)
    attempt_path = table_dir / ATTEMPT_TABLE
    subject_path = table_dir / SUBJECT_TABLE
    missing = [path for path in (attempt_path, subject_path)
               if not path.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing Stage B audit tables: ' +
            ', '.join(str(path) for path in missing))

    attempt_df = pd.read_csv(attempt_path)
    subject_df = pd.read_csv(subject_path)
    required_attempt = {'subject', 'attempt_key', 'level', 'outcome',
                        'included', 'split_half_sb', 'n_active_trs_screen',
                        'session_tr_start'}
    required_subject = {'subject', 'same_level_r', 'different_level_r',
                        'p_all'}
    for name, table, required in [
            (attempt_path, attempt_df, required_attempt),
            (subject_path, subject_df, required_subject)]:
        missing_columns = sorted(required - set(table.columns))
        if missing_columns:
            raise ValueError(f'{name} missing columns: {missing_columns}')
    if attempt_df['attempt_key'].duplicated().any():
        raise ValueError(
            f'{attempt_path} has duplicated attempt_key values')
    return attempt_df, subject_df


def load_patterns(patterns_dir, attempt_df):
    """Load per-subject patterns and attach attempt timing from the table."""
    patterns_dir = Path(patterns_dir)
    paths = sorted(patterns_dir.glob('sub-*_attempt_patterns.npz'))
    if not paths:
        raise FileNotFoundError(
            f'No attempt-pattern archives in {patterns_dir}')

    start_by_key = attempt_df.set_index('attempt_key')['session_tr_start']
    patterns_by_subject = {}
    meta_by_subject = {}
    for path in paths:
        match = re.match(r'sub-(\d+)_attempt_patterns\.npz', path.name)
        if match is None:
            continue
        subject = int(match.group(1))
        with np.load(path) as data:
            patterns = data['patterns']
            meta = pd.DataFrame({
                'attempt_key': data['attempt_key'].astype(str),
                'level': data['level'].astype(str),
            })
        if len(patterns) != len(meta):
            raise ValueError(
                f'{path.name}: {len(patterns)} patterns for {len(meta)} rows')
        missing = sorted(set(meta['attempt_key']) - set(start_by_key.index))
        if missing:
            raise ValueError(
                f'{path.name}: attempt keys absent from {ATTEMPT_TABLE}: '
                f'{missing[:3]}')
        meta['session_tr_start'] = meta['attempt_key'].map(start_by_key)
        patterns_by_subject[subject] = patterns
        meta_by_subject[subject] = meta
    return patterns_by_subject, meta_by_subject


def plot_reliability_distribution(attempt_df, output_path):
    included = attempt_df[attempt_df['included']]
    subjects = sorted(included['subject'].unique())
    if not subjects:
        return
    data = [included.loc[included['subject'] == subject, 'split_half_sb']
            .dropna().to_numpy() for subject in subjects]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=[f'sub-{s:02d}' for s in subjects], widths=0.6)
    rng = np.random.default_rng(RANDOM_SEED)
    for index, subject in enumerate(subjects, start=1):
        subset = included[included['subject'] == subject]
        jitter = rng.uniform(-0.12, 0.12, size=len(subset))
        colors = [OUTCOME_COLORS.get(outcome, 'black')
                  for outcome in subset['outcome']]
        ax.scatter(np.full(len(subset), index) + jitter,
                   subset['split_half_sb'], s=12, c=colors, alpha=0.7)
    ax.axhline(MIN_MEDIAN_SPLIT_HALF, color='grey', linestyle='--',
               linewidth=0.9)
    ax.text(len(subjects) + 0.55, MIN_MEDIAN_SPLIT_HALF,
            f'{MIN_MEDIAN_SPLIT_HALF:.2f}', va='center', fontsize=7,
            color='grey')
    ax.set_ylabel('Split-half reliability (Spearman-Brown)')
    ax.set_title('Within-attempt BOLD pattern reliability', loc='left')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_same_vs_different(subject_df, output_path):
    subjects = subject_df['subject'].to_numpy()
    x = np.arange(len(subjects))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, subject_df['same_level_r'], width,
           color='#0072B2', label='same level')
    ax.bar(x + width / 2, subject_df['different_level_r'], width,
           color='#D55E00', label='different level')
    for index, row in subject_df.iterrows():
        p_value = row['p_all']
        label = 'p=NA' if not np.isfinite(p_value) else f'p={p_value:.3f}'
        top = np.nanmax([row['same_level_r'], row['different_level_r'], 0])
        ax.text(index, top + 0.03, label, ha='center', fontsize=7)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'sub-{int(s):02d}' for s in subjects])
    ax.set_ylabel('Mean pattern correlation')
    ax.set_title('Same-level versus different-level attempt patterns',
                 loc='left')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_active_vs_reliability(attempt_df, output_path):
    included = attempt_df[attempt_df['included']]
    fig, ax = plt.subplots(figsize=(6, 4))
    for outcome in OUTCOME_ORDER:
        subset = included[included['outcome'] == outcome]
        ax.scatter(subset['n_active_trs_screen'], subset['split_half_sb'],
                   s=16, alpha=0.7, color=OUTCOME_COLORS[outcome],
                   label=outcome)
    ax.axhline(MIN_MEDIAN_SPLIT_HALF, color='grey', linestyle='--',
               linewidth=0.9)
    ax.set_xlabel('Screen/Gameplay TRs in attempt')
    ax.set_ylabel('Split-half reliability (Spearman-Brown)')
    ax.set_title('Attempt length and pattern reliability', loc='left')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_similarity_heatmap(patterns_by_subject, meta_by_subject, output_path):
    if not patterns_by_subject:
        return
    subject = max(patterns_by_subject,
                  key=lambda key: len(patterns_by_subject[key]))
    patterns = patterns_by_subject[subject]
    meta = meta_by_subject[subject]
    if len(patterns) < 2:
        return
    order = sorted(
        range(len(meta)),
        key=lambda index: (level_sort_key(meta.iloc[index]['level']),
                           meta.iloc[index]['session_tr_start']))
    ordered = patterns[order]
    with np.errstate(invalid='ignore', divide='ignore'):
        correlation = np.corrcoef(ordered)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap='RdBu_r',
                      interpolation='nearest')
    ordered_levels = meta.iloc[order]['level'].astype(str).to_numpy()
    boundaries = [0]
    labels = [ordered_levels[0]]
    for index in range(1, len(ordered_levels)):
        if ordered_levels[index] != ordered_levels[index - 1]:
            boundaries.append(index)
            labels.append(ordered_levels[index])
    boundaries.append(len(ordered_levels))
    ticks = [(boundaries[i] + boundaries[i + 1] - 1) / 2
             for i in range(len(boundaries) - 1)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f'sub-{subject:02d}: attempt pattern similarity', loc='left')
    fig.colorbar(image, ax=ax, label='Pearson r')
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot the Stage B PMe reliability audit tables.')
    parser.add_argument(
        '--table-dir', type=Path,
        default=PATHS['variance_partitioning'].parent / 'pme_reliability')
    parser.add_argument(
        '--output-dir', type=Path,
        default=PATHS['figures'] / 'pme_reliability')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.table_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attempt_df, subject_df = load_tables(args.table_dir)
    patterns_by_subject, meta_by_subject = load_patterns(
        args.table_dir / PATTERN_DIR, attempt_df)
    logger.info('Tables: %s', args.table_dir)
    logger.info('Output: %s', args.output_dir)

    plot_reliability_distribution(
        attempt_df, args.output_dir / 'reliability_distribution.png')
    plot_same_vs_different(
        subject_df, args.output_dir / 'same_vs_different_similarity.png')
    plot_active_vs_reliability(
        attempt_df, args.output_dir / 'active_trs_vs_reliability.png')
    plot_similarity_heatmap(
        patterns_by_subject, meta_by_subject,
        args.output_dir / 'attempt_similarity_heatmap.png')

    logger.info('Done.')


if __name__ == '__main__':
    main()
