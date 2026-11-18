"""Plot the Stage A PMe audit tables.

Reads the attempt, level and subject CSV outputs of audit_pme_epochs.py and
writes the audit figures. No audit computation is repeated.
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
REPETITION_LEVELS = [1, 2, 3, 4, 5]
DURATION_MARKS = [8, 10, 15]
RANDOM_SEED = 0

ATTEMPT_TABLE = 'attempt_table.csv'
LEVEL_TABLE = 'level_repetition_summary.csv'
SUBJECT_TABLE = 'subject_repetition_summary.csv'


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
    """Load and validate the three Stage A audit tables."""
    table_dir = Path(table_dir)
    paths = {
        'attempt': table_dir / ATTEMPT_TABLE,
        'level': table_dir / LEVEL_TABLE,
        'subject': table_dir / SUBJECT_TABLE,
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing Stage A audit tables: ' + ', '.join(str(p) for p in missing))

    attempt_df = pd.read_csv(paths['attempt'])
    level_df = pd.read_csv(paths['level'])
    subject_df = pd.read_csv(paths['subject'])

    required_attempt = {'subject', 'level', 'outcome', 'session_tr_start',
                        'session_tr_end', 'n_trs', 'screen_gameplay_available',
                        'active_fraction_proxy'}
    required_level = {'subject', 'level', 'n_attempts'}
    required_subject = ({'subject'} |
                        {f'n_levels_ge{k}' for k in REPETITION_LEVELS} |
                        {f'n_{outcome}' for outcome in OUTCOME_ORDER})
    for name, table, required in [
            ('attempt', attempt_df, required_attempt),
            ('level', level_df, required_level),
            ('subject', subject_df, required_subject)]:
        missing_columns = sorted(required - set(table.columns))
        if missing_columns:
            raise ValueError(
                f'{paths[name]} missing columns: {missing_columns}')
    if (attempt_df['screen_gameplay_available'].any() and
            'active_fraction_screen' not in attempt_df.columns):
        raise ValueError(
            f'{paths["attempt"]} reports Screen/Gameplay data but has no '
            'active_fraction_screen column')
    return attempt_df, level_df, subject_df


def plot_timelines(attempt_df, output_path):
    subjects = sorted(attempt_df['subject'].unique())
    fig, axes = plt.subplots(len(subjects), 1, figsize=(12, 2.4 * len(subjects)),
                             squeeze=False)
    for ax, subject in zip(axes[:, 0], subjects):
        subject_df = attempt_df[attempt_df['subject'] == subject]
        levels = sorted(subject_df['level'].unique(), key=level_sort_key)
        level_to_y = {level: idx for idx, level in enumerate(levels)}
        for _, row in subject_df.iterrows():
            y = level_to_y[row['level']]
            ax.hlines(y, row['session_tr_start'], row['session_tr_end'],
                      color=OUTCOME_COLORS.get(row['outcome'], '#999999'),
                      linewidth=5, alpha=0.9)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels(levels, fontsize=7)
        ax.set_xlabel('TR within held-out session')
        ax.set_title(f'sub-{subject:02d}', loc='left')
        ax.grid(axis='x', alpha=0.2)
    handles = [plt.Line2D([0], [0], color=OUTCOME_COLORS[o], lw=5, label=o)
               for o in OUTCOME_ORDER]
    fig.legend(handles=handles, loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_repetition_heatmap(level_df, output_path):
    pivot = level_df.pivot(index='level', columns='subject',
                           values='n_attempts').fillna(0)
    pivot = pivot.loc[sorted(pivot.index, key=level_sort_key)]
    fig, ax = plt.subplots(figsize=(6.5, max(6, 0.32 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect='auto', cmap='viridis',
                      vmin=0, vmax=max(1, int(pivot.to_numpy().max())))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f'sub-{int(s):02d}' for s in pivot.columns],
                       rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = int(pivot.iloc[i, j])
            if value > 0:
                ax.text(j, i, str(value), ha='center', va='center',
                        color='white' if value < image.norm.vmax * 0.6 else 'black',
                        fontsize=7)
    ax.set_title('Attempts per level in the held-out session', loc='left')
    fig.colorbar(image, ax=ax, label='Attempts')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_cumulative_repetitions(subject_df, output_path):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for _, row in subject_df.iterrows():
        counts = [row[f'n_levels_ge{k}'] for k in REPETITION_LEVELS]
        ax.plot(REPETITION_LEVELS, counts, marker='o',
                label=f"sub-{int(row['subject']):02d}")
    ax.set_xlabel('Minimum attempts per level (k)')
    ax.set_ylabel('Number of levels with >= k attempts')
    ax.set_xticks(REPETITION_LEVELS)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Levels available for repeated-epoch analyses', loc='left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_duration_distribution(attempt_df, output_path):
    subjects = sorted(attempt_df['subject'].unique())
    data = [attempt_df.loc[attempt_df['subject'] == s, 'n_trs'].dropna().to_numpy()
            for s in subjects]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=[f'sub-{s:02d}' for s in subjects], widths=0.6)
    rng = np.random.default_rng(RANDOM_SEED)
    for idx, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, s=8,
                   color='black', alpha=0.35)
    for mark in DURATION_MARKS:
        ax.axhline(mark, color='grey', linestyle='--', linewidth=0.8)
        ax.text(len(subjects) + 0.55, mark, f'{mark} TRs', va='center',
                fontsize=7, color='grey')
    ax.set_ylabel('Attempt duration (TRs)')
    ax.set_title('Attempt durations in the held-out session', loc='left')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_outcome_counts(subject_df, output_path):
    subjects = subject_df['subject'].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bottom = np.zeros(len(subjects))
    for outcome in OUTCOME_ORDER:
        values = subject_df[f'n_{outcome}'].to_numpy()
        ax.bar([f'sub-{int(s):02d}' for s in subjects], values,
               bottom=bottom, color=OUTCOME_COLORS[outcome], label=outcome)
        bottom += values
    ax.set_ylabel('Attempts')
    ax.set_title('Attempt outcomes in the held-out session', loc='left')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_active_fraction(attempt_df, output_path):
    column = ('active_fraction_screen'
              if attempt_df['screen_gameplay_available'].any()
              else 'active_fraction_proxy')
    label = ('Screen/Gameplay active fraction'
             if column == 'active_fraction_screen'
             else 'Feature-activity active fraction (proxy)')
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for outcome in OUTCOME_ORDER:
        subset = attempt_df[attempt_df['outcome'] == outcome]
        ax.scatter(subset['n_trs'], subset[column], s=14, alpha=0.7,
                   color=OUTCOME_COLORS[outcome], label=outcome)
    ax.set_xlabel('Attempt duration (TRs)')
    ax.set_ylabel(label)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Active TRs per attempt', loc='left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot the Stage A PMe audit tables.')
    parser.add_argument(
        '--table-dir', type=Path,
        default=PATHS['variance_partitioning'].parent / 'pme_audit')
    parser.add_argument(
        '--output-dir', type=Path,
        default=PATHS['figures'] / 'pme_audit')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.table_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attempt_df, level_df, subject_df = load_tables(args.table_dir)
    logger.info('Tables: %s', args.table_dir)
    logger.info('Output: %s', args.output_dir)

    plot_timelines(attempt_df, args.output_dir / 'attempt_timelines.png')
    plot_repetition_heatmap(level_df,
                           args.output_dir / 'repetition_heatmap.png')
    plot_cumulative_repetitions(
        subject_df, args.output_dir / 'cumulative_repetitions.png')
    plot_duration_distribution(
        attempt_df, args.output_dir / 'duration_distribution.png')
    plot_outcome_counts(
        subject_df, args.output_dir / 'outcome_counts.png')
    plot_active_fraction(
        attempt_df, args.output_dir / 'active_fraction.png')

    logger.info('Done.')


if __name__ == '__main__':
    main()
