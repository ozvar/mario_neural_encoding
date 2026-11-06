"""
Per-participant descriptive statistics for the downsampled behavioral features.

Feature values are proportion-active within each TR, so they are bounded in
[0, 1] but concentrated on the endpoints. Density estimates such as violins
misrepresent that structure, so the distribution is summarized as an explicit
endpoint composition: the mean over active TRs, plus the fractions of TRs at
exactly 0, strictly inside (0, 1), and at exactly 1.

The CSV also carries `bernoulli_ratio` = sd / sqrt(p(1-p)), which indexes
sub-TR flickering: near 1 means the feature is effectively switch-like after TR
averaging, lower means the proportion-active encoding is carrying graded
information a binary encoding would lose.

Usage:
    python scripts/python/summarize_feature_distributions.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mario_encoding.config import PATHS, FEATURE_SPACES
from mario_encoding.group_configs import GROUP_N5_PRIMARY
from mario_encoding.utils.data_loading import (
    compute_baseline_mask,
    load_practice_features_with_names,
    load_practice_metadata,
)
from mario_encoding.utils.viz_utils import save_figure

FONT_SIZE = 8
TICK_SIZE = 7
ZERO_VARIANCE_TOL = 1e-10


# ============================================================================
# Data loading
# ============================================================================

def load_subject_active_features(
    subject: int,
    sessions: list[int],
    practice_metadata_path: Path,
    downsampled_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """Pool active-TR features across one subject's training and test sessions."""
    X_list = []
    feature_names = None

    for session in sessions:
        metadata = load_practice_metadata(subject, session, practice_metadata_path)
        X_sess, feat_names = load_practice_features_with_names(
            subject, session, metadata['practice_runs'], downsampled_path
        )
        if feature_names is None:
            feature_names = feat_names
        elif feat_names != feature_names:
            raise ValueError(
                f"sub-{subject:02d} ses-{session:03d} feature order differs from first session"
            )
        X_list.append(X_sess)

    X = np.vstack(X_list)
    active_mask = compute_baseline_mask(X, run_onsets=None, threshold=0.01)
    return X[active_mask].astype(float), feature_names


def select_and_order_features(
    X: np.ndarray,
    feature_names: list[str],
    feature_spaces: dict,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Keep only configured features; return data, names, and per-feature space labels."""
    ordered_names, space_labels = [], []
    for space in ('motor', 'scene'):
        for feature in feature_spaces[space]:
            if feature in feature_names:
                ordered_names.append(feature)
                space_labels.append(space)

    col_idx = [feature_names.index(f) for f in ordered_names]
    return X[:, col_idx], ordered_names, space_labels


# ============================================================================
# Statistics
# ============================================================================

def summarize_features(
    X: np.ndarray,
    feature_names: list[str],
    space_labels: list[str],
    subject: int,
) -> pd.DataFrame:
    """Per-feature endpoint composition and dispersion over active TRs."""
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    p_zero = (X == 0).mean(axis=0)
    p_one = (X == 1).mean(axis=0)
    p_interior = 1.0 - p_zero - p_one

    # sd / sqrt(p(1-p)): 1 means switch-like after TR averaging, lower means graded.
    bernoulli_sd = np.sqrt(np.clip(mean * (1.0 - mean), 0.0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        bernoulli_ratio = np.where(bernoulli_sd > 0, sd / bernoulli_sd, np.nan)

    return pd.DataFrame({
        'subject': subject,
        'feature': feature_names,
        'space': space_labels,
        'mean': mean,
        'sd': sd,
        'p_zero': p_zero,
        'p_interior': p_interior,
        'p_one': p_one,
        'n_unique': [len(np.unique(X[:, j])) for j in range(X.shape[1])],
        'bernoulli_ratio': bernoulli_ratio,
        'zero_variance': sd ** 2 <= ZERO_VARIANCE_TOL,
    })


def order_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Motor block then scene block, each sorted by descending mean."""
    return pd.concat([
        summary[summary['space'] == space].sort_values('mean', ascending=False)
        for space in ('motor', 'scene')
    ], ignore_index=True)


# ============================================================================
# Plotting
# ============================================================================

def plot_feature_summary(
    summary: pd.DataFrame,
    subject: int,
    output_dir: Path,
) -> None:
    """Save the mean-activity and endpoint-composition panels (SVG + PNG)."""
    ordered = order_for_display(summary)
    n_features = len(ordered)
    y = np.arange(n_features)
    palette = sns.color_palette('muted')
    n_motor = int((ordered['space'] == 'motor').sum())

    fig, axes = plt.subplots(
        1, 2, figsize=(9, 0.19 * n_features + 1.2), sharey=True,
        gridspec_kw={'width_ratios': [1.0, 1.3]}, layout='constrained',
    )

    ax = axes[0]
    ax.barh(y, ordered['mean'], color=[
        palette[0] if s == 'motor' else palette[2] for s in ordered['space']
    ], height=0.75)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Mean activity over active TRs', fontsize=FONT_SIZE)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered['feature'], fontsize=TICK_SIZE)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.invert_yaxis()

    ax = axes[1]
    left = np.zeros(n_features)
    for column, colour, label in (
        ('p_zero', palette[7], 'exactly 0'),
        ('p_interior', palette[1], 'interior (0, 1)'),
        ('p_one', palette[3], 'exactly 1'),
    ):
        ax.barh(y, ordered[column], left=left, color=colour, label=label, height=0.75)
        left = left + ordered[column].to_numpy()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction of active TRs', fontsize=FONT_SIZE)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False)

    # Separator between the motor and scene blocks.
    for ax in axes:
        ax.axhline(n_motor - 0.5, color='0.15', linewidth=1.0, linestyle='--')

    fig.suptitle(f'sub-{subject:02d}  feature distributions over active TRs',
                 fontsize=FONT_SIZE)

    stem = f'sub-{subject:02d}_feature_distributions'
    for ext in ('svg', 'png'):
        save_figure(fig, output_dir, f'{stem}.{ext}', dpi=200)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--output-dir', type=Path,
                        default=PATHS['figures'] / 'feature_summary')
    args = parser.parse_args()

    all_summaries = []
    for subj_cfg in GROUP_N5_PRIMARY['subjects']:
        subject = subj_cfg['subject']
        sessions = list(subj_cfg['train_sessions']) + list(subj_cfg['test_sessions'])

        X, feature_names = load_subject_active_features(
            subject, sessions, PATHS['practice_phase_metadata'],
            PATHS['per_run_downsampled_to_TR'],
        )
        X, feature_names, space_labels = select_and_order_features(
            X, feature_names, FEATURE_SPACES
        )
        summary = summarize_features(X, feature_names, space_labels, subject)
        plot_feature_summary(summary, subject, args.output_dir)

        summary.to_csv(
            args.output_dir / f'sub-{subject:02d}_feature_distribution_stats.csv',
            index=False,
        )
        all_summaries.append(summary)

    pooled = pd.concat(all_summaries, ignore_index=True)
    pooled.to_csv(
        args.output_dir / f"{GROUP_N5_PRIMARY['output_label']}_feature_distribution_stats.csv",
        index=False,
    )


if __name__ == '__main__':
    main()
