#!/usr/bin/env python3
"""
Compute and visualize the behavioral feature correlation matrix for GROUP_N5_PRIMARY.

Pools active (non-ITI) TRs across all subjects' training sessions and computes
Pearson correlations across the 45 motor+scene features. Features are ordered
by semantic group: motor first, then scene in config order.

Usage:
    python check_feature_correlations.py [--output-dir PATH]
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mario_encoding.config import PATHS, FEATURE_SPACES
from mario_encoding.group_configs import GROUP_N5_PRIMARY
from mario_encoding.utils.data_loading import (
    load_practice_metadata,
    load_practice_features_with_names,
    compute_baseline_mask,
)
from mario_encoding.utils.viz_utils import save_figure

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
logger = logging.getLogger(__name__)


def load_subject_active_features(
    subject: int,
    train_sessions: list[int],
    paths: dict,
) -> tuple[np.ndarray, list[str]]:
    """Load and baseline-mask features for one subject's training sessions."""
    X_list = []
    feature_names = None

    for session in train_sessions:
        metadata = load_practice_metadata(subject, session, paths['practice_phase_metadata'])
        X_sess, feat_names = load_practice_features_with_names(
            subject, session, metadata['practice_runs'], paths['per_run_downsampled_to_TR']
        )
        if feature_names is None:
            feature_names = feat_names
        X_list.append(X_sess)

    X = np.vstack(X_list)
    active_mask = compute_baseline_mask(X, run_onsets=None, threshold=0.01)
    n_active = int(active_mask.sum())
    logger.info(f"  sub-{subject:02d}: {n_active}/{len(active_mask)} active TRs "
                f"({n_active / len(active_mask) * 100:.1f}%)")
    return X[active_mask], feature_names


def select_and_order_features(
    X: np.ndarray,
    feature_names: list[str],
    feature_spaces: dict,
) -> tuple[np.ndarray, list[str]]:
    """Reorder columns to config order: motor first, then scene."""
    ordered_names = (
        [f for f in feature_spaces['motor'] if f in feature_names]
        + [f for f in feature_spaces['scene'] if f in feature_names]
    )
    col_idx = [feature_names.index(f) for f in ordered_names]
    return X[:, col_idx], ordered_names


def pool_group_features(
    group_config: dict,
    paths: dict,
    feature_spaces: dict,
) -> tuple[np.ndarray, list[str]]:
    """Pool active-TR features across all subjects."""
    X_all = []
    ordered_names = None

    for subj_cfg in group_config['subjects']:
        subject = subj_cfg['subject']
        X_subj, feat_names = load_subject_active_features(
            subject, subj_cfg['train_sessions'], paths
        )
        X_subj, names = select_and_order_features(X_subj, feat_names, feature_spaces)

        if ordered_names is None:
            ordered_names = names
        elif names != ordered_names:
            raise ValueError(
                f"sub-{subject:02d} surviving features differ from first subject"
            )
        X_all.append(X_subj)

    X_pool = np.vstack(X_all)
    logger.info(f"Pooled matrix: {X_pool.shape[0]} TRs x {X_pool.shape[1]} features")
    return X_pool, ordered_names


def drop_zero_variance_features(
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Drop features with zero variance in the pooled active-TR matrix."""
    valid = X.var(axis=0) > 1e-10
    dropped = [n for n, v in zip(feature_names, valid) if not v]
    if dropped:
        logger.warning(f"Dropping {len(dropped)} zero-variance feature(s) on active TRs: {dropped}")
    return X[:, valid], [n for n, v in zip(feature_names, valid) if v]


def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix and report summary statistics."""
    corr = np.corrcoef(X.T)
    off_diag = corr[np.triu_indices(len(corr), k=1)]
    logger.info(f"Off-diagonal r: [{off_diag.min():.3f}, {off_diag.max():.3f}], "
                f"mean |r|={np.abs(off_diag).mean():.3f}")
    return corr


def plot_correlation_matrix(
    corr: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    mask: np.ndarray | None = None,
    file_stem: str = 'feature_correlation_matrix',
) -> None:
    """Save correlation heatmap (SVG + PNG). Pass mask to show a triangle only."""
    fig, ax = plt.subplots(figsize=(11, 9))

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        vmin=-1, vmax=1,
        center=0,
        cmap='RdBu_r',
        square=True,
        xticklabels=feature_names,
        yticklabels=feature_names,
        linewidths=0,
        cbar_kws={'shrink': 0.7, 'label': 'Pearson r'},
    )

    ax.tick_params(axis='x', labelsize=6.5, rotation=90)
    ax.tick_params(axis='y', labelsize=6.5, rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()

    for fmt in ('svg', 'png'):
        save_figure(fig, output_dir, f'{file_stem}.{fmt}', dpi=200)
        logger.info(f"Saved: {output_dir / f'{file_stem}.{fmt}'}")


def main():
    parser = argparse.ArgumentParser(
        description='Feature correlation matrix for GROUP_N5_PRIMARY'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=PATHS['figures'] / 'feature_corr',
        help='Directory for output figures (default: results/figures/feature_corr/)'
    )
    args = parser.parse_args()

    logger.info("Loading and pooling behavioral features...")
    X, feature_names = pool_group_features(GROUP_N5_PRIMARY, PATHS, FEATURE_SPACES)

    X, feature_names = drop_zero_variance_features(X, feature_names)

    logger.info("Computing correlation matrix...")
    corr = compute_correlation_matrix(X)

    logger.info("Plotting...")
    plot_correlation_matrix(corr, feature_names, args.output_dir)

    lower_tri_mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    plot_correlation_matrix(
        corr, feature_names, args.output_dir,
        mask=lower_tri_mask,
        file_stem='feature_correlation_matrix_lower_triangle',
    )


if __name__ == '__main__':
    main()
