#!/usr/bin/env python3
"""
Plot behavioral feature activity as a raster for one exemplar run per subject.

For each subject in the group, selects the middle training session and first
practice run, and renders a (features x TRs) heatmap of raw feature activity.
Motor features are continuous [0,1] (press proportions); scene features are
binary. Features are ordered motor-first then scene in config semantic order;
Valley is excluded (zero variance across all subjects' training sessions).

Saves one SVG and one PNG per subject to results/figures/feature_distributions/.

Usage:
    python scripts/python/plot_feature_distributions.py [--group GROUP_N5_PRIMARY]
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY
from mario_encoding.utils import sns_styleset, save_figure
from mario_encoding.utils.data_loading import (
    load_practice_metadata,
    load_practice_features_with_names,
)

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}

TR        = PARAMETERS['TR']
FONT_SIZE = 8
TICK_SIZE = 7

# Excluded globally: zero variance across all subjects' training sessions
EXCLUDED_FEATURES = {'Valley'}


# ============================================================================
# Feature ordering
# ============================================================================

def build_feature_order(feature_spaces: dict) -> list[str]:
    """Motor-first, then scene in config order; excluded features dropped."""
    return [
        f
        for space in ('motor', 'scene')
        for f in feature_spaces[space]
        if f not in EXCLUDED_FEATURES
    ]


# ============================================================================
# Data loading
# ============================================================================

def pick_exemplar_session(train_sessions: list[int], session_index: int) -> int:
    return train_sessions[session_index]


def load_exemplar_features(
    subject: int,
    session: int,
    run_index: int,
    paths: dict,
    ordered_features: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Load one run's features, select and order per ordered_features.

    Returns:
        X     -- (n_trs, n_features) float array
        names -- feature names in display order
        t     -- time axis in seconds
    """
    metadata = load_practice_metadata(subject, session, paths['practice_phase_metadata'])
    run = metadata['practice_runs'][run_index]
    X_sess, feat_names = load_practice_features_with_names(
        subject, session, [run], paths['per_run_downsampled_to_TR']
    )
    available = set(feat_names)
    names = [f for f in ordered_features if f in available]
    col_idx = [feat_names.index(f) for f in names]
    X = X_sess[:, col_idx].astype(float)
    t = np.arange(len(X)) * TR
    return X, names, t


# ============================================================================
# Plotting
# ============================================================================

def plot_feature_raster(
    X: np.ndarray,
    feature_names: list[str],
    t: np.ndarray,
    subject: int,
    session: int,
    output_dir: Path,
) -> None:
    """Save feature activity raster (SVG + PNG) for one subject's exemplar run."""
    n_features = len(feature_names)
    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(
        X.T,
        aspect='auto',
        interpolation='nearest',
        cmap='Blues',
        vmin=0, vmax=1,
        extent=[t[0], t[-1], n_features - 0.5, -0.5],
    )

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(feature_names, fontsize=TICK_SIZE)
    ax.set_xlabel('Time (s)', fontsize=FONT_SIZE)
    ax.tick_params(axis='x', labelsize=TICK_SIZE)

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01)
    cbar.set_label('Feature activity', fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    plt.tight_layout()

    stem = f'sub-{subject:02d}_ses-{session:03d}'
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
    parser.add_argument('--group', default='GROUP_N5_PRIMARY',
                        choices=list(GROUP_REGISTRY.keys()))
    parser.add_argument('--session-index', type=int, default=None,
                        help='Index into train_sessions (default: middle)')
    parser.add_argument('--run-index', type=int, default=0,
                        help='Index into practice_runs for chosen session (default: 0)')
    parser.add_argument('--output-dir', type=Path,
                        default=PATHS['figures'] / 'feature_distributions')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    sns_styleset()

    group_cfg = GROUP_REGISTRY[args.group]
    ordered_features = build_feature_order(FEATURE_SPACES)

    for subj_cfg in group_cfg['subjects']:
        subject = subj_cfg['subject']
        train_sessions = subj_cfg['train_sessions']

        session_idx = (args.session_index if args.session_index is not None
                       else len(train_sessions) // 2)
        session = pick_exemplar_session(train_sessions, session_idx)

        logger.info(f"sub-{subject:02d}: ses-{session:03d} run-index {args.run_index}")
        X, feature_names, t = load_exemplar_features(
            subject, session, args.run_index, PATHS, ordered_features
        )
        logger.info(f"  {len(t)} TRs x {len(feature_names)} features")

        plot_feature_raster(X, feature_names, t, subject, session, args.output_dir)
        logger.info(f"  Saved sub-{subject:02d}_ses-{session:03d}.{{svg,png}}")

    logger.info("Done.")


if __name__ == '__main__':
    main()
