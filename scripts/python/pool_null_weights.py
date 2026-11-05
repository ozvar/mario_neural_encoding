"""
Pool per-subject phase-randomized VP weights into group-level null weight files
for PCA significance testing.

For each null iteration index k, loads weights_full.npz from the k-th completed
null fit for each subject (sorted by experiment_id timestamp), applies the same
per-voxel rescaling as prepare_pca_weights.py (Dupre la Tour 2025), averages
across FIR delays, optionally applies the same per-subject voxel selection that
fit_pca.py used on real data (R2 filter + subsample), and concatenates across
subjects into a single (n_features, n_voxels_total) array.

Null iterations are matched across subjects by sorted order. If subjects have
unequal numbers of completed null fits, the minimum count is used. Use
--max-iterations to cap the number of pooled files produced.

IMPORTANT: pass --pca-results-dir to apply the same voxel selection used by the
real PCA fit. Without it, null pooling uses all voxels from each null fit and
the null distribution will NOT be comparable to the real PCA (which is voxel-
filtered + subsampled).

Output format matches what fit_pca.py --null-weight-files expects:
    weights_joint: (n_features, n_voxels_total) float32

Usage:
    # Recommended: match voxel selection to a real PCA run
    python pool_null_weights.py --group GROUP_N5_PRIMARY \\
        --pca-results-dir results/pca/GROUP_N5_PRIMARY
    python pool_null_weights.py --group GROUP_N5_PRIMARY --max-iterations 200
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from mario_encoding.config import PATHS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}


def setup_logging(output_dir: Path, label: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'{label}_pool_null_weights.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)


def get_null_dirs(subject_cfg: dict) -> list[Path]:
    """
    Return all completed null iteration directories for one subject, sorted by
    experiment_id (lexicographic order = chronological for timestamp IDs).

    A directory is considered complete if weights_full.npz is present.
    """
    subject = subject_cfg['subject']
    train_sessions = subject_cfg['train_sessions']
    test_sessions = subject_cfg['test_sessions']

    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    base_dir = (PATHS['variance_partitioning'] /
                f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}_phase_randomized')

    if not base_dir.exists():
        return []

    return sorted([
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / 'weights_full.npz').exists() and (d / 'R2_scores.npz').exists()
    ])


def load_and_rescale(null_dir: Path, voxel_selection: np.ndarray | None) -> np.ndarray:
    """
    Load one null VP output directory, apply Dupre la Tour (2025) per-voxel rescaling,
    average across FIR delays, optionally subset to a saved voxel selection, and
    return a joint (n_features, n_voxels_kept) float32 array.

    Rescaling is identical to prepare_pca_weights.py:
        factor = sqrt(max(0, R2_full)) / norm(weights_full, per voxel)
    The norm is computed on the full delay-expanded weight matrix so the scale
    reflects total weight magnitude across all features and delays.
    """
    weights_data = np.load(null_dir / 'weights_full.npz')
    r2_data = np.load(null_dir / 'R2_scores.npz')

    weights_full = weights_data['weights_full']
    motor_per_delay = weights_data['motor_weights_per_delay']
    scene_per_delay = weights_data['scene_weights_per_delay']
    R2_full = r2_data['R2_full']

    norms = np.linalg.norm(weights_full, axis=0)
    scale = np.sqrt(np.maximum(0, R2_full))
    rescale_factors = scale / (norms + 1e-10)

    motor_avg = (motor_per_delay * rescale_factors[None, None, :]).mean(axis=0)
    scene_avg = (scene_per_delay * rescale_factors[None, None, :]).mean(axis=0)

    joint = np.concatenate([motor_avg, scene_avg], axis=0).astype(np.float32)
    if voxel_selection is not None:
        if voxel_selection.max() >= joint.shape[1]:
            raise ValueError(
                f"voxel_selection max index {voxel_selection.max()} >= n_voxels "
                f"{joint.shape[1]} in {null_dir}. Selection probably came from "
                f"a different VP fit."
            )
        joint = joint[:, voxel_selection]
    return joint


def pool_iteration(subject_cfgs: list[dict],
                   null_dirs_per_subject: list[list[Path]],
                   voxel_selections: list[np.ndarray | None],
                   iter_idx: int,
                   logger: logging.Logger) -> np.ndarray:
    """
    Rescale and concatenate weights across all subjects for one null iteration index.

    Returns (n_features, n_voxels_total) float32.
    """
    subject_weights = []
    for sub_idx, subject_cfg in enumerate(subject_cfgs):
        null_dir = null_dirs_per_subject[sub_idx][iter_idx]
        weights = load_and_rescale(null_dir, voxel_selections[sub_idx])
        logger.info(f"    sub-{subject_cfg['subject']:02d}: {weights.shape}  [{null_dir.name}]")
        subject_weights.append(weights)

    pooled = np.concatenate(subject_weights, axis=1)
    return pooled


def load_voxel_selections(
    pca_results_dir: Path | None,
    subject_cfgs: list[dict],
    logger: logging.Logger,
) -> list[np.ndarray | None]:
    """
    Load per-subject voxel_selection_*.npy from a fit_pca.py output directory.

    Returns a list of int-index arrays, or a list of Nones if pca_results_dir is None.
    """
    if pca_results_dir is None:
        logger.warning(
            "No --pca-results-dir provided: pooling will use ALL voxels per subject. "
            "Null eigenvalues will NOT be comparable to a voxel-filtered real PCA fit."
        )
        return [None] * len(subject_cfgs)

    pca_results_dir = Path(pca_results_dir)
    if not pca_results_dir.exists():
        raise FileNotFoundError(f"PCA results dir not found: {pca_results_dir}")

    selections = []
    for sub_cfg in subject_cfgs:
        sid = sub_cfg['subject']
        sel_path = pca_results_dir / f'voxel_selection_sub-{sid:02d}.npy'
        if not sel_path.exists():
            raise FileNotFoundError(
                f"Voxel selection not found: {sel_path}. Run fit_pca.py first "
                f"so it writes voxel_selection_sub-XX.npy for each subject."
            )
        sel = np.load(sel_path)
        logger.info(f"  sub-{sid:02d}: loaded {len(sel)} voxel indices from {sel_path.name}")
        selections.append(sel)
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pool phase-randomized VP weights into group null files for PCA'
    )
    parser.add_argument('--group', required=True, choices=list(GROUP_REGISTRY.keys()))
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Cap number of pooled iterations (default: use all available)')
    parser.add_argument('--pca-results-dir', type=str, default=None,
                        help='Directory containing voxel_selection_sub-XX.npy from a '
                             'real fit_pca.py run. Strongly recommended: pools using the '
                             'same voxel filter+subsample so null eigenvalues are '
                             'comparable to the real PCA fit.')
    args = parser.parse_args()

    group_cfg = GROUP_REGISTRY[args.group]
    label = group_cfg['output_label']
    output_dir = PATHS['pca'] / 'null' / label
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, label)

    logger.info('=' * 80)
    logger.info('POOL NULL WEIGHTS')
    logger.info('=' * 80)
    logger.info(f'Group: {args.group}')
    logger.info(f'Subjects: {[s["subject"] for s in group_cfg["subjects"]]}')
    logger.info(f'Output: {output_dir}')

    # ------------------------------------------------------------------
    # Collect null directories per subject
    # ------------------------------------------------------------------
    logger.info('')
    logger.info('Scanning null output directories...')
    subject_cfgs = group_cfg['subjects']
    null_dirs_per_subject = []

    for sub_cfg in subject_cfgs:
        dirs = get_null_dirs(sub_cfg)
        logger.info(f'  sub-{sub_cfg["subject"]:02d}: {len(dirs)} completed null fits')
        null_dirs_per_subject.append(dirs)

    counts = [len(d) for d in null_dirs_per_subject]
    if any(c == 0 for c in counts):
        missing = [sub_cfg['subject'] for sub_cfg, c in zip(subject_cfgs, counts) if c == 0]
        raise RuntimeError(
            f'No completed null fits found for sub-{missing}. '
            f'Run submit_vp_jobs.py --null --submit first.'
        )

    n_available = min(counts)
    n_iterations = min(n_available, args.max_iterations) if args.max_iterations else n_available

    if min(counts) != max(counts):
        logger.warning(
            f'Unequal null counts across subjects {counts}. '
            f'Using minimum: {n_available}.'
        )
    logger.info(f'Pooling {n_iterations} iterations (available: {n_available})')

    # ------------------------------------------------------------------
    # Load voxel selections (must match real PCA fit)
    # ------------------------------------------------------------------
    logger.info('')
    logger.info('Loading per-subject voxel selections...')
    pca_results_dir = Path(args.pca_results_dir) if args.pca_results_dir else None
    voxel_selections = load_voxel_selections(pca_results_dir, subject_cfgs, logger)

    # ------------------------------------------------------------------
    # Pool and save
    # ------------------------------------------------------------------
    logger.info('')
    for k in range(n_iterations):
        logger.info(f'Iteration {k+1}/{n_iterations}')
        pooled = pool_iteration(
            subject_cfgs, null_dirs_per_subject, voxel_selections, k, logger,
        )

        out_file = output_dir / f'pooled_{k:04d}.npz'
        np.savez_compressed(out_file, weights_joint=pooled)
        logger.info(f'  Saved: {out_file.name}  shape={pooled.shape}')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info('')
    logger.info('=' * 80)
    logger.info('COMPLETE')
    logger.info('=' * 80)
    logger.info(f'Pooled {n_iterations} null iterations to: {output_dir}')
    logger.info(
        f'Run fit_pca.py with: '
        f'--null-weight-files {output_dir}/pooled_*.npz'
    )


if __name__ == '__main__':
    main()
