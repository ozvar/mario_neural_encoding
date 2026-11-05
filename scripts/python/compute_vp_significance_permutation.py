"""
Block-permutation significance test for variance partitioning predictions.

Implements LeBel et al. 2023 (Scientific Data) procedure:
- For each voxel, permute the held-out Y_test by shuffling blocks of `block_size`
  TRs within each test run (preserving temporal autocorrelation within blocks).
- Recompute observed statistics (Pearson r for full model; product measure for
  each band) using the *existing* fitted weights (no refit).
- Per-voxel p = (1 + #(null >= observed)) / (1 + n_perm).
- BH-FDR across voxels, separately per statistic.

References:
    LeBel et al. 2023, Scientific Data, Sec. "Statistical significance testing".
    Adolf et al. 2014, Front Neuroinform, for the block-permutation foundation.

Usage:
    # single subject
    python scripts/python/compute_vp_significance_permutation.py \\
        --subject 5 --train-sessions 9 10 11 12 13 14 15 16 17 \\
        --test-sessions 18 --experiment-id 20260427_150334

    # group loop (per-subject sequential; writes one significance.npz each)
    python scripts/python/compute_vp_significance_permutation.py \\
        --group GROUP_N5_PRIMARY
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY
from mario_encoding.utils.data_loading import (
    concatenate_sessions_with_names,
    select_and_validate_features,
    filter_zero_variance_features,
    filter_zero_variance_voxels,
    preprocess_data,
    compute_baseline_mask,
    apply_delays_per_run,
    drop_baseline_samples,
)
from mario_encoding.variance_partitioning import (
    map_features_to_spaces,
    expand_feature_space_indices_for_delays,
    create_feature_space_arrays,
)

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}


def setup_logging(subject: int, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'sub-{subject:02d}_significance_permutation.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def get_vp_dir(subject: int, train_sessions: list, test_sessions: list,
               experiment_id: str) -> Path:
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    return PATHS['variance_partitioning'] / dataset_dir / experiment_id


def load_fit_artifacts(vp_dir: Path, logger: logging.Logger) -> dict:
    """Load weights, training X mean, masks, and metadata from a completed fit."""
    weights_file = vp_dir / 'weights_full.npz'
    mean_file = vp_dir / 'train_x_mean.npy'
    voxel_mask_file = vp_dir / 'valid_voxels_mask.npy'
    feature_mask_file = vp_dir / 'valid_features_mask.npy'
    metadata_file = vp_dir / 'metadata.json'

    for p in (weights_file, mean_file, voxel_mask_file, feature_mask_file, metadata_file):
        if not p.exists():
            raise FileNotFoundError(f"Missing fit artifact: {p}")

    weights = np.load(weights_file)['weights_full']
    train_x_mean = np.load(mean_file)
    valid_voxels_mask = np.load(voxel_mask_file)
    valid_features_mask = np.load(feature_mask_file)
    with open(metadata_file) as f:
        metadata = json.load(f)

    logger.info(f"Loaded weights: {weights.shape}")
    logger.info(f"Loaded train_x_mean: {train_x_mean.shape}")
    logger.info(f"Valid voxels: {int(valid_voxels_mask.sum())}/{len(valid_voxels_mask)}")
    logger.info(f"Valid features: {int(valid_features_mask.sum())}/{len(valid_features_mask)}")
    return {
        'weights': weights,
        'train_x_mean': train_x_mean,
        'valid_voxels_mask': valid_voxels_mask,
        'valid_features_mask': valid_features_mask,
        'metadata': metadata,
    }


def prepare_test_data(subject: int, test_sessions: list, train_x_mean: np.ndarray,
                      valid_features_mask: np.ndarray, valid_voxels_mask: np.ndarray,
                      logger: logging.Logger) -> dict:
    """
    Recreate the same X_test_delayed / Y_test as the fit script produced.

    Reuses the exact pipeline from fit_variance_partitioning.py: load -> select
    features (config) -> apply saved valid_features_mask -> apply saved
    valid_voxels_mask -> preprocess Y -> apply saved train mean (NOT a fresh fit)
    -> per-run delay -> drop baseline -> split per-space.
    """
    pipeline = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']

    logger.info("Reloading test data...")
    X_test, Y_test, run_onsets, session_onsets, level_onsets, feature_names = \
        concatenate_sessions_with_names(
            subject, test_sessions, PATHS['practice_phase_metadata'],
            PATHS['per_run_downsampled_to_TR'], fmri_path, pipeline, logger,
        )

    X_test, feature_names_sel, sel_mask = select_and_validate_features(
        X_test, feature_names, FEATURE_SPACES, logger,
    )

    # Apply saved fit-time masks.
    if X_test.shape[1] != len(valid_features_mask):
        raise ValueError(
            f"Selected feature count {X_test.shape[1]} != saved mask length "
            f"{len(valid_features_mask)}; FEATURE_SPACES config may have changed since fit."
        )
    X_test = X_test[:, valid_features_mask]

    if Y_test.shape[1] != len(valid_voxels_mask):
        raise ValueError(
            f"Loaded Y voxel count {Y_test.shape[1]} != saved mask length "
            f"{len(valid_voxels_mask)}; preprocessing pipeline may have changed."
        )
    Y_test = Y_test[:, valid_voxels_mask]

    active_mask = compute_baseline_mask(X_test, run_onsets, logger=logger)

    X_test, Y_test = preprocess_data(X_test, Y_test, run_onsets, level_onsets, logger)

    # Center X with the fit-time training mean, not a fresh statistic.
    if X_test.shape[1] != train_x_mean.shape[0]:
        raise ValueError(
            f"X_test feature count {X_test.shape[1]} != train_x_mean length "
            f"{train_x_mean.shape[0]}"
        )
    X_test = X_test - train_x_mean[None, :]

    delays = PARAMETERS['variance_partitioning']['delays']
    X_test_delayed = apply_delays_per_run(X_test, run_onsets, delays)

    X_test_delayed, Y_test, run_onsets, session_onsets = drop_baseline_samples(
        X_test_delayed, Y_test, run_onsets, session_onsets, active_mask, logger,
    )

    logger.info(f"Test data after pipeline: X={X_test_delayed.shape}, Y={Y_test.shape}, "
                f"n_runs={len(run_onsets)}")
    return {
        'X_test_delayed': X_test_delayed,
        'Y_test': Y_test,
        'run_onsets': run_onsets,
        'feature_names_kept': [n for n, k in zip(feature_names_sel, valid_features_mask) if k],
    }


def build_per_band_predictions(X_test_delayed: np.ndarray, weights: np.ndarray,
                               feature_names_kept: list, logger: logging.Logger
                               ) -> tuple:
    """
    Split delayed X by feature space, compute Y_pred_full and per-band Y_pred_j.

    Uses the A1-fixed sequential slicing of weights (space-major, delay-major
    within space) matching how GroupRidgeCV stores coef_.
    """
    delays = PARAMETERS['variance_partitioning']['delays']
    n_delays = len(delays)

    # Reconstruct feature -> space mapping from kept names.
    # FEATURE_SPACES values are lists of feature names; intersect with kept.
    feature_spaces_filtered = {
        space: [f for f in feats if f in feature_names_kept]
        for space, feats in FEATURE_SPACES.items()
    }
    space_to_indices = {
        space: np.array([feature_names_kept.index(f) for f in feats], dtype=int)
        for space, feats in feature_spaces_filtered.items()
    }
    n_features_original = len(feature_names_kept)
    space_to_indices_delayed = expand_feature_space_indices_for_delays(
        space_to_indices, n_features_original, delays, logger,
    )
    X_test_list, space_names_ordered = create_feature_space_arrays(
        X_test_delayed, space_to_indices_delayed, feature_spaces_filtered, logger,
    )

    # Sequential slicing of weights (A1).
    n_voxels = weights.shape[1]
    Y_pred_per_band = {}
    Y_pred_full = np.zeros((X_test_delayed.shape[0], n_voxels), dtype=np.float32)
    start = 0
    for X_space, space_name in zip(X_test_list, space_names_ordered):
        n_rows = X_space.shape[1]
        W_space = weights[start:start + n_rows, :]
        start += n_rows
        Y_pred_band = X_space @ W_space
        Y_pred_per_band[space_name] = Y_pred_band.astype(np.float32, copy=False)
        Y_pred_full += Y_pred_band
    if start != weights.shape[0]:
        raise ValueError(
            f"Weight slicing consumed {start} of {weights.shape[0]} rows; "
            f"feature space ordering inconsistent with fit."
        )

    return Y_pred_full, Y_pred_per_band, space_names_ordered


def compute_observed_statistics(Y_test: np.ndarray, Y_pred_full: np.ndarray,
                                Y_pred_per_band: dict, logger: logging.Logger
                                ) -> dict:
    """Pearson r for full model; product measure per band. All per-voxel."""
    # Per-voxel Pearson correlation, vectorised.
    yt = Y_test - Y_test.mean(axis=0, keepdims=True)
    yp = Y_pred_full - Y_pred_full.mean(axis=0, keepdims=True)
    num = (yt * yp).sum(axis=0)
    denom = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0))
    denom = np.where(denom > 0, denom, 1.0)
    r_full = (num / denom).astype(np.float32)

    # Product measure per band: PM_j = sum(Y_pred_j * (2*Y - Y_pred_full)) / sum(Y^2).
    denom_pm = (Y_test ** 2).sum(axis=0)
    denom_pm = np.where(denom_pm > 0, denom_pm, 1.0)
    pm_per_band = {}
    for space, Y_pred_band in Y_pred_per_band.items():
        num_pm = (Y_pred_band * (2 * Y_test - Y_pred_full)).sum(axis=0)
        pm_per_band[space] = (num_pm / denom_pm).astype(np.float32)
        logger.info(f"  observed PM ({space}): mean={pm_per_band[space].mean():.4f}, "
                    f"max={pm_per_band[space].max():.4f}")
    logger.info(f"  observed r_full: mean={r_full.mean():.4f}, "
                f"max={r_full.max():.4f}")
    return {'r_full': r_full, 'pm_per_band': pm_per_band}


def build_within_run_blocks(run_onsets: np.ndarray, n_samples: int,
                            block_size: int) -> list:
    """
    For each run, return the list of (start, end) block intervals covering it.

    Trailing partial block (if run length not divisible by block_size) is kept
    as a smaller block rather than dropped.
    """
    blocks_per_run = []
    run_ends = list(run_onsets[1:]) + [n_samples]
    for run_start, run_end in zip(run_onsets, run_ends):
        intervals = []
        cursor = int(run_start)
        while cursor < run_end:
            stop = min(cursor + block_size, int(run_end))
            intervals.append((cursor, stop))
            cursor = stop
        blocks_per_run.append(intervals)
    return blocks_per_run


def permuted_Y_indices(blocks_per_run: list, rng: np.random.Generator) -> np.ndarray:
    """
    Construct a sample-index permutation that shuffles blocks within each run.
    Returns an int array of length n_samples; Y[perm_idx] is Y_perm.
    """
    perm_parts = []
    for intervals in blocks_per_run:
        order = rng.permutation(len(intervals))
        for k in order:
            s, e = intervals[k]
            perm_parts.append(np.arange(s, e, dtype=np.int64))
    return np.concatenate(perm_parts)


def run_permutation_test(Y_test: np.ndarray, Y_pred_full: np.ndarray,
                         Y_pred_per_band: dict, observed: dict,
                         run_onsets: np.ndarray, block_size: int,
                         n_perm: int, seed: int, logger: logging.Logger
                         ) -> dict:
    """
    Tail-count permutation test for r_full and per-band PM, using the same
    block-shuffle per iteration applied to all voxels.

    Memory plan: keep only running exceedance counts per voxel per statistic;
    never materialise (n_perm, n_voxels) null arrays.
    """
    n_samples, n_voxels = Y_test.shape
    blocks_per_run = build_within_run_blocks(run_onsets, n_samples, block_size)
    n_blocks = sum(len(b) for b in blocks_per_run)
    logger.info(f"Permutation: n_perm={n_perm}, block_size={block_size}, "
                f"n_blocks={n_blocks} across {len(blocks_per_run)} runs, "
                f"shortest run={min(len(b) for b in blocks_per_run)} blocks")

    rng = np.random.default_rng(seed)

    obs_r = observed['r_full']
    obs_pm = observed['pm_per_band']

    # Sufficient stats (centred Y_pred_full, Y_pred_band) computed once; for r,
    # the denominator yp_norm is permutation-invariant.
    yp_full = Y_pred_full - Y_pred_full.mean(axis=0, keepdims=True)
    yp_full_norm = np.sqrt((yp_full ** 2).sum(axis=0))
    yp_full_norm = np.where(yp_full_norm > 0, yp_full_norm, 1.0)

    exceed_r = np.zeros(n_voxels, dtype=np.int64)
    exceed_pm = {space: np.zeros(n_voxels, dtype=np.int64) for space in obs_pm}

    log_every = max(1, n_perm // 20)

    for i in range(n_perm):
        idx = permuted_Y_indices(blocks_per_run, rng)
        Y_perm = Y_test[idx]

        # r_full_perm per voxel.
        yt = Y_perm - Y_perm.mean(axis=0, keepdims=True)
        yt_norm = np.sqrt((yt ** 2).sum(axis=0))
        yt_norm = np.where(yt_norm > 0, yt_norm, 1.0)
        r_perm = (yt * yp_full).sum(axis=0) / (yt_norm * yp_full_norm)
        exceed_r += (r_perm >= obs_r).astype(np.int64)

        # PM_perm per band per voxel. Denominator sum(Y_perm^2) == sum(Y^2)
        # because permutation only reorders; we still compute fresh for safety.
        denom_pm = (Y_perm ** 2).sum(axis=0)
        denom_pm = np.where(denom_pm > 0, denom_pm, 1.0)
        residual = 2 * Y_perm - Y_pred_full
        for space, Y_pred_band in Y_pred_per_band.items():
            pm_perm = (Y_pred_band * residual).sum(axis=0) / denom_pm
            exceed_pm[space] += (pm_perm >= obs_pm[space]).astype(np.int64)

        if (i + 1) % log_every == 0:
            logger.info(f"  permutation {i + 1}/{n_perm}")

    p_r = (1.0 + exceed_r) / (1.0 + n_perm)
    p_pm = {space: (1.0 + cnt) / (1.0 + n_perm) for space, cnt in exceed_pm.items()}
    return {'p_r_full': p_r, 'p_pm_per_band': p_pm}


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> tuple:
    """Benjamini-Hochberg FDR. Returns (q_values, significant_mask)."""
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    threshold_line = np.arange(1, n + 1) / n * alpha
    below = sorted_p <= threshold_line
    if below.any():
        max_idx = int(np.where(below)[0][-1])
        threshold = sorted_p[max_idx]
    else:
        threshold = 0.0

    q_sorted = np.zeros(n)
    q_sorted[-1] = min(sorted_p[-1], 1.0)
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(sorted_p[i] * n / (i + 1), q_sorted[i + 1], 1.0)
    q_values = np.zeros(n)
    q_values[order] = q_sorted
    return q_values, p_values <= threshold


def save_significance(vp_dir: Path, observed: dict, p_dict: dict,
                      n_test_active: int,
                      params: dict, logger: logging.Logger) -> None:
    out_file = vp_dir / 'significance.npz'
    save_dict = {
        'r_full': observed['r_full'],
        'p_r_full': p_dict['p_r_full'],
        'n_test_active': np.int32(n_test_active),
    }
    q_r, sig_r = fdr_correction(p_dict['p_r_full'], alpha=params['fdr_alpha'])
    save_dict['p_fdr_r_full'] = q_r
    save_dict[f'sig_r_full_fdr_q{int(params["fdr_alpha"] * 100):02d}'] = sig_r

    for space, pm in observed['pm_per_band'].items():
        save_dict[f'pm_{space}'] = pm
        save_dict[f'p_pm_{space}'] = p_dict['p_pm_per_band'][space]
        q_pm, sig_pm = fdr_correction(p_dict['p_pm_per_band'][space],
                                      alpha=params['fdr_alpha'])
        save_dict[f'p_fdr_pm_{space}'] = q_pm
        save_dict[f'sig_pm_{space}_fdr_q{int(params["fdr_alpha"] * 100):02d}'] = sig_pm

    save_dict['n_perm'] = params['n_perm']
    save_dict['block_size'] = params['block_size']
    save_dict['seed'] = params['seed']
    save_dict['fdr_alpha'] = params['fdr_alpha']

    np.savez_compressed(out_file, **save_dict)
    logger.info(f"Saved significance to {out_file}")

    n_voxels = len(observed['r_full'])
    n_sig_r = int(sig_r.sum())
    logger.info(f"  r_full: {n_sig_r}/{n_voxels} voxels significant at FDR q<{params['fdr_alpha']} "
                f"({n_sig_r / n_voxels * 100:.1f}%)")
    for space in observed['pm_per_band']:
        sig = save_dict[f'sig_pm_{space}_fdr_q{int(params["fdr_alpha"] * 100):02d}']
        logger.info(f"  pm_{space}: {int(sig.sum())}/{n_voxels} voxels significant "
                    f"({sig.sum() / n_voxels * 100:.1f}%)")


def run_one_subject(subject: int, train_sessions: list, test_sessions: list,
                    experiment_id: str, perm_params: dict) -> None:
    vp_dir = get_vp_dir(subject, train_sessions, test_sessions, experiment_id)
    if not vp_dir.exists():
        raise FileNotFoundError(f"VP fit directory not found: {vp_dir}")

    logger = setup_logging(subject, vp_dir)
    logger.info("=" * 80)
    logger.info(f"PERMUTATION SIGNIFICANCE | sub-{subject:02d}")
    logger.info(f"VP dir: {vp_dir}")
    logger.info(f"Train sessions: {train_sessions}")
    logger.info(f"Test sessions: {test_sessions}")
    logger.info(f"Parameters: {perm_params}")
    logger.info("=" * 80)

    fit_artifacts = load_fit_artifacts(vp_dir, logger)
    test = prepare_test_data(
        subject, test_sessions, fit_artifacts['train_x_mean'],
        fit_artifacts['valid_features_mask'], fit_artifacts['valid_voxels_mask'],
        logger,
    )
    Y_pred_full, Y_pred_per_band, _ = build_per_band_predictions(
        test['X_test_delayed'], fit_artifacts['weights'],
        test['feature_names_kept'], logger,
    )
    observed = compute_observed_statistics(
        test['Y_test'], Y_pred_full, Y_pred_per_band, logger,
    )
    p_dict = run_permutation_test(
        test['Y_test'], Y_pred_full, Y_pred_per_band, observed,
        test['run_onsets'], perm_params['block_size'], perm_params['n_perm'],
        perm_params['seed'], logger,
    )
    save_significance(vp_dir, observed, p_dict,
                      n_test_active=int(test['Y_test'].shape[0]),
                      params=perm_params, logger=logger)
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--group', type=str, choices=list(GROUP_REGISTRY.keys()))
    src.add_argument('--subject', type=int)

    parser.add_argument('--train-sessions', type=int, nargs='+')
    parser.add_argument('--test-sessions', type=int, nargs='+')
    parser.add_argument('--experiment-id', type=str)
    parser.add_argument('--seed', type=int, default=None,
                        help='Override config seed for reproducibility')
    parser.add_argument('--n-perm', type=int, default=None)
    args = parser.parse_args()

    perm_cfg = PARAMETERS['permutation']
    perm_params = {
        'n_perm': args.n_perm if args.n_perm is not None else perm_cfg['n_perm'],
        'block_size': perm_cfg['block_size'],
        'seed': args.seed if args.seed is not None else perm_cfg['seed'],
        'fdr_alpha': perm_cfg['fdr_alpha'],
    }

    if args.group is not None:
        group_cfg = GROUP_REGISTRY[args.group]
        for sub_cfg in group_cfg['subjects']:
            try:
                run_one_subject(
                    sub_cfg['subject'], sub_cfg['train_sessions'],
                    sub_cfg['test_sessions'], sub_cfg['experiment_id'],
                    perm_params,
                )
            except FileNotFoundError as e:
                print(f"[sub-{sub_cfg['subject']:02d}] skipped: {e}")
    else:
        if not (args.train_sessions and args.test_sessions and args.experiment_id):
            parser.error("--subject requires --train-sessions, --test-sessions, "
                         "and --experiment-id")
        run_one_subject(args.subject, args.train_sessions, args.test_sessions,
                        args.experiment_id, perm_params)


if __name__ == '__main__':
    main()
