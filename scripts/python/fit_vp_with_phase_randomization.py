"""
Fit variance partitioning models with optional phase randomization testing.

This script extends fit_variance_partitioning.py to support testing whether
shared variance is inflated by temporal autocorrelation. When --phase-randomize
is specified, motor features are phase-randomized (preserving autocorrelation
but breaking temporal relationships) before model fitting.

Phase randomization test:
1. Preserves autocorrelation structure of features
2. Breaks specific temporal relationships between motor and scene
3. Maintains run boundaries (randomizes within each run independently)
4. Applied BEFORE preprocessing/delays for fair comparison

Run 100+ times with --phase-randomize to build null distribution, then compare
to observed shared variance from standard run (without --phase-randomize).
"""
import argparse
import json
import logging
import os
import pickle
import gc
from pathlib import Path

import numpy as np
from scipy.fft import fft, ifft
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from voxelwise_tutorials.delayer import Delayer
from himalaya.backend import set_backend

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.utils.experiment_utils import create_experiment_config, save_experiment_config
from mario_encoding.utils.data_loading import (
    concatenate_sessions_with_names,
    filter_baseline_periods,
    select_and_validate_features,
    filter_zero_variance_features,
    filter_zero_variance_voxels,
    preprocess_data
)
from mario_encoding.variance_partitioning import (
    map_features_to_spaces,
    validate_all_spaces_nonempty,
    expand_feature_space_indices_for_delays,
    create_feature_space_arrays,
    fit_full_model,
    fit_restricted_model,
    compute_unique_variance,
    compute_shared_variance,
    compute_segregation_index,
    validate_variance_partition,
    extract_model_weights,
    decompose_weights_by_feature_space,
    save_model_weights,
)


def phase_randomize_preserving_runs(X, run_onsets, logger, mode='independent', rng=None):
    """
    Phase randomize features within each run, with two options for cross-feature
    structure.

    mode='independent' (legacy default):
        Each feature gets its own random phase vector per run. Preserves each
        feature's autocorrelation but destroys cross-feature correlations.
        This is the appropriate null for "is feature X related to BOLD at all?"
        questions but NOT for testing structure in the cross-feature covariance
        of encoding-model weights (e.g. PCA significance), because real data
        carries feature-feature correlations that this null erases.

    mode='joint':
        All features share the SAME random phase vector within a run, so the
        cross-spectral matrix of the design is preserved. This is the
        appropriate null for testing PCA structure in fitted weights: under H0
        the design has the same marginal spectra and same cross-feature
        correlations as the real data, but no temporal alignment with BOLD.

    Preserves (both modes):
        - Each feature's marginal power spectrum / autocorrelation
        - Run boundaries (no cross-run contamination)

    Parameters
    ----------
    X : array (n_samples, n_features)
    run_onsets : array
        Starting indices of each run.
    logger : logging.Logger
    mode : {'independent', 'joint'}
    rng : numpy Generator or None
        Source of randomness. If None, uses numpy's legacy global state
        (irreproducible).
    """
    if mode not in {'independent', 'joint'}:
        raise ValueError(f"mode must be 'independent' or 'joint', got {mode!r}")
    if rng is None:
        rng = np.random  # legacy global state; not reproducible
        rand_fn = rng.rand
    else:
        rand_fn = rng.random

    logger.info(f"Phase randomizing features within runs (mode={mode})...")
    X_shuffled = np.zeros_like(X)
    n_runs = len(run_onsets)
    n_features = X.shape[1]

    for run_idx in range(n_runs):
        start = run_onsets[run_idx]
        end = run_onsets[run_idx + 1] if run_idx < n_runs - 1 else len(X)
        n_samples_run = end - start

        logger.info(f"  Run {run_idx + 1}/{n_runs}: samples {start}-{end} (n={n_samples_run})")

        if mode == 'joint':
            # One phase vector shared across all features in this run
            random_phases = np.exp(2j * np.pi * rand_fn(n_samples_run))
            random_phases[0] = 1
            if n_samples_run % 2 == 0:
                random_phases[n_samples_run // 2] = 1
            for feat_idx in range(n_features):
                fft_x = fft(X[start:end, feat_idx])
                fft_shuffled = np.abs(fft_x) * random_phases
                X_shuffled[start:end, feat_idx] = np.real(ifft(fft_shuffled))
        else:
            for feat_idx in range(n_features):
                fft_x = fft(X[start:end, feat_idx])
                random_phases = np.exp(2j * np.pi * rand_fn(len(fft_x)))
                random_phases[0] = 1
                if len(fft_x) % 2 == 0:
                    random_phases[len(fft_x) // 2] = 1
                fft_shuffled = np.abs(fft_x) * random_phases
                X_shuffled[start:end, feat_idx] = np.real(ifft(fft_shuffled))

    logger.info("  Phase randomization complete")

    # Diagnostics
    logger.info("  Verifying autocorrelation preservation...")
    n_diag = min(5, n_features)
    orig_ac = np.mean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1] for i in range(n_diag)])
    shuf_ac = np.mean([np.corrcoef(X_shuffled[:-1, i], X_shuffled[1:, i])[0, 1] for i in range(n_diag)])
    logger.info(f"    Original lag-1 autocorr (mean of first {n_diag} features): {orig_ac:.4f}")
    logger.info(f"    Shuffled lag-1 autocorr (mean of first {n_diag} features): {shuf_ac:.4f}")

    if mode == 'joint' and n_features >= 2:
        # Cross-feature correlation diagnostic
        idx0, idx1 = 0, 1
        orig_xc = np.corrcoef(X[:, idx0], X[:, idx1])[0, 1]
        shuf_xc = np.corrcoef(X_shuffled[:, idx0], X_shuffled[:, idx1])[0, 1]
        logger.info(
            f"    Cross-feature corr (feats {idx0},{idx1}): "
            f"orig={orig_xc:.4f}, shuffled={shuf_xc:.4f} "
            f"(joint mode should approximately preserve this)"
        )

    return X_shuffled


def setup_logging(subject, session_range, log_dir, phase_randomize):
    """Configure logging to both file and console."""
    suffix = '_phase_randomized' if phase_randomize else ''
    log_file = log_dir / f'sub-{subject:02d}_ses-{session_range[0]:03d}-{session_range[1]:03d}_variance_partitioning{suffix}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Reset any existing handlers
    )

    return logging.getLogger(__name__)


def get_output_directory(subject, train_sessions, test_sessions, experiment_id, base_path, phase_randomize):
    """
    Create output directory for this subject/session/experiment combination.

    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    experiment_id : str
        Timestamp-based experiment ID
    base_path : Path
    phase_randomize : bool
        If True, adds '_phase_randomized' suffix to directory name

    Returns:
    --------
    output_dir : Path
        Directory to save all models and results
    """
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    suffix = '_phase_randomized' if phase_randomize else ''
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}{suffix}'
    output_dir = base_path / dataset_dir / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fit_and_save_models_sequentially(X_train_list, Y_train, X_test_list, Y_test,
                                    space_names_ordered, run_onsets_train,
                                    params, backend, output_dir,
                                    compute_product_measure, compute_unique_variance,
                                    space_to_indices_delayed, n_delays, logger):
    """
    Fit models and compute variance decomposition metrics.

    Always fits full model. Optionally computes:
    - Product measure (1 model, fast)
    - Unique variance (N+1 models, slower)

    Parameters:
    -----------
    X_train_list : list of arrays
        One array per feature space (training data)
    Y_train : array
    X_test_list : list of arrays
        One array per feature space (test data)
    Y_test : array
    space_names_ordered : list of str
        Feature space names in order
    run_onsets_train : array
    params : dict
        Variance partitioning parameters from config
    backend : str
    output_dir : Path
    compute_product_measure : bool
        Whether to compute product measure decomposition
    compute_unique_variance : bool
        Whether to fit restricted models and compute unique variance
    space_to_indices_delayed : dict
        Maps space names to indices in the delay-expanded feature matrix
    n_delays : int
        Number of FIR delays
    logger : logging.Logger

    Returns:
    --------
    results : dict with keys 'R2_full', optionally 'R2_restricted', 'R2_unique', 'product_measure'
    """
    results = {
        'R2_full': None,
        'R2_restricted': {},
        'R2_unique': {},
        'product_measure': {}
    }

    # ========================================================================
    # FIT FULL MODEL
    # ========================================================================
    logger.info("="*80)
    logger.info("FITTING FULL MODEL (all feature spaces)")
    logger.info("="*80)

    model_full = fit_full_model(
        X_train_list, Y_train, run_onsets_train, params, backend, logger
    )

    # Score on test set
    logger.info("Scoring full model on test set...")
    results['R2_full'] = model_full.score(X_test_list, Y_test)

    # Convert to numpy if needed
    if hasattr(results['R2_full'], 'cpu'):
        results['R2_full'] = results['R2_full'].cpu().numpy()

    logger.info(f"  Total test R2: {results['R2_full'].sum():.6f}")
    logger.info(f"  Mean test R2: {results['R2_full'].mean():.6f}")
    logger.info(f"  Max test R2: {results['R2_full'].max():.6f}")

    # ========================================================================
    # COMPUTE PRODUCT MEASURE (if requested)
    # ========================================================================
    if compute_product_measure:
        logger.info("")
        logger.info("="*80)
        logger.info("COMPUTING PRODUCT MEASURE")
        logger.info("="*80)
        from mario_encoding.variance_partitioning import compute_product_measure as compute_pm
        from mario_encoding.variance_partitioning import validate_product_measure

        results['product_measure'] = compute_pm(
            model_full, X_test_list, Y_test, space_names_ordered, logger
        )

        validate_product_measure(results['product_measure'], results['R2_full'], logger)

    # Extract and save weights before freeing the model
    logger.info("")
    logger.info("="*80)
    logger.info("EXTRACTING WEIGHTS FROM FULL MODEL")
    logger.info("="*80)
    weights_full = extract_model_weights(model_full, logger)
    weights_by_space = decompose_weights_by_feature_space(
        weights_full, space_to_indices_delayed, space_names_ordered, n_delays, logger
    )
    save_model_weights(
        weights_full, weights_by_space, space_names_ordered, output_dir, 'full', logger
    )

    # Delete to free memory
    del model_full
    gc.collect()
    logger.info("  Freed model memory [OK]")

    # ========================================================================
    # FIT RESTRICTED MODELS (if unique variance requested)
    # ========================================================================
    if compute_unique_variance:
        logger.info("")
        logger.info("="*80)
        logger.info("COMPUTING UNIQUE VARIANCE (fitting N restricted models)")
        logger.info("="*80)

        for space_name in space_names_ordered:
            logger.info("")
            logger.info("="*80)
            logger.info(f"FITTING RESTRICTED MODEL (excluding {space_name})")
            logger.info("="*80)

            # Get index to exclude
            excluded_idx = space_names_ordered.index(space_name)

            # Create restricted feature lists
            X_train_restricted = [X_train_list[i] for i in range(len(X_train_list)) if i != excluded_idx]
            X_test_restricted = [X_test_list[i] for i in range(len(X_test_list)) if i != excluded_idx]

            # Fit restricted model
            model_restricted = fit_restricted_model(
                X_train_list, Y_train, space_name, space_names_ordered,
                run_onsets_train, params, backend, logger
            )

            # Score on test set
            logger.info(f"Scoring restricted model (no {space_name}) on test set...")
            results['R2_restricted'][space_name] = model_restricted.score(X_test_restricted, Y_test)

            # Convert to numpy if needed
            if hasattr(results['R2_restricted'][space_name], 'cpu'):
                results['R2_restricted'][space_name] = results['R2_restricted'][space_name].cpu().numpy()

            logger.info(f"  Mean R2 (no {space_name}): {results['R2_restricted'][space_name].mean():.6f}")

            # Compute unique variance for this space
            results['R2_unique'][space_name] = compute_unique_variance(
                results['R2_full'], results['R2_restricted'][space_name], space_name, logger
            )

            # Save restricted model
            model_path = output_dir / f'model_no_{space_name}.pkl'
            logger.info(f"Saving restricted model to {model_path}...")
            with open(model_path, 'wb') as f:
                pickle.dump(model_restricted, f)
            logger.info("  Saved [OK]")

            # Delete to free memory
            del model_restricted
            gc.collect()
            logger.info("  Freed memory [OK]")

    return results


def save_results(subject, train_sessions, test_sessions, results,
                feature_spaces_filtered, space_to_indices, valid_voxels_mask,
                feature_names, valid_features_mask, output_dir, logger):
    """
    Save all results to disk.

    Saves:
    - R2 scores (full, restricted, unique, shared, segregation index)
    - Fisher z transforms
    - Product measures
    - Voxel mask
    - Metadata
    """
    logger.info("="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)

    from mario_encoding.variance_partitioning import fisher_z_transform

    # Compute Fisher z transforms
    logger.info("Computing Fisher z transforms...")
    fisher_z_full = fisher_z_transform(results['R2_full'])
    logger.info(f"  Fisher z (full): mean={fisher_z_full.mean():.6f}, max={fisher_z_full.max():.6f}")

    # Save R2 scores in compressed format
    r2_file = output_dir / 'R2_scores.npz'
    logger.info(f"Saving R2 scores to {r2_file}...")

    save_dict = {
        'R2_full': results['R2_full'],
        'fisher_z_R2_full': fisher_z_full,
    }

    # Add restricted and unique R2 for each space (if computed)
    if results['R2_unique']:
        logger.info("Adding unique variance and Fisher z transforms...")
        for space_name in results['R2_restricted'].keys():
            save_dict[f'R2_no_{space_name}'] = results['R2_restricted'][space_name]
            save_dict[f'R2_unique_{space_name}'] = results['R2_unique'][space_name]

            # Fisher z for restricted and unique
            fisher_z_restricted = fisher_z_transform(results['R2_restricted'][space_name])
            fisher_z_unique = fisher_z_transform(results['R2_unique'][space_name])
            save_dict[f'fisher_z_R2_no_{space_name}'] = fisher_z_restricted
            save_dict[f'fisher_z_R2_unique_{space_name}'] = fisher_z_unique

        # Compute shared variance and segregation index
        logger.info("")

        R2_shared = compute_shared_variance(results['R2_full'], results['R2_unique'], logger)
        segregation_index = compute_segregation_index(results['R2_full'], results['R2_unique'], logger)

        save_dict['R2_shared'] = R2_shared
        save_dict['segregation_index'] = segregation_index

        # Fisher z for shared variance
        fisher_z_shared = fisher_z_transform(R2_shared)
        save_dict['fisher_z_R2_shared'] = fisher_z_shared

    # Add product measures (if computed)
    if results['product_measure']:
        logger.info("Adding product measures...")
        for space_name, pm_values in results['product_measure'].items():
            save_dict[f'product_measure_{space_name}'] = pm_values
            logger.info(f"  {space_name}: mean={pm_values.mean():.6f}")

    np.savez_compressed(r2_file, **save_dict)
    logger.info("  Saved [OK]")

    # Save voxel mask
    voxel_mask_file = output_dir / 'valid_voxels_mask.npy'
    logger.info(f"Saving voxel mask to {voxel_mask_file}...")
    np.save(voxel_mask_file, valid_voxels_mask)
    logger.info("  Saved [OK]")

    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    logger.info(f"Saving metadata to {metadata_file}...")

    metadata = {
        'subject': subject,
        'train_sessions': train_sessions,
        'test_sessions': test_sessions,
        'n_voxels_original': int(len(valid_voxels_mask)),
        'n_voxels_kept': int(valid_voxels_mask.sum()),
        'n_features_original': int(len(feature_names)),
        'n_features_kept': int(valid_features_mask.sum()),
        'feature_spaces': {
            space: features for space, features in feature_spaces_filtered.items()
        },
        'feature_space_indices': {
            space: indices.tolist() for space, indices in space_to_indices.items()
        }
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved [OK]")

    # Print summary
    logger.info("")
    logger.info("RESULTS SUMMARY:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Models saved: model_full.pkl + restricted models")
    logger.info(f"  R2 scores: R2_scores.npz")
    logger.info(f"  Metadata: metadata.json")
    logger.info(f"  Voxel mask: valid_voxels_mask.npy")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Fit variance partitioning models with optional phase randomization testing'
    )
    parser.add_argument('--subject', type=int, required=True, help='Subject number')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--session', type=int, help='Single training session number')
    group.add_argument('--session-range', type=int, nargs=2, metavar=('START', 'END'),
                      help='Inclusive range of training session numbers (e.g., 6 10)')
    group.add_argument('--sessions', type=int, nargs='+', help='Explicit list of training session numbers')

    parser.add_argument('--test-sessions', type=int, nargs='+', required=True,
                       help='Session(s) to use as held-out test set')
    parser.add_argument('--backend', default='torch_cuda', choices=['torch_cuda', 'numpy', 'cupy'],
                       help='Himalaya computational backend')
    parser.add_argument('--skip-baseline-filtering', action='store_true',
                       help='Skip baseline/ITI filtering (use all TRs)')
    parser.add_argument('--compute-product-measure', action='store_true', default=True,
                       help='Compute product measure decomposition (default: True)')
    parser.add_argument('--no-compute-product-measure', action='store_false',
                       dest='compute_product_measure',
                       help='Skip product measure computation')
    parser.add_argument('--compute-unique-variance', action='store_true', default=False,
                       help='Compute unique variance by fitting N+1 models (default: False)')
    parser.add_argument('--no-compute-unique-variance', action='store_false',
                       dest='compute_unique_variance',
                       help='Skip unique variance computation (default)')

    parser.add_argument('--phase-randomize', action='store_true',
                       help='Phase randomize features to build null distribution for PCA significance testing')
    parser.add_argument('--randomize-space', type=str, default=None,
                       choices=['motor', 'scene', 'actions', 'activity'],
                       help='Randomize a single feature space (default: randomize all spaces)')
    parser.add_argument('--null-mode', type=str, default='joint',
                       choices=['independent', 'joint'],
                       help="Phase randomization mode (default: joint). "
                            "'independent' = per-feature random phases (breaks cross-feature "
                            "correlations -- inappropriate as null for PCA on weights). "
                            "'joint' = one shared random phase vector per run across all features "
                            "(preserves cross-spectral structure, recommended for PCA null).")
    parser.add_argument('--seed', type=int, default=None,
                       help="Random seed for phase randomization. Default: read from "
                            "SLURM_ARRAY_TASK_ID env var if set, else unseeded (irreproducible).")

    args = parser.parse_args()

    # Parse training sessions
    if args.session is not None:
        train_sessions = [args.session]
    elif args.session_range is not None:
        train_sessions = list(range(args.session_range[0], args.session_range[1] + 1))
    else:
        train_sessions = args.sessions

    test_sessions = args.test_sessions

    # Validate no overlap
    if set(train_sessions) & set(test_sessions):
        raise ValueError("Train and test sessions must not overlap!")

    # Setup logging
    session_range = (min(train_sessions), max(train_sessions))
    logger = setup_logging(args.subject, session_range, PATHS['logs'], args.phase_randomize)

    logger.info("="*80)
    logger.info("VARIANCE PARTITIONING WITH BANDED RIDGE REGRESSION")
    if args.phase_randomize:
        logger.info("*** PHASE RANDOMIZATION TEST MODE ***")
        logger.info(f"*** Randomizing: {args.randomize_space} features ***")
    logger.info("="*80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Training sessions: {train_sessions}")
    logger.info(f"Test sessions: {test_sessions}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Feature spaces: {list(FEATURE_SPACES.keys())}")
    logger.info(f"Compute product measure: {args.compute_product_measure}")
    logger.info(f"Compute unique variance: {args.compute_unique_variance}")
    logger.info(f"Phase randomize: {args.phase_randomize}")
    if args.phase_randomize:
        logger.info(f"Randomize space: {'all' if args.randomize_space is None else args.randomize_space}")
        logger.info(f"Null mode: {args.null_mode}")
        # Resolve seed: explicit arg > SLURM_ARRAY_TASK_ID > None (unseeded)
        if args.seed is not None:
            resolved_seed = args.seed
            seed_source = 'cli'
        elif 'SLURM_ARRAY_TASK_ID' in os.environ:
            resolved_seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
            seed_source = 'SLURM_ARRAY_TASK_ID'
        else:
            resolved_seed = None
            seed_source = 'none (irreproducible)'
        logger.info(f"Seed: {resolved_seed} (source: {seed_source})")
        args._resolved_seed = resolved_seed

    # Set backend
    backend = set_backend(args.backend, on_error="warn")
    logger.info(f"Using backend: {backend}")

    # Create experiment configuration
    logger.info("")
    logger.info("="*80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    params = PARAMETERS['variance_partitioning']
    config, experiment_id = create_experiment_config(
        feature_spaces=FEATURE_SPACES,
        delays=params['delays'],
        solver=params['solver'],
        n_iter=params['solver_params']['n_iter'],
        alphas=params['solver_params']['alphas'],
        baseline_filtering=not args.skip_baseline_filtering
    )

    config['phase_randomize'] = args.phase_randomize
    if args.phase_randomize:
        config['randomize_space'] = args.randomize_space if args.randomize_space else 'all'

    # Get output directory
    output_dir = get_output_directory(
        args.subject, train_sessions, test_sessions, experiment_id,
        PATHS['variance_partitioning_results'], args.phase_randomize
    )

    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Output directory: {output_dir}")

    # Save config
    config_path = output_dir / 'config.json'
    logger.info(f"Saving config to: {config_path}")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info("  Saved [OK]")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("LOADING DATA")
    logger.info("="*80)

    pipeline = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']
    logger.info(f"Preprocessing pipeline: {pipeline}")
    logger.info(f"fMRI path: {fmri_path}")

    logger.info("Loading training data...")
    X_train, Y_train, run_onsets_train, session_onsets_train, level_onsets_train, feature_names = concatenate_sessions_with_names(
        args.subject, train_sessions, PATHS['practice_phase_metadata'],
        PATHS['per_run_downsampled_to_TR'], fmri_path, pipeline, logger
    )

    logger.info("Loading test data...")
    X_test, Y_test, run_onsets_test, session_onsets_test, level_onsets_test, feature_names_test_original = concatenate_sessions_with_names(
        args.subject, test_sessions, PATHS['practice_phase_metadata'],
        PATHS['per_run_downsampled_to_TR'], fmri_path, pipeline, logger
    )

    if feature_names != feature_names_test_original:
        raise ValueError("Train and test data have different features before selection!")

    # ========================================================================
    # SELECT FEATURES ACCORDING TO CONFIG
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("FEATURE SELECTION FROM CONFIG")
    logger.info("="*80)

    logger.info("Selecting features for training data...")
    X_train, feature_names, selection_mask_train = select_and_validate_features(
        X_train, feature_names, FEATURE_SPACES, logger
    )

    logger.info("")
    logger.info("Selecting features for test data...")
    X_test, feature_names_test, selection_mask_test = select_and_validate_features(
        X_test, feature_names_test_original, FEATURE_SPACES, logger
    )

    # Verify selection masks match
    if not np.array_equal(selection_mask_train, selection_mask_test):
        raise ValueError("Feature selection masks differ between train and test data!")

    # ========================================================================
    # FILTER BASELINE PERIODS
    # ========================================================================
    if not args.skip_baseline_filtering:
        logger.info("")
        logger.info("="*80)
        logger.info("BASELINE FILTERING")
        logger.info("="*80)

        X_train, Y_train, run_onsets_train, session_onsets_train = filter_baseline_periods(
            X_train, Y_train, run_onsets_train, session_onsets_train, logger
        )
        X_test, Y_test, run_onsets_test, session_onsets_test = filter_baseline_periods(
            X_test, Y_test, run_onsets_test, session_onsets_test, logger
        )
    else:
        logger.info("")
        logger.info("="*80)
        logger.info("SKIPPING BASELINE FILTERING (using all TRs)")
        logger.info("="*80)


    # ========================================================================
    # FILTER ZERO-VARIANCE FEATURES AND VOXELS
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("FEATURE FILTERING")
    logger.info("="*80)

    X_train, X_test, valid_features_mask = filter_zero_variance_features(
        X_train, X_test, run_onsets_train, logger
    )

    logger.info("")
    logger.info("="*80)
    logger.info("VOXEL FILTERING")
    logger.info("="*80)

    Y_train, Y_test, valid_voxels_mask = filter_zero_variance_voxels(
        Y_train, Y_test, run_onsets_train, logger
    )

    # ========================================================================
    # MAP FEATURES TO SPACES (needed for phase randomization)
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("FEATURE SPACE MAPPING")
    logger.info("="*80)

    feature_spaces_filtered, space_to_indices = map_features_to_spaces(
        feature_names, valid_features_mask, FEATURE_SPACES, logger
    )

    # Validate all spaces have >0 features
    validate_all_spaces_nonempty(feature_spaces_filtered, logger)

    # ========================================================================
    # PREPROCESS DATA
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("PREPROCESSING")
    logger.info("="*80)

    logger.info("Preprocessing training data...")
    X_train, Y_train = preprocess_data(X_train, Y_train, run_onsets_train, level_onsets_train, logger)

    logger.info("Preprocessing test data...")
    X_test, Y_test = preprocess_data(X_test, Y_test, run_onsets_test, level_onsets_test, logger)

    # ========================================================================
    # PHASE RANDOMIZATION (if requested)
    # ========================================================================
    if args.phase_randomize:
        logger.info("")
        logger.info("="*80)
        spaces_to_randomize = (
            [args.randomize_space] if args.randomize_space
            else list(feature_spaces_filtered.keys())
        )
        logger.info(f"PHASE RANDOMIZATION: {spaces_to_randomize} (mode={args.null_mode})")
        logger.info("="*80)
        logger.info("Preserves autocorrelation structure, breaks temporal feature-fMRI relationships.")
        if args.null_mode == 'joint':
            logger.info("Joint mode also preserves cross-feature correlations.")
        logger.info("")

        # Build one RNG and use derived per-call RNGs so train and test do not
        # share state across spaces (but the run is reproducible from seed).
        base_seed = getattr(args, '_resolved_seed', None)
        ss = np.random.SeedSequence(base_seed) if base_seed is not None else np.random.SeedSequence()
        # Two child seeds per space (train, test), enumerated in spaces order
        child_seeds = ss.spawn(2 * len(spaces_to_randomize))

        for s_idx, space_name in enumerate(spaces_to_randomize):
            randomize_indices = space_to_indices[space_name]
            logger.info(f"Randomizing {len(randomize_indices)} features from '{space_name}'")

            rng_train = np.random.default_rng(child_seeds[2 * s_idx])
            rng_test = np.random.default_rng(child_seeds[2 * s_idx + 1])

            logger.info("  Phase randomizing TRAINING data...")
            X_train[:, randomize_indices] = phase_randomize_preserving_runs(
                X_train[:, randomize_indices], run_onsets_train, logger,
                mode=args.null_mode, rng=rng_train,
            )

            logger.info("  Phase randomizing TEST data...")
            X_test[:, randomize_indices] = phase_randomize_preserving_runs(
                X_test[:, randomize_indices], run_onsets_test, logger,
                mode=args.null_mode, rng=rng_test,
            )

        logger.info("")
        logger.info("Phase randomization complete. Proceeding with model fitting...")
        logger.info("="*80)

    # ========================================================================
    # APPLY DELAYS AND CREATE FEATURE SPACE ARRAYS
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("APPLYING DELAYS AND CREATING FEATURE SPACE ARRAYS")
    logger.info("="*80)

    params = PARAMETERS['variance_partitioning']
    delays = params['delays']
    n_features_original = X_train.shape[1]

    logger.info(f"Applying FIR delays: {delays}")

    # Apply StandardScaler (mean-centering only) and Delayer
    scaler = StandardScaler(with_mean=True, with_std=False)
    delayer = Delayer(delays=delays)

    X_train_scaled = scaler.fit_transform(X_train)
    X_train_delayed = delayer.fit_transform(X_train_scaled)

    X_test_scaled = scaler.transform(X_test)
    X_test_delayed = delayer.transform(X_test_scaled)

    logger.info(f"  Train shape after delays: {X_train_delayed.shape}")
    logger.info(f"  Test shape after delays: {X_test_delayed.shape}")

    # Expand feature space indices for delays
    space_to_indices_delayed = expand_feature_space_indices_for_delays(
        space_to_indices, n_features_original, delays, logger
    )

    # Create feature space arrays
    X_train_list, space_names_ordered = create_feature_space_arrays(
        X_train_delayed, space_to_indices_delayed, feature_spaces_filtered, logger
    )

    X_test_list, _ = create_feature_space_arrays(
        X_test_delayed, space_to_indices_delayed, feature_spaces_filtered, logger
    )

    # ========================================================================
    # FIT MODELS SEQUENTIALLY
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("FITTING MODELS")
    logger.info("="*80)

    results = fit_and_save_models_sequentially(
        X_train_list, Y_train, X_test_list, Y_test,
        space_names_ordered, run_onsets_train,
        params, backend, output_dir,
        args.compute_product_measure, args.compute_unique_variance,
        space_to_indices_delayed, len(delays), logger
    )

    # ========================================================================
    # VALIDATE METRICS
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("VALIDATION")
    logger.info("="*80)

    if results['R2_unique']:
        validate_variance_partition(results['R2_full'], results['R2_unique'], logger)
    else:
        logger.info("Unique variance not computed - skipping validation")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    logger.info("")
    save_results(
        args.subject, train_sessions, test_sessions, results,
        feature_spaces_filtered, space_to_indices, valid_voxels_mask,
        feature_names, valid_features_mask, output_dir, logger
    )

    # ========================================================================
    # COMPLETE
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("COMPLETE")
    logger.info("="*80)

    if args.phase_randomize:
        logger.info("")
        logger.info("REMINDER: This was a phase randomization null run.")
        logger.info("Run 100-500 times per subject to build the null eigenvalue distribution.")


if __name__ == '__main__':
    main()
