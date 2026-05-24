"""
Fit variance partitioning models using banded ridge regression.

This script fits GroupRidgeCV models to decompose explained variance into
unique contributions from different feature spaces (perception, motor, action, 
scene, activity).

Models are fit sequentially and saved immediately to manage memory:
1. Full model (all feature spaces)
2. Five restricted models (each excluding one feature space)

Unique variance is computed as: R2_unique_X = R2_full - R2_without_X
"""
import argparse
import json
import logging
import pickle
import gc
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from himalaya.backend import set_backend

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.utils.experiment_utils import create_experiment_config, save_experiment_config
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
    validate_all_spaces_nonempty,
    expand_feature_space_indices_for_delays,
    create_feature_space_arrays,
    fit_full_model,
    fit_restricted_model,
    compute_unique_variance,
    compute_shared_variance,
    compute_integration_index,
    validate_variance_partition,
    extract_model_weights,
    decompose_weights_by_feature_space,
    save_model_weights
)


def setup_logging(subject, session_range, log_dir):
    """Configure logging to both file and console."""
    log_file = log_dir / f'sub-{subject:02d}_ses-{session_range[0]:03d}-{session_range[1]:03d}_variance_partitioning.log'
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


def get_output_directory(subject, train_sessions, test_sessions, experiment_id, base_path):
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
    
    Returns:
    --------
    output_dir : Path
        Directory to save all models and results
    """
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    output_dir = base_path / dataset_dir / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fit_and_save_models_sequentially(X_train_list, Y_train, X_test_list, Y_test, 
                                    space_names_ordered, cv_onsets_train,
                                    space_to_indices_delayed, n_delays,
                                    params, backend, output_dir, 
                                    compute_product_measure, compute_unique_variance,
                                    save_model, logger):
    """
    Fit models and compute variance decomposition metrics.
    
    Extracts and saves weights immediately after fitting each model.
    Optionally saves full model objects if save_model=True (for permutation testing).
    
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
    cv_onsets_train : array
        CV fold boundaries (either run or session onsets depending on scheme)
    space_to_indices_delayed : dict
        Keys are space names, values are arrays of indices in delayed feature matrix
    n_delays : int
        Number of FIR delays
    params : dict
        Variance partitioning parameters from config
    backend : str
    output_dir : Path
    compute_product_measure : bool
        Whether to compute product measure decomposition
    compute_unique_variance : bool
        Whether to fit restricted models and compute unique variance
    save_model : bool
        Whether to save full model object (WARNING: large files)
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
        X_train_list, Y_train, cv_onsets_train, params, backend, logger
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
    
    # ========================================================================
    # EXTRACT AND SAVE WEIGHTS
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("EXTRACTING WEIGHTS FROM FULL MODEL")
    logger.info("="*80)
    
    weights_full = extract_model_weights(model_full, logger)
    weights_by_space = decompose_weights_by_feature_space(
        weights_full, space_to_indices_delayed, space_names_ordered, n_delays, logger
    )
    save_model_weights(
        weights_full, weights_by_space, space_names_ordered,
        output_dir, 'full', logger
    )

    # ========================================================================
    # SAVE HYPERPARAMETERS (per-band alphas for grid inspection)
    # ========================================================================
    logger.info("")
    logger.info("Saving hyperparameters from full model...")

    deltas = model_full.deltas_
    best_alphas = model_full.best_alphas_
    if hasattr(deltas, 'cpu'):
        deltas = deltas.cpu().numpy()
    if hasattr(best_alphas, 'cpu'):
        best_alphas = best_alphas.cpu().numpy()

    # Per-band effective alpha = 1 / exp(deltas[g, v]); see himalaya GroupRidgeCV docs.
    per_band_alpha = 1.0 / np.exp(deltas)

    hyper_file = output_dir / 'hyperparameters_full.npz'
    np.savez_compressed(
        hyper_file,
        deltas=deltas,
        best_alphas=best_alphas,
        per_band_alpha=per_band_alpha,
        space_names=np.array(space_names_ordered),
    )
    logger.info(f"  Saved to: {hyper_file}")
    for idx, name in enumerate(space_names_ordered):
        a = per_band_alpha[idx]
        logger.info(f"  {name}: alpha p5={np.percentile(a, 5):.3g}, "
                    f"p50={np.percentile(a, 50):.3g}, p95={np.percentile(a, 95):.3g}")
    
    # ========================================================================
    # OPTIONALLY SAVE FULL MODEL (for permutation testing)
    # ========================================================================
    if save_model:
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING FULL MODEL")
        logger.info("="*80)
        logger.warning("WARNING: Model files are very large (~100GB)")
        
        model_path = output_dir / 'model_full.pkl'
        logger.info(f"Saving model to: {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(model_full, f)
        logger.info("  Saved [OK]")
    
    # Delete model to free memory
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
                cv_onsets_train, params, backend, logger
            )
            
            # Score on test set
            logger.info("Scoring restricted model on test set...")
            R2_restricted = model_restricted.score(X_test_restricted, Y_test)
            
            # Convert to numpy if needed
            if hasattr(R2_restricted, 'cpu'):
                R2_restricted = R2_restricted.cpu().numpy()
            
            results['R2_restricted'][space_name] = R2_restricted
            
            logger.info(f"  Mean test R2 (without {space_name}): {R2_restricted.mean():.6f}")
            
            # Compute unique variance immediately
            R2_unique = results['R2_full'] - R2_restricted
            results['R2_unique'][space_name] = R2_unique
            
            logger.info(f"  Mean unique R2 for {space_name}: {R2_unique.mean():.6f}")
            logger.info(f"  % voxels with positive unique R2: {(R2_unique > 0).sum() / len(R2_unique) * 100:.1f}%")
            
            logger.info(f"  Skipping weight extraction for restricted model (not needed)")
            
            # Clean up
            del model_restricted
            gc.collect()
            logger.info("  Freed model memory [OK]")
    else:
        logger.info("")
        logger.info("="*80)
        logger.info("SKIPPING UNIQUE VARIANCE COMPUTATION (not requested)")
        logger.info("="*80)
    
    return results


def save_results(subject, train_sessions, test_sessions, results, 
                feature_spaces_filtered, space_to_indices, valid_voxels_mask,
                feature_names, valid_features_mask, output_dir, logger):
    """
    Save R2 scores, Fisher z transforms, product measures, and metadata.
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    results : dict
        Contains R2_full, optionally R2_restricted, R2_unique, product_measure
    feature_spaces_filtered : dict
    space_to_indices : dict
    valid_voxels_mask : boolean array
    feature_names : list of str
    valid_features_mask : boolean array
    output_dir : Path
    logger : logging.Logger
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
        
        # Compute shared variance and integration index
        logger.info("")

        R2_shared = compute_shared_variance(results['R2_full'], results['R2_unique'], logger)
        
        save_dict['R2_shared'] = R2_shared
        
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
    
    # Save feature mask (for permutation testing)
    features_mask_file = output_dir / 'valid_features_mask.npy'
    logger.info(f"Saving features mask to {features_mask_file}...")
    np.save(features_mask_file, valid_features_mask)
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
        'computed_metrics': {
            'R2_full': True,
            'fisher_z': True,
            'product_measure': bool(results['product_measure']),
            'unique_variance': bool(results['R2_unique'])
        },
        'feature_spaces': {
            name: features for name, features in feature_spaces_filtered.items()
        },
        'feature_space_sizes': {
            name: len(features) for name, features in feature_spaces_filtered.items()
        },
        'original_feature_names': feature_names,
        'kept_feature_names': [name for name, keep in zip(feature_names, valid_features_mask) if keep],
        'delays': PARAMETERS['variance_partitioning']['delays'],
        'feature_spaces_filtered': feature_spaces_filtered
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved [OK]")
    
    # Print summary
    logger.info("")
    logger.info("RESULTS SUMMARY:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Weights saved: weights_full.npz")
    logger.info(f"  R2 scores: R2_scores.npz")
    logger.info(f"  Metadata: metadata.json")
    logger.info(f"  Voxel mask: valid_voxels_mask.npy")
    logger.info(f"  Features mask: valid_features_mask.npy")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Fit variance partitioning models using banded ridge regression'
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
    parser.add_argument('--cv-scheme', type=str, default='loro',
                       choices=['loro', 'loso'],
                       help='Cross-validation scheme: loro (leave-one-run-out) or loso (leave-one-session-out)')
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
    parser.add_argument('--save-model', action='store_true', default=False,
                       help='Save full model for permutation testing (WARNING: large files ~100GB)')
    
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
    
    # Resolve fMRI path and pipeline from config
    pipeline = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']

    # Setup logging
    session_range = (min(train_sessions), max(train_sessions))
    logger = setup_logging(args.subject, session_range, PATHS['logs'])
    
    logger.info("="*80)
    logger.info("VARIANCE PARTITIONING WITH BANDED RIDGE REGRESSION")
    logger.info("="*80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Training sessions: {train_sessions}")
    logger.info(f"Test sessions: {test_sessions}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Preprocessing pipeline: {pipeline}")
    logger.info(f"fMRI path: {fmri_path}")
    logger.info(f"Feature spaces: {list(FEATURE_SPACES.keys())}")
    logger.info(f"Compute product measure: {args.compute_product_measure}")
    logger.info(f"Compute unique variance: {args.compute_unique_variance}")
    logger.info(f"Save model: {args.save_model}")
    
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
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Timestamp: {config['timestamp']}")
    logger.info(f"Delays: {config['variance_partitioning']['delays']}")
    logger.info(f"Solver: {config['variance_partitioning']['solver']}")
    logger.info(f"N iterations: {config['variance_partitioning']['solver_params']['n_iter']}")
    logger.info(f"N alphas: {len(config['variance_partitioning']['solver_params']['alphas'])}")
    logger.info(f"Baseline filtering: {config['preprocessing']['baseline_filtering']}")
    
    # Create output directory with experiment ID
    output_dir = get_output_directory(
        args.subject, train_sessions, test_sessions, experiment_id,
        PATHS['variance_partitioning']
    )
    logger.info(f"Output directory: {output_dir}")
    
    # Save experiment config
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
    # COMPUTE BASELINE MASK (do not drop yet — needed for per-run delaying)
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("BASELINE MASK")
    logger.info("="*80)

    if not args.skip_baseline_filtering:
        active_mask_train = compute_baseline_mask(X_train, run_onsets_train, logger=logger)
        active_mask_test = compute_baseline_mask(X_test, run_onsets_test, logger=logger)
    else:
        logger.info("Skipping baseline filtering (using all TRs)")
        active_mask_train = np.ones(X_train.shape[0], dtype=bool)
        active_mask_test = np.ones(X_test.shape[0], dtype=bool)

    # ========================================================================
    # FILTER ZERO-VARIANCE FEATURES AND VOXELS (on full data, ITI included)
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
    # MAP FEATURES TO SPACES
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("FEATURE SPACE MAPPING")
    logger.info("="*80)

    feature_spaces_filtered, space_to_indices = map_features_to_spaces(
        feature_names, valid_features_mask, FEATURE_SPACES, logger
    )

    validate_all_spaces_nonempty(feature_spaces_filtered, logger)

    # ========================================================================
    # PREPROCESS DATA (z-score Y within runs on full data, NaN to 0)
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("PREPROCESSING")
    logger.info("="*80)

    logger.info("Preprocessing training data (Y z-scored within runs, including baseline TRs)...")
    X_train, Y_train = preprocess_data(X_train, Y_train, run_onsets_train, level_onsets_train, logger)

    logger.info("Preprocessing test data...")
    X_test, Y_test = preprocess_data(X_test, Y_test, run_onsets_test, level_onsets_test, logger)

    # ========================================================================
    # MEAN-CENTER X (training active mean) AND APPLY DELAYS PER RUN
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("MEAN-CENTERING AND APPLYING DELAYS PER RUN")
    logger.info("="*80)

    params = PARAMETERS['variance_partitioning']
    delays = params['delays']
    n_features_original = X_train.shape[1]

    # Fit centering on active training TRs only so post-drop active samples have mean ~0.
    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(X_train[active_mask_train])
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    logger.info(f"Mean-centering: fit on {int(active_mask_train.sum())} active training TRs; "
                f"applied to full train ({len(X_train)}) and test ({len(X_test)})")

    logger.info(f"Applying FIR delays per run: {delays}")
    X_train_delayed = apply_delays_per_run(X_train, run_onsets_train, delays)
    X_test_delayed = apply_delays_per_run(X_test, run_onsets_test, delays)
    logger.info(f"  Train delayed shape (pre-baseline-drop): {X_train_delayed.shape}")
    logger.info(f"  Test delayed shape (pre-baseline-drop): {X_test_delayed.shape}")

    # ========================================================================
    # DROP BASELINE TRs (after delays — preserves causal ITI context in lags)
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("DROPPING BASELINE TRs")
    logger.info("="*80)

    X_train_delayed, Y_train, run_onsets_train, session_onsets_train = drop_baseline_samples(
        X_train_delayed, Y_train, run_onsets_train, session_onsets_train, active_mask_train, logger
    )
    X_test_delayed, Y_test, run_onsets_test, session_onsets_test = drop_baseline_samples(
        X_test_delayed, Y_test, run_onsets_test, session_onsets_test, active_mask_test, logger
    )

    # ========================================================================
    # SELECT CV SCHEME (after baseline drop so onsets index active samples)
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("CROSS-VALIDATION SCHEME")
    logger.info("="*80)

    if args.cv_scheme == 'loro':
        cv_onsets_train = run_onsets_train
        logger.info("Using LORO (Leave-One-Run-Out) CV")
    elif args.cv_scheme == 'loso':
        cv_onsets_train = session_onsets_train
        logger.info("Using LOSO (Leave-One-Session-Out) CV")
        logger.info(f"  Training sessions: {train_sessions}")
    logger.info(f"  Number of CV folds: {len(cv_onsets_train)}")

    # ========================================================================
    # SPLIT DELAYED X INTO PER-SPACE ARRAYS
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("CREATING FEATURE SPACE ARRAYS")
    logger.info("="*80)

    space_to_indices_delayed = expand_feature_space_indices_for_delays(
        space_to_indices, n_features_original, delays, logger
    )

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
        space_names_ordered, cv_onsets_train,
        space_to_indices_delayed, len(delays),
        params, backend, output_dir,
        args.compute_product_measure, args.compute_unique_variance,
        args.save_model, logger
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


if __name__ == '__main__':
    main()
