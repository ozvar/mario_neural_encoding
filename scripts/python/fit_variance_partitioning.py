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
from sklearn.pipeline import make_pipeline
from voxelwise_tutorials.delayer import Delayer
from himalaya.backend import set_backend

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.utils.data_loading import (
    concatenate_sessions_with_names,
    filter_baseline_periods,
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
    validate_variance_partition
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


def get_output_directory(subject, train_sessions, test_sessions, base_path):
    """
    Create output directory for this subject/session combination.
    
    Returns:
    --------
    output_dir : Path
        Directory to save all models and results
    """
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dir_name = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    
    output_dir = base_path / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def fit_and_save_models_sequentially(X_train_list, Y_train, X_test_list, Y_test, 
                                    space_names_ordered, run_onsets_train,
                                    params, backend, output_dir, logger):
    """
    Fit all models sequentially, saving each immediately to avoid memory issues.
    
    Returns R2 scores only, not models (models are saved to disk).
    
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
    logger : logging.Logger
        
    Returns:
    --------
    results : dict with keys 'R2_full', 'R2_restricted', 'R2_unique'
    """
    results = {
        'R2_full': None,
        'R2_restricted': {},
        'R2_unique': {}
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
    
    logger.info(f"  Mean test R2: {results['R2_full'].mean():.6f}")
    logger.info(f"  Median test R2: {np.median(results['R2_full']):.6f}")
    logger.info(f"  Max test R2: {results['R2_full'].max():.6f}")
    
    # Save model immediately
    model_path = output_dir / 'model_full.pkl'
    logger.info(f"Saving full model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model_full, f)
    logger.info("  Saved [OK]“")
    
    # Delete to free memory
    del model_full
    gc.collect()
    logger.info("  Freed memory [OK]“")
    
    # ========================================================================
    # FIT RESTRICTED MODELS (one per feature space)
    # ========================================================================
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
        logger.info("Scoring restricted model on test set...")
        R2_restricted = model_restricted.score(X_test_restricted, Y_test)
        
        # Convert to numpy if needed
        if hasattr(R2_restricted, 'cpu'):
            R2_restricted = R2_restricted.cpu().numpy()
        
        results['R2_restricted'][space_name] = R2_restricted
        
        logger.info(f"  Mean test R2 (without {space_name}): {R2_restricted.mean():.6f}")
        logger.info(f"  Median test R2: {np.median(R2_restricted):.6f}")
        
        # Compute unique variance immediately
        R2_unique = results['R2_full'] - R2_restricted
        results['R2_unique'][space_name] = R2_unique
        
        logger.info(f"  Mean unique R2 for {space_name}: {R2_unique.mean():.6f}")
        logger.info(f"  Median unique R2: {np.median(R2_unique):.6f}")
        logger.info(f"  % voxels with positive unique R2: {(R2_unique > 0).sum() / len(R2_unique) * 100:.1f}%")
        
        # Save model
        model_path = output_dir / f'model_no_{space_name}.pkl'
        logger.info(f"Saving restricted model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(model_restricted, f)
        logger.info("  Saved [OK]“")
        
        # Clean up
        del model_restricted
        gc.collect()
        logger.info("  Freed memory [OK]“")
    
    return results


def save_results(subject, train_sessions, test_sessions, results, 
                feature_spaces_filtered, space_to_indices, valid_voxels_mask,
                feature_names, valid_features_mask, output_dir, logger):
    """
    Save R2 scores and metadata.
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    results : dict
        Contains R2_full, R2_restricted, R2_unique
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
    
    # Save R2 scores in compressed format
    r2_file = output_dir / 'R2_scores.npz'
    logger.info(f"Saving R2 scores to {r2_file}...")
    
    save_dict = {
        'R2_full': results['R2_full'],
    }
    
    # Add restricted and unique R2 for each space
    for space_name in results['R2_restricted'].keys():
        save_dict[f'R2_no_{space_name}'] = results['R2_restricted'][space_name]
        save_dict[f'R2_unique_{space_name}'] = results['R2_unique'][space_name]
    
    np.savez_compressed(r2_file, **save_dict)
    logger.info("  Saved [OK]“")
    
    # Save voxel mask
    voxel_mask_file = output_dir / 'valid_voxels_mask.npy'
    logger.info(f"Saving voxel mask to {voxel_mask_file}...")
    np.save(voxel_mask_file, valid_voxels_mask)
    logger.info("  Saved [OK]“")
    
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
            name: features for name, features in feature_spaces_filtered.items()
        },
        'feature_space_sizes': {
            name: len(features) for name, features in feature_spaces_filtered.items()
        },
        'original_feature_names': feature_names,
        'kept_feature_names': [name for name, keep in zip(feature_names, valid_features_mask) if keep]
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved [OK]“")
    
    # Print summary
    logger.info("")
    logger.info("RESULTS SUMMARY:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Models saved: model_full.pkl + 5 restricted models")
    logger.info(f"  R2 scores: R2_scores.npz")
    logger.info(f"  Metadata: metadata.json")
    logger.info(f"  Voxel mask: valid_voxels_mask.npy")


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
    parser.add_argument('--skip-baseline-filtering', action='store_true',
                       help='Skip baseline/ITI filtering (use all TRs)')
    
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
    logger = setup_logging(args.subject, session_range, PATHS['logs'])
    
    logger.info("="*80)
    logger.info("VARIANCE PARTITIONING WITH BANDED RIDGE REGRESSION")
    logger.info("="*80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Training sessions: {train_sessions}")
    logger.info(f"Test sessions: {test_sessions}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Feature spaces: {list(FEATURE_SPACES.keys())}")
    
    # Set backend
    backend = set_backend(args.backend, on_error="warn")
    logger.info(f"Using backend: {backend}")
    
    # Create output directory
    output_dir = get_output_directory(
        args.subject, train_sessions, test_sessions, 
        PATHS['variance_partitioning']
    )
    logger.info(f"Output directory: {output_dir}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("LOADING DATA")
    logger.info("="*80)
    
    logger.info("Loading training data...")
    X_train, Y_train, run_onsets_train, level_onsets_train, feature_names = concatenate_sessions_with_names(
        args.subject, train_sessions, PATHS['practice_phase_metadata'],
        PATHS['per_run_downsampled_to_TR'], PATHS['fmriprep_data'], logger
    )
    
    logger.info("Loading test data...")
    X_test, Y_test, run_onsets_test, level_onsets_test, _ = concatenate_sessions_with_names(
        args.subject, test_sessions, PATHS['practice_phase_metadata'],
        PATHS['per_run_downsampled_to_TR'], PATHS['fmriprep_data'], logger
    )
    
    # ========================================================================
    # FILTER BASELINE PERIODS
    # ========================================================================
    if not args.skip_baseline_filtering:
        logger.info("")
        logger.info("="*80)
        logger.info("BASELINE FILTERING")
        logger.info("="*80)
        
        X_train, Y_train, run_onsets_train = filter_baseline_periods(
            X_train, Y_train, run_onsets_train, logger
        )
        X_test, Y_test, run_onsets_test = filter_baseline_periods(
            X_test, Y_test, run_onsets_test, logger
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
    # MAP FEATURES TO SPACES
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
        params, backend, output_dir, logger
    )
    
    # ========================================================================
    # VALIDATE VARIANCE PARTITIONING
    # ========================================================================
    logger.info("")
    logger.info("="*80)
    logger.info("VALIDATION")
    logger.info("="*80)
    
    validate_variance_partition(results['R2_full'], results['R2_unique'], logger)
    
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
