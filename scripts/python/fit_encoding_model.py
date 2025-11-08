"""
Fit voxelwise encoding models on practice phase data.

This script fits ridge regression models with FIR delays to predict fMRI responses
from behavioral features. Models are trained using leave-one-run-out cross-validation
on training sessions and evaluated on held-out test sessions.
"""
import argparse
import json
import logging
import pickle
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.utils import generate_leave_one_run_out, zscore_runs
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend
from scipy.stats import zscore

from mario_encoding.config import PATHS, PARAMETERS


def setup_logging(subject, session_range, log_dir):
    """Configure logging to both file and console."""
    log_file = log_dir / f'sub-{subject:02d}_ses-{session_range[0]:03d}-{session_range[1]:03d}_fit.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_practice_metadata(subject, session, practice_metadata_path):
    """
    Load practice phase metadata JSON for one session.
    
    Parameters:
    -----------
    subject : int
    session : int
    practice_metadata_path : Path
        
    Returns:
    --------
    metadata : dict
    """
    metadata_path = practice_metadata_path / f'sub-{subject:02d}_ses-{session:03d}_practice_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Practice metadata not found: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def load_practice_features(subject, session, practice_runs, downsampled_path):
    """
    Load behavioral features for practice runs and concatenate.
    
    Parameters:
    -----------
    subject : int
    session : int
    practice_runs : list of int
    downsampled_path : Path
        
    Returns:
    --------
    X : array of shape (n_samples, n_features)
    """
    X_list = []
    for run in practice_runs:
        filepath = (downsampled_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 
                   f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv')
        if not filepath.exists():
            raise FileNotFoundError(f"Downsampled features not found: {filepath}")
        df = pd.read_csv(filepath, sep='\t')
        feature_cols = [col for col in df.columns if col not in ['TR_index', 'TR_time', 'level']]
        X_list.append(df[feature_cols].values)
    X = np.vstack(X_list).astype('float32')
    return X


def load_practice_fmri(subject, session, practice_runs, fmriprep_path):
    """
    Load CIFTI fMRI data for practice runs and concatenate.
    
    Parameters:
    -----------
    subject : int
    session : int
    practice_runs : list of int
    fmriprep_path : Path
        
    Returns:
    --------
    Y : array of shape (n_samples, n_grayordinates)
    """
    Y_list = []
    for run in practice_runs:
        filepath = (fmriprep_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func' /
                   f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run}_space-fsLR_den-91k_bold.dtseries.nii')
        if not filepath.exists():
            raise FileNotFoundError(f"CIFTI file not found: {filepath}")
        cifti = nib.load(str(filepath))
        fmri_data = cifti.get_fdata()
        Y_list.append(fmri_data)
    Y = np.vstack(Y_list).astype('float32')
    return Y




def validate_session_files(subject, session, metadata, downsampled_path, fmriprep_path):
    """
    Validate that all required files exist before loading.
    
    Parameters:
    -----------
    subject : int
    session : int
    metadata : dict
    downsampled_path : Path
    fmriprep_path : Path
    """
    practice_runs = metadata['practice_runs']
    for run in practice_runs:
        filepath = (downsampled_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 
                   f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv')
        if not filepath.exists():
            raise FileNotFoundError(f"Missing behavioral features: {filepath}")
    for run in practice_runs:
        filepath = (fmriprep_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func' /
                   f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run}_space-fsLR_den-91k_bold.dtseries.nii')
        if not filepath.exists():
            raise FileNotFoundError(f"Missing CIFTI file: {filepath}")


def load_session_data(subject, session, practice_metadata_path, downsampled_path, fmriprep_path, logger):
    """
    Load all practice phase data for one session.
    
    Parameters:
    -----------
    subject : int
    session : int
    practice_metadata_path : Path
    downsampled_path : Path
    fmriprep_path : Path
    logger : logging.Logger
        
    Returns:
    --------
    X : array of shape (n_samples, n_features)
    Y : array of shape (n_samples, n_grayordinates)
    metadata : dict
    """
    logger.info(f"Loading data for sub-{subject:02d} ses-{session:03d}")
    metadata = load_practice_metadata(subject, session, practice_metadata_path)
    logger.info(f"  Found {metadata['n_runs']} practice runs: {metadata['practice_runs']}")
    logger.info(f"  Total samples: {metadata['n_samples']}")
    logger.info(f"  Total levels: {metadata['n_levels']}")
    validate_session_files(subject, session, metadata, downsampled_path, fmriprep_path)
    logger.info("  All required files validated")
    X = load_practice_features(subject, session, metadata['practice_runs'], downsampled_path)
    logger.info(f"  Loaded features: {X.shape}")
    Y = load_practice_fmri(subject, session, metadata['practice_runs'], fmriprep_path)
    logger.info(f"  Loaded fMRI: {Y.shape}")
    if X.shape[0] != metadata['n_samples']:
        raise ValueError(f"Feature samples ({X.shape[0]}) != metadata n_samples ({metadata['n_samples']})")
    if Y.shape[0] != metadata['n_samples']:
        raise ValueError(f"fMRI samples ({Y.shape[0]}) != metadata n_samples ({metadata['n_samples']})")
    return X, Y, metadata


def concatenate_sessions(subject, sessions, practice_metadata_path, downsampled_path, fmriprep_path, logger):
    """
    Load and concatenate data from multiple sessions.
    
    Parameters:
    -----------
    subject : int
    sessions : list of int
    practice_metadata_path : Path
    downsampled_path : Path
    fmriprep_path : Path
    logger : logging.Logger
        
    Returns:
    --------
    X : array of shape (n_samples_total, n_features)
    Y : array of shape (n_samples_total, n_grayordinates)
    run_onsets : array of int
    level_onsets : array of int
    """
    X_list = []
    Y_list = []
    run_onsets_list = [0]
    level_onsets_list = [0]
    cumulative_samples = 0
    
    for session in sessions:
        X_sess, Y_sess, metadata = load_session_data(subject, session, practice_metadata_path, 
                                                      downsampled_path, fmriprep_path, logger)
        X_list.append(X_sess)
        Y_list.append(Y_sess)
        run_onsets_sess = np.array(metadata['run_onsets']) + cumulative_samples
        level_onsets_sess = np.array(metadata['level_onsets']) + cumulative_samples
        run_onsets_list.extend(run_onsets_sess[1:].tolist())
        level_onsets_list.extend(level_onsets_sess[1:].tolist())
        cumulative_samples += metadata['n_samples']
    
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    run_onsets = np.array(run_onsets_list)
    level_onsets = np.array(level_onsets_list)
    logger.info(f"Concatenated {len(sessions)} sessions:")
    logger.info(f"  Total samples: {X.shape[0]}")
    logger.info(f"  Total runs: {len(run_onsets)}")
    logger.info(f"  Total levels: {len(level_onsets)}")
    return X, Y, run_onsets, level_onsets


def preprocess_data(X, Y, run_onsets, level_onsets, logger):
    """
    Preprocess features and fMRI responses.
    
    Parameters:
    -----------
    X : array of shape (n_samples, n_features)
    Y : array of shape (n_samples, n_grayordinates)
    run_onsets : array of int
    level_onsets : array of int (unused, kept for compatibility)
    logger : logging.Logger
        
    Returns:
    --------
    X_preprocessed : array of shape (n_samples, n_features)
    Y_preprocessed : array of shape (n_samples, n_grayordinates_valid)
    """
    logger.info("Preprocessing data...")
    logger.info("  Z-scoring features within runs")
    X = zscore_runs(X, run_onsets)
    logger.info("  Z-scoring fMRI within runs")
    Y = zscore_runs(Y, run_onsets)
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    logger.info(f"  Final X shape: {X.shape}, dtype: {X.dtype}")
    logger.info(f"  Final Y shape: {Y.shape}, dtype: {Y.dtype}")
    return X, Y


def fit_and_evaluate(X_train, Y_train, X_test, Y_test, run_onsets_train, params, backend, logger):
    """
    Fit model on training data and evaluate on test data.
    
    Note: Delayer is applied BEFORE creating CV object to ensure proper alignment.
    
    Parameters:
    -----------
    X_train : array of shape (n_samples_train, n_features)
    Y_train : array of shape (n_samples_train, n_grayordinates)
    X_test : array of shape (n_samples_test, n_features)
    Y_test : array of shape (n_samples_test, n_grayordinates)
    run_onsets_train : array of int
    params : dict
    backend : himalaya backend
    logger : logging.Logger
        
    Returns:
    --------
    pipeline : fitted Pipeline
    cv_scores : array of shape (n_grayordinates,)
    test_scores : array of shape (n_grayordinates,)
    best_alphas : array of shape (n_grayordinates,)
    """
    # Apply Delayer BEFORE creating CV to ensure proper index alignment
    logger.info("Applying FIR delays to features...")
    delayer = Delayer(delays=params['delays'])
    X_train_delayed = delayer.fit_transform(X_train)
    X_test_delayed = delayer.transform(X_test)
    logger.info(f"  Train features: {X_train.shape} → {X_train_delayed.shape}")
    logger.info(f"  Test features: {X_test.shape} → {X_test_delayed.shape}")
    
    # NOW create CV object on the delayed data
    alphas = np.logspace(params['alpha_min'], params['alpha_max'], params['n_alphas'])
    logger.info(f"Testing {len(alphas)} alpha values from 10^{params['alpha_min']} to 10^{params['alpha_max']}")
    cv = generate_leave_one_run_out(X_train_delayed.shape[0], run_onsets_train)
    logger.info(f"Using leave-one-run-out CV with {len(list(cv))} folds")
    
    # Create pipeline WITHOUT Delayer (already applied)
    pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        KernelRidgeCV(alphas=alphas, cv=cv, solver_params=params['solver_params'])
    )
    
    logger.info("Fitting model...")
    pipeline.fit(X_train_delayed, Y_train)
    logger.info("  Model fitting complete")
    cv_scores = backend.to_numpy(pipeline.score(X_train_delayed, Y_train))
    best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
    logger.info(f"  Mean CV R² score: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")
    logger.info(f"  Median CV R² score: {np.median(cv_scores):.4f}")
    logger.info(f"  Max CV R² score: {cv_scores.max():.4f}")
    logger.info("Evaluating on test set...")
    test_scores = backend.to_numpy(pipeline.score(X_test_delayed, Y_test))
    logger.info(f"  Mean test R² score: {test_scores.mean():.4f} (std: {test_scores.std():.4f})")
    logger.info(f"  Median test R² score: {np.median(test_scores):.4f}")
    logger.info(f"  Max test R² score: {test_scores.max():.4f}")
    return pipeline, cv_scores, test_scores, best_alphas


def save_results(subject, train_sessions, test_sessions, pipeline, cv_scores, test_scores, 
                best_alphas, voxel_mask, models_path, cv_scores_path, logger):
    """
    Save fitted model and evaluation results.
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    pipeline : fitted Pipeline
    cv_scores : array
    test_scores : array
    best_alphas : array
    voxel_mask : boolean array
    models_path : Path
    cv_scores_path : Path
    logger : logging.Logger
    """
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    base_name = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    
    model_path = models_path / f'{base_name}_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove CV generator before pickling (it's not needed for prediction)
    pipeline_copy = pipeline
    if hasattr(pipeline[-1], 'cv'):
        pipeline[-1].cv = None
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline_copy, f)
    logger.info(f"Saved model to: {model_path}")
    
    cv_scores_file = cv_scores_path / f'{base_name}_cv_scores.npy'
    cv_scores_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cv_scores_file, cv_scores)
    logger.info(f"Saved CV scores to: {cv_scores_file}")
    
    test_scores_file = cv_scores_path / f'{base_name}_test_scores.npy'
    np.save(test_scores_file, test_scores)
    logger.info(f"Saved test scores to: {test_scores_file}")
    
    alphas_file = cv_scores_path / f'{base_name}_best_alphas.npy'
    np.save(alphas_file, best_alphas)
    logger.info(f"Saved best alphas to: {alphas_file}")
    
    voxel_mask_file = cv_scores_path / f'{base_name}_voxel_mask.npy'
    np.save(voxel_mask_file, voxel_mask)
    logger.info(f"Saved voxel mask to: {voxel_mask_file}")


def filter_zero_variance_features(X_train, X_test, run_onsets_train, logger):
    """
    Filter features with zero variance in any training run.
    
    Parameters:
    -----------
    X_train : array of shape (n_samples_train, n_features)
    X_test : array of shape (n_samples_test, n_features)
    run_onsets_train : array of int
    logger : logging.Logger
        
    Returns:
    --------
    X_train_filtered : array
    X_test_filtered : array
    valid_features_mask : boolean array
    """
    logger.info("Identifying zero-variance features on training data (checking within each run)...")
    run_splits_train_X = np.split(X_train, run_onsets_train[1:])
    zero_var_in_any_run_X = np.zeros(X_train.shape[1], dtype=bool)
    for i, run_data in enumerate(run_splits_train_X):
        run_var = run_data.var(axis=0)
        zero_var_this_run = (run_var < 1e-10)
        zero_var_in_any_run_X |= zero_var_this_run
        if zero_var_this_run.sum() > 0:
            logger.info(f"  Run {i+1}: {zero_var_this_run.sum()} features with zero variance")
    
    valid_features_mask = ~zero_var_in_any_run_X
    n_features_original = X_train.shape[1]
    n_features_kept = valid_features_mask.sum()
    n_features_dropped = n_features_original - n_features_kept
    logger.info(f"  Original features: {n_features_original}")
    logger.info(f"  Zero-variance in at least one run: {n_features_dropped}")
    logger.info(f"  Kept features: {n_features_kept}")
    
    if n_features_dropped > 0:
        logger.info("Applying feature mask to both train and test data...")
        X_train = X_train[:, valid_features_mask]
        X_test = X_test[:, valid_features_mask]
        logger.info(f"  Train X shape: {X_train.shape}")
        logger.info(f"  Test X shape: {X_test.shape}")
    
    return X_train, X_test, valid_features_mask


def filter_zero_variance_voxels(Y_train, Y_test, run_onsets_train, logger):
    """
    Filter voxels with zero variance in any training run.
    
    Parameters:
    -----------
    Y_train : array of shape (n_samples_train, n_grayordinates)
    Y_test : array of shape (n_samples_test, n_grayordinates)
    run_onsets_train : array of int
    logger : logging.Logger
        
    Returns:
    --------
    Y_train_filtered : array
    Y_test_filtered : array
    valid_voxels_mask : boolean array
    """
    logger.info("Identifying zero-variance voxels on training data (checking within each run)...")
    run_splits_train = np.split(Y_train, run_onsets_train[1:])
    zero_var_in_any_run = np.zeros(Y_train.shape[1], dtype=bool)
    for i, run_data in enumerate(run_splits_train):
        run_var = run_data.var(axis=0)
        zero_var_this_run = (run_var < 1e-10)
        zero_var_in_any_run |= zero_var_this_run
        if zero_var_this_run.sum() > 0:
            logger.info(f"  Run {i+1}: {zero_var_this_run.sum()} voxels with zero variance")
    
    valid_voxels_mask = ~zero_var_in_any_run
    n_voxels_original = Y_train.shape[1]
    n_voxels_kept = valid_voxels_mask.sum()
    n_voxels_dropped = n_voxels_original - n_voxels_kept
    logger.info(f"  Original voxels: {n_voxels_original}")
    logger.info(f"  Zero-variance in at least one run: {n_voxels_dropped} ({n_voxels_dropped/n_voxels_original*100:.2f}%)")
    logger.info(f"  Kept voxels: {n_voxels_kept}")
    
    if n_voxels_dropped > 0:
        logger.info("Applying voxel mask to both train and test data...")
        Y_train = Y_train[:, valid_voxels_mask]
        Y_test = Y_test[:, valid_voxels_mask]
        logger.info(f"  Train Y shape: {Y_train.shape}")
        logger.info(f"  Test Y shape: {Y_test.shape}")
    
    return Y_train, Y_test, valid_voxels_mask


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Fit voxelwise encoding models on practice phase data')
    parser.add_argument('--subject', type=int, required=True, help='Subject number')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--session', type=int, help='Single session number')
    group.add_argument('--session-range', type=int, nargs=2, metavar=('START', 'END'),
                      help='Inclusive range of session numbers (e.g., 6 10)')
    group.add_argument('--sessions', type=int, nargs='+', help='Explicit list of session numbers')
    parser.add_argument('--test-sessions', type=int, nargs='+', required=True,
                       help='Session(s) to use as held-out test set')
    parser.add_argument('--backend', default='torch_cuda', choices=['torch_cuda', 'numpy', 'cupy'],
                       help='Himalaya computational backend')
    args = parser.parse_args()
    
    if args.session is not None:
        train_sessions = [args.session]
    elif args.session_range is not None:
        train_sessions = list(range(args.session_range[0], args.session_range[1] + 1))
    else:
        train_sessions = args.sessions
    test_sessions = args.test_sessions
    if set(train_sessions) & set(test_sessions):
        raise ValueError("Train and test sessions must not overlap!")
    
    session_range = (min(train_sessions), max(train_sessions))
    logger = setup_logging(args.subject, session_range, PATHS['fit_logs'])
    logger.info("="*80)
    logger.info("VOXELWISE ENCODING MODEL FITTING")
    logger.info("="*80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Training sessions: {train_sessions}")
    logger.info(f"Test sessions: {test_sessions}")
    logger.info(f"Backend: {args.backend}")
    backend = set_backend(args.backend, on_error="warn")
    logger.info(f"Using backend: {backend}")
    
    logger.info("\nLoading training data...")
    X_train, Y_train, run_onsets_train, level_onsets_train = concatenate_sessions(
        args.subject, train_sessions, PATHS['practice_phase_metadata'], 
        PATHS['per_run_downsampled_to_TR'], PATHS['fmriprep_data'], logger
    )
    logger.info("\nLoading test data...")
    X_test, Y_test, run_onsets_test, level_onsets_test = concatenate_sessions(
        args.subject, test_sessions, PATHS['practice_phase_metadata'],
        PATHS['per_run_downsampled_to_TR'], PATHS['fmriprep_data'], logger
    )
    
    logger.info("\n" + "="*80)
    logger.info("FEATURE FILTERING")
    logger.info("="*80)
    X_train, X_test, valid_features_mask = filter_zero_variance_features(
        X_train, X_test, run_onsets_train, logger
    )
    
    logger.info("\n" + "="*80)
    logger.info("VOXEL FILTERING")
    logger.info("="*80)
    Y_train, Y_test, valid_voxels_mask = filter_zero_variance_voxels(
        Y_train, Y_test, run_onsets_train, logger
    )
    
    logger.info("\nPreprocessing training data...")
    X_train, Y_train = preprocess_data(X_train, Y_train, run_onsets_train, level_onsets_train, logger)
    logger.info("\nPreprocessing test data...")
    X_test, Y_test = preprocess_data(X_test, Y_test, run_onsets_test, level_onsets_test, logger)
    
    logger.info("\n" + "="*80)
    logger.info("MODEL FITTING AND EVALUATION")
    logger.info("="*80)
    pipeline, cv_scores, test_scores, best_alphas = fit_and_evaluate(
        X_train, Y_train, X_test, Y_test, run_onsets_train,
        PARAMETERS['encoding_model'], backend, logger
    )
    
    logger.info("\n" + "="*80)
    logger.info("SAVING RESULTS")
    logger.info("="*80)
    save_results(args.subject, train_sessions, test_sessions, pipeline, cv_scores, test_scores, 
                best_alphas, valid_voxels_mask, PATHS['models'], PATHS['cv_scores'], logger)
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
