"""
Changes from original:
1. load_practice_features_with_names() returns features in FILE order
2. concatenate_sessions_with_names() preserves FILE order
3. FEATURE_SPACES is only used for grouping, not ordering
"""
import json
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from voxelwise_tutorials.utils import zscore_runs


def get_fmri_filepath(subject: int, session: int, run: int, fmri_path: Path, pipeline: str) -> Path:
    """
    Construct the CIFTI filepath for a single run based on preprocessing pipeline.

    Parameters:
    -----------
    subject : int
    session : int
    run : int
    fmri_path : Path
    pipeline : str
        'fmriprep' or 'hcp'

    Returns:
    --------
    filepath : Path
    """
    if pipeline == 'fmriprep':
        return (fmri_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func' /
                f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run}_space-fsLR_den-91k_bold.dtseries.nii')
    elif pipeline == 'hcp':
        session_dir = f'{subject:02d}_{session:03d}'
        run_dir = f'tfMRI_MRO_RUN{run}_PA'
        return (fmri_path / session_dir / 'hcp' / session_dir /
                'MNINonLinear' / 'Results' / run_dir /
                f'{run_dir}_Atlas_MSMAll_hp0_clean.dtseries.nii')
    else:
        raise ValueError(f"Unknown pipeline: '{pipeline}'. Expected 'fmriprep' or 'hcp'.")


def get_template_cifti_path(subject: int, fmri_path: Path, pipeline: str) -> Path:
    """
    Get path to a template CIFTI file for brain structure information.

    Parameters:
    -----------
    subject : int
    fmri_path : Path
    pipeline : str
        'fmriprep' or 'hcp'

    Returns:
    --------
    template_path : Path
    """
    if pipeline == 'fmriprep':
        subject_dir = fmri_path / f'sub-{subject:02d}'
        cifti_files = sorted(subject_dir.rglob('*_space-fsLR_den-91k_bold.dtseries.nii'))
    elif pipeline == 'hcp':
        subject_glob = f'{subject:02d}_*'
        cifti_files = sorted(fmri_path.glob(
            f'{subject_glob}/hcp/{subject_glob}/MNINonLinear/Results/tfMRI_MRO_RUN*_PA/'
            f'*_Atlas_MSMAll_hp0_clean.dtseries.nii'
        ))
    else:
        raise ValueError(f"Unknown pipeline: '{pipeline}'. Expected 'fmriprep' or 'hcp'.")

    if not cifti_files:
        raise FileNotFoundError(
            f"No CIFTI files found for subject {subject} in {fmri_path} (pipeline='{pipeline}')"
        )
    return cifti_files[0]


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


def load_practice_features_with_names(subject, session, practice_runs, downsampled_path):
    """
    Load behavioral features for practice runs and return both data and column names.
    
    IMPORTANT: Returns features in the order they appear in the TSV files.
    This is the source of truth for feature ordering.
    
    Parameters:
    -----------
    subject : int
    session : int
    practice_runs : list of int
    downsampled_path : Path
        
    Returns:
    --------
    X : array of shape (n_samples, n_features)
    feature_names : list of str
        Features in the order they appear in files
    """
    X_list = []
    feature_names = None
    
    for run in practice_runs:
        filepath = (downsampled_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 
                   f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv')
        if not filepath.exists():
            raise FileNotFoundError(f"Downsampled features not found: {filepath}")
        
        df = pd.read_csv(filepath, sep='\t')
        
        # Exclude metadata columns, keep feature columns in FILE order
        feature_cols = [col for col in df.columns if col not in ['TR_index', 'TR_time', 'level', 'score', 'time', 'coins']]
        
        # First run defines the canonical feature order
        if feature_names is None:
            feature_names = feature_cols
        else:
            # Verify all runs have same features in same order
            if feature_cols != feature_names:
                raise ValueError(f"Feature column mismatch in run {run}. Expected {feature_names}, got {feature_cols}")
        
        X_list.append(df[feature_cols].values)
    
    X = np.vstack(X_list).astype('float32')
    
    return X, feature_names


def load_practice_fmri(subject: int, session: int, practice_runs: list, fmri_path: Path, pipeline: str) -> np.ndarray:
    """
    Load CIFTI fMRI data for practice runs and concatenate.
    
    Parameters:
    -----------
    subject : int
    session : int
    practice_runs : list of int
    fmri_path : Path
    pipeline : str
        'fmriprep' or 'hcp'
        
    Returns:
    --------
    Y : array of shape (n_samples, n_grayordinates)
    """
    Y_list = []
    
    for run in practice_runs:
        filepath = get_fmri_filepath(subject, session, run, fmri_path, pipeline)
        if not filepath.exists():
            raise FileNotFoundError(f"CIFTI file not found: {filepath}")
        
        cifti = nib.load(str(filepath))
        fmri_data = cifti.get_fdata()
        Y_list.append(fmri_data)
    
    Y = np.vstack(Y_list).astype('float32')
    
    return Y


def validate_session_files(subject: int, session: int, metadata: dict,
                           downsampled_path: Path, fmri_path: Path, pipeline: str) -> None:
    """
    Validate that all required files exist before loading.
    
    Parameters:
    -----------
    subject : int
    session : int
    metadata : dict
    downsampled_path : Path
    fmri_path : Path
    pipeline : str
        'fmriprep' or 'hcp'
    """
    practice_runs = metadata['practice_runs']
    
    # Check behavioral feature files
    for run in practice_runs:
        filepath = (downsampled_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 
                   f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv')
        if not filepath.exists():
            raise FileNotFoundError(f"Missing behavioral features: {filepath}")
    
    # Check fMRI CIFTI files
    for run in practice_runs:
        filepath = get_fmri_filepath(subject, session, run, fmri_path, pipeline)
        if not filepath.exists():
            raise FileNotFoundError(f"Missing CIFTI file: {filepath}")


def load_session_data_with_names(subject: int, session: int, practice_metadata_path: Path,
                                 downsampled_path: Path, fmri_path: Path, pipeline: str, logger):
    """
    Load all practice phase data for one session, returning feature names.
    
    Parameters:
    -----------
    subject : int
    session : int
    practice_metadata_path : Path
    downsampled_path : Path
    fmri_path : Path
    pipeline : str
        'fmriprep' or 'hcp'
    logger : logging.Logger
        
    Returns:
    --------
    X : array of shape (n_samples, n_features)
    Y : array of shape (n_samples, n_grayordinates)
    metadata : dict
    feature_names : list of str
        Features in FILE order
    """
    logger.info(f"Loading data for sub-{subject:02d} ses-{session:03d}")
    
    metadata = load_practice_metadata(subject, session, practice_metadata_path)
    logger.info(f"  Found {metadata['n_runs']} practice runs: {metadata['practice_runs']}")
    logger.info(f"  Total samples: {metadata['n_samples']}")
    logger.info(f"  Total levels: {metadata['n_levels']}")
    
    validate_session_files(subject, session, metadata, downsampled_path, fmri_path, pipeline)
    logger.info("  All required files validated")
    
    X, feature_names = load_practice_features_with_names(subject, session, metadata['practice_runs'], downsampled_path)
    logger.info(f"  Loaded features: {X.shape}")
    logger.info(f"  Feature names (in file order): {len(feature_names)}")
    
    Y = load_practice_fmri(subject, session, metadata['practice_runs'], fmri_path, pipeline)
    logger.info(f"  Loaded fMRI: {Y.shape}")
    
    # Validate shapes
    if X.shape[0] != metadata['n_samples']:
        raise ValueError(f"Feature samples ({X.shape[0]}) != metadata n_samples ({metadata['n_samples']})")
    if Y.shape[0] != metadata['n_samples']:
        raise ValueError(f"fMRI samples ({Y.shape[0]}) != metadata n_samples ({metadata['n_samples']})")
    
    return X, Y, metadata, feature_names


def concatenate_sessions_with_names(subject: int, sessions: list, practice_metadata_path: Path,
                                    downsampled_path: Path, fmri_path: Path, pipeline: str, logger):
    """
    Load and concatenate data from multiple sessions, preserving feature names.
    
    IMPORTANT: Feature order comes from the first session's files and is preserved.
    All subsequent sessions are verified to have the same feature order.
    
    Parameters:
    -----------
    subject : int
    sessions : list of int
    practice_metadata_path : Path
    downsampled_path : Path
    fmri_path : Path
    pipeline : str
        'fmriprep' or 'hcp'
    logger : logging.Logger
        
    Returns:
    --------
    X : array of shape (n_samples_total, n_features)
    Y : array of shape (n_samples_total, n_grayordinates)
    run_onsets : array of int
    session_onsets : array of int
    level_onsets : array of int
    feature_names : list of str
        Features in FILE order from first session (source of truth)
    """
    X_list = []
    Y_list = []
    run_onsets_list = [0]
    session_onsets_list = [0]
    level_onsets_list = [0]
    cumulative_samples = 0
    feature_names = None
    
    for session in sessions:
        X_sess, Y_sess, metadata, feat_names = load_session_data_with_names(
            subject, session, practice_metadata_path, downsampled_path, fmri_path, pipeline, logger
        )
        
        # First session defines canonical feature order
        if feature_names is None:
            feature_names = feat_names
            logger.info(f"  Canonical feature order established from session {session}")
        else:
            # Verify feature consistency across sessions
            if feat_names != feature_names:
                raise ValueError(
                    f"Feature mismatch in session {session}. "
                    f"Expected {len(feature_names)} features in order: {feature_names[:5]}..., "
                    f"but got {len(feat_names)} features: {feat_names[:5]}..."
                )
        
        X_list.append(X_sess)
        Y_list.append(Y_sess)
        
        # Track run and level onsets
        run_onsets_sess = np.array(metadata['run_onsets']) + cumulative_samples
        level_onsets_sess = np.array(metadata['level_onsets']) + cumulative_samples
        run_onsets_list.extend(run_onsets_sess[1:].tolist())
        level_onsets_list.extend(level_onsets_sess[1:].tolist())
        
        cumulative_samples += metadata['n_samples']
        session_onsets_list.append(cumulative_samples)
    
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    run_onsets = np.array(run_onsets_list)
    session_onsets = np.array(session_onsets_list[:-1])  # Remove last (beyond data)
    level_onsets = np.array(level_onsets_list)
    
    logger.info(f"Concatenated {len(sessions)} sessions:")
    logger.info(f"  Total samples: {X.shape[0]}")
    logger.info(f"  Total runs: {len(run_onsets)}")
    logger.info(f"  Total sessions: {len(session_onsets)}")
    logger.info(f"  Total levels: {len(level_onsets)}")
    
    return X, Y, run_onsets, session_onsets, level_onsets, feature_names


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
    
    logger.info("  Z-scoring fMRI within runs")
    Y = zscore_runs(Y, run_onsets)
    
    # Handle NaNs
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)
    
    logger.info(f"  Final X shape: {X.shape}, dtype: {X.dtype}")
    logger.info(f"  Final Y shape: {Y.shape}, dtype: {Y.dtype}")
    
    return X, Y


def filter_baseline_periods(X, Y, run_onsets, logger, threshold=0.01):
    """
    Remove ITI/baseline periods where all behavioral features are zero.
    
    Parameters:
    -----------
    X : array of shape (n_samples, n_features)
    Y : array of shape (n_samples, n_grayordinates)
    run_onsets : array of int
    logger : logging.Logger
    threshold : float
        Activity threshold for identifying non-baseline TRs
        
    Returns:
    --------
    X_active : array
    Y_active : array
    run_onsets_active : array
    """
    logger.info("Filtering baseline/ITI periods...")
    
    # Split by runs
    X_runs = np.split(X, run_onsets[1:])
    Y_runs = np.split(Y, run_onsets[1:])
    
    X_active_list = []
    Y_active_list = []
    run_onsets_active = [0]
    cumulative = 0
    total_dropped = 0
    
    for i, (X_run, Y_run) in enumerate(zip(X_runs, Y_runs)):
        # Identify active TRs (where features are nonzero)
        activity = np.abs(X_run).sum(axis=1)
        active_mask = activity > threshold
        
        n_dropped = (~active_mask).sum()
        total_dropped += n_dropped
        
        if n_dropped > 0:
            logger.info(f"  Run {i+1}: Kept {active_mask.sum()}/{len(active_mask)} TRs (dropped {n_dropped} ITI/baseline)")
        
        # Keep only active TRs
        X_active_list.append(X_run[active_mask])
        Y_active_list.append(Y_run[active_mask])
        
        cumulative += active_mask.sum()
        run_onsets_active.append(cumulative)
    
    X_active = np.vstack(X_active_list)
    Y_active = np.vstack(Y_active_list)
    run_onsets_active = np.array(run_onsets_active[:-1])
    
    logger.info(f"Total TRs dropped: {total_dropped} ({total_dropped/len(X)*100:.1f}%)")
    logger.info(f"Active TRs kept: {len(X_active)} ({len(X_active)/len(X)*100:.1f}%)")
    
    return X_active, Y_active, run_onsets_active


def select_and_validate_features(X, feature_names, feature_spaces_dict, logger):
    """
    Select features specified in config and validate they exist in loaded data.
    
    This is the authoritative feature selection step that happens BEFORE any
    filtering or modeling. Only features listed in FEATURE_SPACES will be kept.
    
    Parameters:
    -----------
    X : array of shape (n_samples, n_features_loaded)
        All features loaded from data files
    feature_names : list of str
        Names of all loaded features (in file order)
    feature_spaces_dict : dict
        From config.py - defines which features to use and their grouping
    logger : logging.Logger
        
    Returns:
    --------
    X_selected : array of shape (n_samples, n_features_selected)
        Only features specified in config
    feature_names_selected : list of str
        Names of selected features (in original file order)
    selection_mask : boolean array
        Mask indicating which loaded features were selected
        
    Raises:
    -------
    ValueError if any requested feature doesn't exist in loaded data
    """
    logger.info("")
    logger.info("="*80)
    logger.info("FEATURE SELECTION FROM CONFIG")
    logger.info("="*80)
    
    # Flatten all features requested in config
    requested_features = []
    for space_name in sorted(feature_spaces_dict.keys()):
        requested_features.extend(feature_spaces_dict[space_name])
    
    logger.info(f"Loaded from data: {len(feature_names)} features")
    logger.info(f"Requested in config: {len(requested_features)} features")
    logger.info(f"Feature spaces defined: {sorted(feature_spaces_dict.keys())}")
    
    # Check for duplicates in config
    if len(requested_features) != len(set(requested_features)):
        duplicates = [f for f in requested_features if requested_features.count(f) > 1]
        raise ValueError(f"Duplicate features in config: {set(duplicates)}")
    
    # Check for missing features
    missing_features = set(requested_features) - set(feature_names)
    if missing_features:
        logger.error(f"Features in config but NOT in data files: {sorted(missing_features)}")
        raise ValueError(
            f"Config requests {len(missing_features)} features not found in data. "
            f"Missing features: {sorted(missing_features)}"
        )
    
    # Check for extra features (informational only)
    extra_features = set(feature_names) - set(requested_features)
    if extra_features:
        logger.warning(f"Features in data but NOT in config (will be DROPPED): {len(extra_features)}")
        if len(extra_features) <= 10:
            logger.warning(f"  Dropped features: {sorted(extra_features)}")
        else:
            logger.warning(f"  Dropped features (first 10): {sorted(extra_features)[:10]}")
    
    # Create selection mask (preserves file order)
    selection_mask = np.array([feat in requested_features for feat in feature_names])
    
    # Select features
    X_selected = X[:, selection_mask]
    feature_names_selected = [feat for feat in feature_names if feat in requested_features]
    
    logger.info("")
    logger.info("Selection summary:")
    logger.info(f"  Loaded: {len(feature_names)}")
    logger.info(f"  Requested: {len(requested_features)}")
    logger.info(f"  Kept: {len(feature_names_selected)}")
    logger.info(f"  Dropped: {len(extra_features)}")
    
    # Log per-space breakdown
    logger.info("")
    logger.info("Per-space selection:")
    for space_name in sorted(feature_spaces_dict.keys()):
        space_feats = feature_spaces_dict[space_name]
        kept = [f for f in space_feats if f in feature_names_selected]
        logger.info(f"  {space_name}: {len(kept)}/{len(space_feats)} features")
        if len(kept) < len(space_feats):
            missing = set(space_feats) - set(kept)
            logger.warning(f"    Missing from data: {sorted(missing)}")
    
    logger.info("="*80)
    
    return X_selected, feature_names_selected, selection_mask


def filter_zero_variance_features(X_train, X_test, run_onsets_train, logger):
    """
    Filter features with zero variance across the entire training set.
    
    Parameters:
    -----------
    X_train : array of shape (n_samples_train, n_features)
    X_test : array of shape (n_samples_test, n_features)
    run_onsets_train : array of int (unused, kept for API consistency)
    logger : logging.Logger
        
    Returns:
    --------
    X_train_filtered : array
    X_test_filtered : array
    valid_features_mask : boolean array
    """
    logger.info("Identifying zero-variance features on training data...")
    
    train_feature_var = X_train.var(axis=0)
    valid_features_mask = train_feature_var > 1e-10
    
    n_features_original = X_train.shape[1]
    n_features_kept = valid_features_mask.sum()
    n_features_dropped = n_features_original - n_features_kept
    
    logger.info(f"  Original features: {n_features_original}")
    logger.info(f"  Zero-variance features: {n_features_dropped}")
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
