"""
Data loading and preprocessing for the Mario encoding models.

Y is z-scored within runs using active-TR statistics before the full-length
series is returned.
"""
import json
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from voxelwise_tutorials.delayer import Delayer

from mario_encoding.utils.screen_annotations import ALL_METADATA_COLUMNS


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


def load_practice_features_with_names(subject, session, practice_runs, downsampled_path,
                                      metadata_columns=None):
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
    metadata_columns : list of str, optional
        Columns to return as aligned metadata rather than predictors. They are
        excluded from X and feature_names.

    Returns:
    --------
    X : array of shape (n_samples, n_features)
    feature_names : list of str
        Features in the order they appear in files
    extra_metadata : dict of str -> array, or None
        Requested metadata columns concatenated in run order.
    """
    X_list = []
    feature_names = None
    metadata_columns = list(metadata_columns) if metadata_columns else []
    metadata_chunks = {col: [] for col in metadata_columns}

    for run in practice_runs:
        filepath = (downsampled_path / f'sub-{subject:02d}' / f'ses-{session:03d}' /
                   f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv')
        if not filepath.exists():
            raise FileNotFoundError(f"Downsampled features not found: {filepath}")

        df = pd.read_csv(filepath, sep='\t')

        if metadata_columns:
            missing = set(metadata_columns) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Metadata columns missing in {filepath.name}: {sorted(missing)}. "
                    "Regenerate steps 2-4 with the Screen/Gameplay annotations."
                )
            for col in metadata_columns:
                metadata_chunks[col].append(df[col].to_numpy())

        # Exclude metadata columns, keep feature columns in FILE order
        feature_cols = [col for col in df.columns if col not in ALL_METADATA_COLUMNS]

        # First run defines the canonical feature order
        if feature_names is None:
            feature_names = feature_cols
        else:
            # Verify all runs have same features in same order
            if feature_cols != feature_names:
                raise ValueError(f"Feature column mismatch in run {run}. Expected {feature_names}, got {feature_cols}")

        X_list.append(df[feature_cols].values)

    X = np.vstack(X_list).astype('float32')
    extra_metadata = None
    if metadata_columns:
        extra_metadata = {
            col: np.concatenate(chunks) for col, chunks in metadata_chunks.items()
        }
        for col, values in extra_metadata.items():
            if len(values) != X.shape[0]:
                raise ValueError(
                    f"Metadata column {col} has {len(values)} rows, expected {X.shape[0]}")

    return X, feature_names, extra_metadata


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
                                 downsampled_path: Path, fmri_path: Path, pipeline: str, logger,
                                 metadata_columns=None):
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
    metadata_columns : list of str, optional
        Columns carried as aligned metadata and excluded from X.

    Returns:
    --------
    X : array of shape (n_samples, n_features)
    Y : array of shape (n_samples, n_grayordinates)
    metadata : dict
    feature_names : list of str
        Features in FILE order
    extra_metadata : dict of str -> array, or None
    """
    logger.info(f"Loading data for sub-{subject:02d} ses-{session:03d}")

    metadata = load_practice_metadata(subject, session, practice_metadata_path)
    logger.info(f"  Found {metadata['n_runs']} practice runs: {metadata['practice_runs']}")
    logger.info(f"  Total samples: {metadata['n_samples']}")
    logger.info(f"  Total levels: {metadata['n_levels']}")

    validate_session_files(subject, session, metadata, downsampled_path, fmri_path, pipeline)
    logger.info("  All required files validated")

    X, feature_names, extra_metadata = load_practice_features_with_names(
        subject, session, metadata['practice_runs'], downsampled_path,
        metadata_columns=metadata_columns)
    logger.info(f"  Loaded features: {X.shape}")
    logger.info(f"  Feature names (in file order): {len(feature_names)}")

    Y = load_practice_fmri(subject, session, metadata['practice_runs'], fmri_path, pipeline)
    logger.info(f"  Loaded fMRI: {Y.shape}")

    # Validate shapes
    if X.shape[0] != metadata['n_samples']:
        raise ValueError(f"Feature samples ({X.shape[0]}) != metadata n_samples ({metadata['n_samples']})")
    if Y.shape[0] != metadata['n_samples']:
        raise ValueError(f"fMRI samples ({Y.shape[0]}) != metadata n_samples ({metadata['n_samples']})")
    if extra_metadata is not None:
        for col, values in extra_metadata.items():
            if len(values) != metadata['n_samples']:
                raise ValueError(
                    f"Metadata column {col} has {len(values)} rows, "
                    f"expected {metadata['n_samples']}")

    return X, Y, metadata, feature_names, extra_metadata


def concatenate_sessions_with_names(subject: int, sessions: list, practice_metadata_path: Path,
                                    downsampled_path: Path, fmri_path: Path, pipeline: str, logger,
                                    metadata_columns=None):
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
    metadata_columns : list of str, optional
        Columns carried as aligned metadata and excluded from X.

    Returns:
    --------
    X : array of shape (n_samples_total, n_features)
    Y : array of shape (n_samples_total, n_grayordinates)
    run_onsets : array of int
    session_onsets : array of int
    level_onsets : array of int
    feature_names : list of str
        Features in FILE order from first session (source of truth)
    extra_metadata : dict of str -> array, or None
        Requested metadata columns concatenated in session and run order.
    """
    X_list = []
    Y_list = []
    run_onsets_list = [0]
    session_onsets_list = [0]
    level_onsets_list = [0]
    cumulative_samples = 0
    feature_names = None
    metadata_columns = list(metadata_columns) if metadata_columns else []
    metadata_chunks = {col: [] for col in metadata_columns}

    for session in sessions:
        X_sess, Y_sess, metadata, feat_names, extra_metadata = load_session_data_with_names(
            subject, session, practice_metadata_path, downsampled_path, fmri_path, pipeline, logger,
            metadata_columns=metadata_columns
        )
        if metadata_columns:
            if extra_metadata is None:
                raise ValueError(f"Session {session} returned no requested metadata")
            for col in metadata_columns:
                values = np.asarray(extra_metadata[col])
                if len(values) != X_sess.shape[0]:
                    raise ValueError(
                        f"Session {session} metadata {col} has {len(values)} rows, "
                        f"expected {X_sess.shape[0]}")
                metadata_chunks[col].append(values)
        
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
        # Include all onsets from this session (not [1:]) so session-boundary first-run
        # onsets are not lost. For the first session, run_onsets_sess[0]=0 is already
        # in run_onsets_list, so skip it; for all others include the full list.
        if cumulative_samples == 0:
            run_onsets_list.extend(run_onsets_sess[1:].tolist())
        else:
            run_onsets_list.extend(run_onsets_sess.tolist())
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

    extra_metadata = None
    if metadata_columns:
        extra_metadata = {
            col: np.concatenate(chunks) for col, chunks in metadata_chunks.items()
        }
        for col, values in extra_metadata.items():
            if len(values) != X.shape[0]:
                raise ValueError(
                    f"Concatenated metadata {col} has {len(values)} rows, "
                    f"expected {X.shape[0]}")
        logger.info(f"  Metadata columns: {sorted(extra_metadata)}")

    return X, Y, run_onsets, session_onsets, level_onsets, feature_names, extra_metadata


def concatenate_level_labels(subject, sessions, practice_metadata_path, level_labels_path, logger=None):
    """Concatenate per-TR level and background labels across sessions.

    Mirrors concatenate_sessions_with_names exactly (sessions in list order, runs in
    metadata['practice_runs'] order, per-run length = downsampled n_TRs), so the returned
    arrays align 1:1 with that function's concatenated X/Y. Produced by attach_level_labels.py.

    Returns
    -------
    level_ids : (n_samples_total,) object
        Per-TR level string ('wXlY'), '' for ITI/rest TRs.
    background : (n_samples_total,) object
        Per-TR 'blue'/'black' background class, '' for ITI/rest TRs.
    run_bounds : list of (start, end)
        Run intervals in the concatenated TR index (end exclusive).
    session_ids : (n_samples_total,) int
        Index into `sessions` for each TR (CV grouping for leave-one-session-out).
    """
    level_list, bg_list, session_ids_list, run_bounds = [], [], [], []
    cumulative = 0
    for s_idx, session in enumerate(sessions):
        metadata = load_practice_metadata(subject, session, practice_metadata_path)
        for run in metadata['practice_runs']:
            label_file = (Path(level_labels_path) / f'sub-{subject:02d}' / f'ses-{session:03d}' /
                          f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_level_labels.tsv')
            df = pd.read_csv(label_file, sep='\t', keep_default_na=False,
                             dtype={'level': str, 'background': str})
            n = len(df)
            run_bounds.append((cumulative, cumulative + n))
            cumulative += n
            level_list.append(df['level'].to_numpy())
            bg_list.append(df['background'].to_numpy())
            session_ids_list.append(np.full(n, s_idx, dtype=int))
    level_ids = np.concatenate(level_list)
    background = np.concatenate(bg_list)
    session_ids = np.concatenate(session_ids_list)
    if logger:
        logger.info(f'Concatenated level labels: {len(level_ids)} TRs, {len(run_bounds)} runs, '
                    f'{len(sessions)} sessions')
    return level_ids, background, run_bounds, session_ids


def zscore_runs_active(Y, run_onsets, active_mask, logger=None):
    """Z-score each run using mean and SD over its active TRs only.

    Statistics are estimated on active TRs and applied to every TR in the run.
    Raises if a run has no active TRs or zero active-TR variance.
    """
    Y = np.asarray(Y)
    active_mask = np.asarray(active_mask, dtype=bool)
    run_starts = np.asarray(run_onsets, dtype=np.int64)
    if active_mask.shape != (Y.shape[0],):
        raise ValueError(
            f"active_mask has length {len(active_mask)}, expected {Y.shape[0]}")
    if run_starts.ndim != 1 or run_starts.size == 0 or run_starts[0] != 0:
        raise ValueError(f"run_onsets must be 1-D and start at 0, got {run_onsets}")
    if np.any(np.diff(run_starts) <= 0):
        raise ValueError("run_onsets must be strictly increasing")

    run_stops = np.r_[run_starts[1:], Y.shape[0]]
    Y_out = np.empty_like(Y)
    for run_index, (start, stop) in enumerate(zip(run_starts, run_stops)):
        run_active = active_mask[start:stop]
        if not run_active.any():
            raise ValueError(f"Run {run_index + 1} has no active TRs")
        active_values = Y[start:stop][run_active]
        if not np.isfinite(active_values).all():
            raise ValueError(f"Run {run_index + 1} has non-finite active Y values")
        mean = active_values.mean(axis=0, dtype=np.float64)
        sd = active_values.std(axis=0, dtype=np.float64)
        if np.any(sd <= 0):
            raise ValueError(f"Run {run_index + 1} has zero active-TR variance")
        Y_out[start:stop] = ((Y[start:stop] - mean) / sd).astype(Y.dtype)

    if logger is not None:
        logger.info(
            f"  Y z-scored per run on active TRs: {int(active_mask.sum())}/"
            f"{len(active_mask)} TRs, {len(run_starts)} runs")
    return Y_out


def preprocess_data(X, Y, run_onsets, active_mask, logger):
    """Preprocess features and fMRI responses.

    Y is z-scored per run on active TRs and applied to the full run. X is left
    unscaled; NaNs in both are replaced by zero.
    """
    logger.info("Preprocessing data...")
    Y = zscore_runs_active(Y, run_onsets, active_mask, logger)

    # In-place NaN replacement avoids a Y-sized transient copy at peak memory.
    np.nan_to_num(X, copy=False)
    np.nan_to_num(Y, copy=False)

    logger.info(f"  Final X shape: {X.shape}, dtype: {X.dtype}")
    logger.info(f"  Final Y shape: {Y.shape}, dtype: {Y.dtype}")

    return X, Y


def verify_run_normalisation(Y, run_onsets, logger, label='Y', tolerance=1e-5):
    """Check that each run of Y is zero-mean and unit-SD per voxel."""
    run_starts = np.asarray(run_onsets, dtype=np.int64)
    run_stops = np.r_[run_starts[1:], Y.shape[0]]
    max_mean = 0.0
    max_sd = 0.0
    for start, stop in zip(run_starts, run_stops):
        run = Y[start:stop]
        max_mean = max(max_mean, float(np.max(np.abs(run.mean(axis=0)))))
        max_sd = max(max_sd, float(np.max(np.abs(run.std(axis=0) - 1.0))))

    if max_mean > tolerance or max_sd > tolerance:
        raise ValueError(
            f"{label} retained samples are not zero-mean/unit-SD per run: "
            f"max |mean|={max_mean:.3g}, max |SD-1|={max_sd:.3g}")
    logger.info(
        f"  {label} retained normalisation check: max |mean|={max_mean:.2e}, "
        f"max |SD-1|={max_sd:.2e}")


def compute_baseline_mask(X, run_onsets, logger=None, threshold=0.01):
    """
    Compute the legacy feature-activity TR mask without dropping any data.

    A TR is active if sum(|X[t]|) > threshold. ITI/baseline TRs (all features
    near zero) are marked inactive. This is kept for the old/new audit and
    descriptive diagnostics; it no longer controls fit-time retention.

    Parameters:
    -----------
    X : array of shape (n_samples, n_features)
    run_onsets : array of int (unused, kept for symmetry with filter_baseline_periods)
    logger : logging.Logger, optional
    threshold : float

    Returns:
    --------
    active_mask : boolean array of shape (n_samples,)
    """
    activity = np.abs(X).sum(axis=1)
    active_mask = activity > threshold
    if logger is not None:
        n_active = int(active_mask.sum())
        n_total = len(active_mask)
        logger.info(f"Legacy feature-activity mask: {n_active}/{n_total} active TRs "
                    f"({n_active / n_total * 100:.1f}%)")
    return active_mask


def apply_delays_per_run(X, run_onsets, delays):
    """
    Apply FIR delays independently within each run, preventing cross-run leakage.

    Each run's delayed block uses zero-padding for delays beyond the run start.
    Output column ordering matches voxelwise_tutorials.Delayer:
    [d1_features | d2_features | ... | dN_features] (delay-major within run).

    Parameters:
    -----------
    X : array of shape (n_samples, n_features)
        Mean-centered feature matrix spanning all runs (no ITI dropped).
    run_onsets : array of int
        Cumulative TR indices at run starts; run k spans
        [run_onsets[k], run_onsets[k+1]) or [run_onsets[-1], n_samples) for the last run.
    delays : list of int

    Returns:
    --------
    X_delayed : array of shape (n_samples, n_features * len(delays))
    """
    n_samples, n_features = X.shape
    X_delayed = np.zeros((n_samples, n_features * len(delays)), dtype=X.dtype)

    run_ends = list(run_onsets[1:]) + [n_samples]
    for run_start, run_end in zip(run_onsets, run_ends):
        delayer = Delayer(delays=delays)
        X_delayed[run_start:run_end] = delayer.fit_transform(X[run_start:run_end])

    return X_delayed


def drop_baseline_samples(X_delayed, Y, run_onsets, session_onsets, active_mask, logger=None):
    """
    Apply a boolean TR mask to delayed X and Y simultaneously; remap onsets.

    Run and session onsets are recomputed in the retained-sample index space using
    the per-sample cumulative kept count, so they remain valid fold boundaries
    for CV after the drop.

    Parameters:
    -----------
    X_delayed : array of shape (n_samples, n_features_delayed)
    Y : array of shape (n_samples, n_voxels)
    run_onsets : array of int (pre-drop)
    session_onsets : array of int (pre-drop)
    active_mask : boolean array of shape (n_samples,)
    logger : logging.Logger, optional

    Returns:
    --------
    X_active : array
    Y_active : array
    run_onsets_active : array of int
    session_onsets_active : array of int
    """
    n_samples = X_delayed.shape[0]
    cumulative_active = np.concatenate([[0], np.cumsum(active_mask.astype(np.int64))])

    X_active = X_delayed[active_mask]
    Y_active = Y[active_mask]

    run_onsets_active = cumulative_active[np.asarray(run_onsets)]
    session_onsets_active = cumulative_active[np.asarray(session_onsets)]

    if logger is not None:
        n_dropped = n_samples - int(active_mask.sum())
        logger.info(f"Dropped {n_dropped} non-gameplay TRs "
                    f"({n_dropped / n_samples * 100:.1f}%) post-delay")
        logger.info(f"  Retained TRs: {int(active_mask.sum())}")
        logger.info(f"  Active run_onsets ({len(run_onsets_active)}): "
                    f"{run_onsets_active[:5].tolist()}{'...' if len(run_onsets_active) > 5 else ''}")
        logger.info(f"  Active session_onsets ({len(session_onsets_active)}): "
                    f"{session_onsets_active.tolist()}")

    return X_active, Y_active, run_onsets_active, session_onsets_active


def filter_baseline_periods(X, Y, run_onsets, session_onsets, logger, threshold=0.01):
    """
    Remove ITI/baseline periods where all behavioral features are zero.

    Parameters:
    -----------
    X : array of shape (n_samples, n_features)
    Y : array of shape (n_samples, n_grayordinates)
    run_onsets : array of int
    session_onsets : array of int
    logger : logging.Logger
    threshold : float
        Activity threshold for identifying non-baseline TRs

    Returns:
    --------
    X_active : array
    Y_active : array
    run_onsets_active : array
    session_onsets_active : array
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

    # Recompute session_onsets in the filtered sample space using the per-sample mask.
    # session_onsets[i] is the pre-filter start index of session i; we count active
    # samples within each session's span to get post-filter positions.
    activity_all = np.abs(X).sum(axis=1) > threshold
    session_onsets_active_list = [0]
    cum_sess = 0
    for i in range(len(session_onsets)):
        sess_start = session_onsets[i]
        sess_end = int(session_onsets[i + 1]) if i + 1 < len(session_onsets) else len(X)
        cum_sess += int(activity_all[sess_start:sess_end].sum())
        if i < len(session_onsets) - 1:
            session_onsets_active_list.append(cum_sess)
    session_onsets_active = np.array(session_onsets_active_list)

    logger.info(f"Total TRs dropped: {total_dropped} ({total_dropped/len(X)*100:.1f}%)")
    logger.info(f"Active TRs kept: {len(X_active)} ({len(X_active)/len(X)*100:.1f}%)")

    return X_active, Y_active, run_onsets_active, session_onsets_active


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
