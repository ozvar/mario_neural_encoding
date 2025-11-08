"""
Downsample framewise behavioral features to TR for fMRI encoding model analysis.

This script takes pruned behavioral features at ~60fps and downsamples them to match
the fMRI sampling rate (TR = 1.49s). Binary features are averaged (proportion active),
continuous features are averaged within each TR window. Handles inter-trial intervals
and post-task baseline by filling gaps with zeros to match fMRI acquisition length.
"""
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from mario_encoding.config import PATHS, PARAMETERS


def identify_feature_types(df, metadata_cols=['TR_bin']):
    """
    Classify columns into binary vs continuous features.
    
    Binary features have only values in {0, 1, NaN} or are boolean dtype.
    Continuous features have other numeric values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    metadata_cols : list
        Columns to exclude from classification
        
    Returns:
    --------
    binary_cols : list
        Column names with only 0/1 values or boolean dtype
    continuous_cols : list
        Column names with other numeric values
    """
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    binary_cols = []
    continuous_cols = []
    
    for col in feature_cols:
        # Check if boolean dtype
        if df[col].dtype == 'bool':
            binary_cols.append(col)
        # Check if numeric dtype
        elif df[col].dtype in ['int64', 'float64', 'Int64', 'Float64']:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                binary_cols.append(col)
            else:
                continuous_cols.append(col)
    
    return binary_cols, continuous_cols


def create_TR_bins(df, tr):
    """
    Assign each frame to a TR bin based on frame_time_in_run.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with frame_time_in_run column
    tr : float
        Repetition time in seconds
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added TR_bin column (integer: 0, 1, 2, ...)
    """
    df = df.copy()
    df['TR_bin'] = (df['frame_time_in_run'] / tr).astype(int)
    return df


def downsample_binary_features(group, binary_cols, method='mean'):
    """
    Aggregate binary features within TR bin.
    
    Parameters:
    -----------
    group : pd.DataFrame
        Group of frames within one TR bin
    binary_cols : list
        Column names of binary features
    method : str
        'mean' for proportion active (default)
        'lanczos' placeholder for future implementation
        
    Returns:
    --------
    pd.Series with aggregated binary features
    """
    if method == 'mean':
        return group[binary_cols].mean()
    elif method == 'lanczos':
        raise NotImplementedError("Lanczos filtering not yet implemented")
    else:
        raise ValueError(f"Unknown method: {method}")


def downsample_continuous_features(group, continuous_cols, method='mean'):
    """
    Aggregate continuous features within TR bin.
    
    Parameters:
    -----------
    group : pd.DataFrame
        Group of frames within one TR bin
    continuous_cols : list
        Column names of continuous features
    method : str
        'mean' for average within TR window (default)
        
    Returns:
    --------
    pd.Series with aggregated continuous features
    """
    if method == 'mean':
        return group[continuous_cols].mean()
    else:
        raise ValueError(f"Unknown method: {method}")


def get_fmri_n_trs(subject, session, run, fmriprep_path=PATHS['fmriprep_data']):
    """
    Get number of TRs from CIFTI file.
    
    Parameters:
    -----------
    subject : int
    session : int
    run : int
    fmriprep_path : Path
        
    Returns:
    --------
    n_trs : int
    """
    cifti_path = (fmriprep_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func' /
                  f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run}_space-fsLR_den-91k_bold.dtseries.nii')
    if not cifti_path.exists():
        raise FileNotFoundError(f"CIFTI file not found: {cifti_path}")
    cifti = nib.load(str(cifti_path))
    return cifti.get_fdata().shape[0]


def downsample_run_to_TR(df, subject, session, run, tr, fmriprep_path, binary_method='mean', continuous_method='mean'):
    """
    Downsample a single run from framewise to TR sampling rate, matching fMRI length.
    
    This function:
    1. Gets actual fMRI n_TRs from CIFTI file
    2. Creates complete TR grid (0 to n_TRs-1)
    3. Aggregates frames into TRs where behavioral data exists
    4. Fills gaps (ITIs and post-task baseline) with zeros
    
    Parameters:
    -----------
    df : pd.DataFrame
        Framewise behavioral features with frame_time_in_run column
    subject : int
    session : int
    run : int
    tr : float
        Repetition time in seconds
    fmriprep_path : Path
        Path to fMRI data
    binary_method : str
        Aggregation method for binary features
    continuous_method : str
        Aggregation method for continuous features
        
    Returns:
    --------
    pd.DataFrame
        Downsampled features (one row per TR) with TR_index and TR_time columns
    """
    n_trs_fmri = get_fmri_n_trs(subject, session, run, fmriprep_path)
    df = create_TR_bins(df, tr)
    binary_cols, continuous_cols = identify_feature_types(df, metadata_cols=['TR_bin', 'frame_time_in_run'])
    all_feature_cols = binary_cols + continuous_cols
    tr_data_dict = {}
    for tr_bin, group in df.groupby('TR_bin'):
        if tr_bin >= n_trs_fmri:
            continue
        binary_agg = downsample_binary_features(group, binary_cols, method=binary_method)
        continuous_agg = downsample_continuous_features(group, continuous_cols, method=continuous_method)
        tr_data = pd.concat([binary_agg, continuous_agg])
        tr_data_dict[tr_bin] = tr_data
    complete_data = []
    for tr_idx in range(n_trs_fmri):
        if tr_idx in tr_data_dict:
            complete_data.append(tr_data_dict[tr_idx])
        else:
            baseline_row = pd.Series(0.0, index=all_feature_cols)
            complete_data.append(baseline_row)
    downsampled = pd.DataFrame(complete_data)
    downsampled.insert(0, 'TR_index', range(n_trs_fmri))
    downsampled.insert(1, 'TR_time', downsampled['TR_index'] * tr)
    return downsampled


def get_run_pruned_features_path(subject, session, run, per_run_pruned_path=PATHS['per_run_pruned_features']):
    """Get path for pruned features TSV (input)."""
    base = per_run_pruned_path / f'sub-{subject:02d}' / f'ses-{session:03d}'
    return base / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-pruned_features.tsv'


def find_run_pruned_feature_files(subject=None, session=None, run=None, per_run_pruned_path=PATHS['per_run_pruned_features']):
    """Find all pruned feature files matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'run-{run:02d}' if run else 'run-*'
    pattern = f'{sub_pattern}/{ses_pattern}/*{run_pattern}_desc-pruned_features.tsv'
    return list(per_run_pruned_path.glob(pattern))


def get_run_downsampled_path(subject, session, run, per_run_downsampled_path=PATHS['per_run_downsampled_to_TR']):
    """Get path for downsampled TSV (output)."""
    base = per_run_downsampled_path / f'sub-{subject:02d}' / f'ses-{session:03d}'
    base.mkdir(parents=True, exist_ok=True)
    return base / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv'


def save_run_downsampled_tsv(df, subject, session, run, per_run_downsampled_path=PATHS['per_run_downsampled_to_TR']):
    """Save downsampled DataFrame to TSV."""
    filepath = get_run_downsampled_path(subject, session, run, per_run_downsampled_path)
    df.to_csv(filepath, sep='\t', index=False)
    return filepath


def extract_subject_session_run_from_path(filepath):
    """Extract subject, session, run numbers from filepath."""
    filepath = Path(filepath)
    filename = filepath.stem
    parts = filename.split('_')
    subject = None
    session = None
    run = None
    for part in parts:
        if part.startswith('sub-'):
            subject = int(part.split('-')[1])
        elif part.startswith('ses-'):
            session = int(part.split('-')[1])
        elif part.startswith('run-'):
            run = int(part.split('-')[1])
    if subject is not None and session is not None and run is not None:
        return subject, session, run
    else:
        raise ValueError(f"Could not parse subject/session/run from filename: {filename}")


def process_run_downsampling(filepath, tr, frame_rate, fmriprep_path, binary_method='mean', 
                            continuous_method='mean', per_run_downsampled_path=PATHS['per_run_downsampled_to_TR']):
    """
    Process a single run file: downsample from framewise to TR.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to pruned features TSV
    tr : float
        Repetition time in seconds
    frame_rate : float
        Frame rate in fps (unused but kept for compatibility)
    fmriprep_path : Path
        Path to fMRI data directory
    binary_method : str
        Aggregation method for binary features
    continuous_method : str
        Aggregation method for continuous features
    per_run_downsampled_path : Path
        Output directory path
        
    Returns:
    --------
    dict with processing results (subject, session, run, status, metrics)
    """
    subject, session, run = extract_subject_session_run_from_path(filepath)
    print(f"  Processing run {run:02d}...")
    try:
        df = pd.read_csv(filepath, sep='\t')
        n_frames_input = len(df)
        downsampled = downsample_run_to_TR(
            df=df,
            subject=subject,
            session=session,
            run=run,
            tr=tr,
            fmriprep_path=fmriprep_path,
            binary_method=binary_method,
            continuous_method=continuous_method
        )
        n_TRs_output = len(downsampled)
        output_filepath = save_run_downsampled_tsv(downsampled, subject, session, run, per_run_downsampled_path)
        compression_ratio = n_frames_input / n_TRs_output
        return {
            'subject': subject,
            'session': session,
            'run': run,
            'filepath': output_filepath,
            'n_frames_input': n_frames_input,
            'n_TRs_output': n_TRs_output,
            'compression_ratio': compression_ratio,
            'status': 'success'
        }
    except Exception as e:
        return {
            'subject': subject,
            'session': session,
            'run': run,
            'filepath': None,
            'n_frames_input': 0,
            'n_TRs_output': 0,
            'compression_ratio': 0,
            'status': f'failed: {str(e)}'
        }


def process_all_run_downsampling(tr, frame_rate, binary_method='mean', continuous_method='mean',
                                 per_run_pruned_path=PATHS['per_run_pruned_features'],
                                 per_run_downsampled_path=PATHS['per_run_downsampled_to_TR'],
                                 fmriprep_path=PATHS['fmriprep_data']):
    """
    Process all run-level pruned feature files: downsample to TR.
    
    Parameters:
    -----------
    tr : float
        Repetition time in seconds
    frame_rate : float
        Frame rate in fps
    binary_method : str
        Aggregation method for binary features
    continuous_method : str
        Aggregation method for continuous features
    per_run_pruned_path : Path
        Input directory path
    per_run_downsampled_path : Path
        Output directory path
    fmriprep_path : Path
        Path to fMRI data directory
        
    Returns:
    --------
    pd.DataFrame with processing results for all runs
    """
    pruned_files = find_run_pruned_feature_files(per_run_pruned_path=per_run_pruned_path)
    if not pruned_files:
        raise ValueError(f"No pruned feature files found in {per_run_pruned_path}")
    print(f"Found {len(pruned_files)} run files to process")
    print()
    results = []
    current_session = None
    for filepath in sorted(pruned_files):
        subject, session, run = extract_subject_session_run_from_path(filepath)
        if (subject, session) != current_session:
            current_session = (subject, session)
            print(f"Processing sub-{subject:02d}_ses-{session:03d}...")
        result = process_run_downsampling(
            filepath=filepath,
            tr=tr,
            frame_rate=frame_rate,
            fmriprep_path=fmriprep_path,
            binary_method=binary_method,
            continuous_method=continuous_method,
            per_run_downsampled_path=per_run_downsampled_path
        )
        results.append(result)
        if result['status'] == 'success':
            print(f"    Saved {result['n_TRs_output']} TRs ({result['compression_ratio']:.1f}x compression) to {Path(result['filepath']).name}")
        else:
            print(f"    Failed: {result['status']}")
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Downsampling behavioral features to TR...")
    print(f"Input path: {PATHS['per_run_pruned_features']}")
    print(f"Output path: {PATHS['per_run_downsampled_to_TR']}")
    print(f"TR: {PARAMETERS['TR']}s")
    print(f"Frame rate: {PARAMETERS['frame_rate']} fps")
    print(f"Frames per TR: {PARAMETERS['frame_rate'] * PARAMETERS['TR']:.1f}")
    print()
    
    results = process_all_run_downsampling(
        tr=PARAMETERS['TR'],
        frame_rate=PARAMETERS['frame_rate'],
        binary_method='mean',
        continuous_method='mean'
    )
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Total runs processed: {len(results)}")
    print(f"Successful: {(results['status'] == 'success').sum()}")
    print(f"Failed: {(results['status'] != 'success').sum()}")
    
    if (results['status'] == 'success').any():
        successful = results[results['status'] == 'success']
        print(f"\nCompression statistics:")
        print(f"  Average frames per run: {successful['n_frames_input'].mean():.0f}")
        print(f"  Average TRs per run: {successful['n_TRs_output'].mean():.0f}")
        print(f"  Average compression ratio: {successful['compression_ratio'].mean():.1f}x")
    
    if (results['status'] != 'success').any():
        print("\nFailed runs:")
        print(results[results['status'] != 'success'][['subject', 'session', 'run', 'status']])
