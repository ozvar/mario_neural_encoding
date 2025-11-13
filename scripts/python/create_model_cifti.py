"""
Utilities for loading and visualizing encoding model results.

This module provides functions to load fitted models, extract voxel masks,
create CIFTI visualization files, and inspect model performance.
"""
import numpy as np
import nibabel as nib
import pickle
from pathlib import Path
from mario_encoding.config import PATHS


def parse_model_filename(model_path):
    """
    Extract subject, train sessions, and test sessions from model filename.
    
    Parameters:
    -----------
    model_path : str or Path
        Path to model file (e.g., 'sub-01_train-ses-007-007_test-ses-014-014_model.pkl')
        
    Returns:
    --------
    dict with keys: subject, train_start, train_end, test_start, test_end
    """
    filepath = Path(model_path)
    filename = filepath.stem  # Remove .pkl extension
    
    parts = filename.split('_')
    
    # Parse subject
    subject = int(parts[0].split('-')[1])
    
    # Parse train sessions
    train_part = [p for p in parts if p.startswith('train-ses-')][0]
    train_sessions = train_part.replace('train-ses-', '').split('-')
    train_start = int(train_sessions[0])
    train_end = int(train_sessions[1])
    
    # Parse test sessions
    test_part = [p for p in parts if p.startswith('test-ses-')][0]
    test_sessions = test_part.replace('test-ses-', '').split('-')
    test_start = int(test_sessions[0])
    test_end = int(test_sessions[1])
    
    return {
        'subject': subject,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end
    }


def get_result_paths(subject, train_sessions, test_sessions, 
                     models_path=PATHS['models'], 
                     cv_scores_path=PATHS['cv_scores']):
    """
    Get paths to all result files for a given model.
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    models_path : Path
    cv_scores_path : Path
        
    Returns:
    --------
    dict with paths to: model, cv_scores, test_scores, best_alphas
    """
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    base_name = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    
    return {
        'model': models_path / f'{base_name}_model.pkl',
        'cv_scores': cv_scores_path / f'{base_name}_cv_scores.npy',
        'test_scores': cv_scores_path / f'{base_name}_test_scores.npy',
        'best_alphas': cv_scores_path / f'{base_name}_best_alphas.npy',
        'base_name': base_name
    }


def load_results(subject, train_sessions, test_sessions,
                 models_path=PATHS['models'],
                 cv_scores_path=PATHS['cv_scores']):
    """
    Load all results for a fitted model.
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    models_path : Path
    cv_scores_path : Path
        
    Returns:
    --------
    dict with keys: pipeline, cv_scores, test_scores, best_alphas
    """
    paths = get_result_paths(subject, train_sessions, test_sessions, 
                            models_path, cv_scores_path)
    
    # with open(paths['model'], 'rb') as f:
    #     pipeline = pickle.load(f)
    
    cv_scores = np.load(paths['cv_scores'])
    test_scores = np.load(paths['test_scores'])
    best_alphas = np.load(paths['best_alphas'])
    
    return {
        # 'pipeline': pipeline,
        'cv_scores': cv_scores,
        'test_scores': test_scores,
        'best_alphas': best_alphas,
        'paths': paths
    }


def reconstruct_voxel_mask(subject, train_sessions, 
                           practice_metadata_path=PATHS['practice_phase_metadata'],
                           fmriprep_path=PATHS['fmriprep_data']):
    """
    Reconstruct which voxels were kept during training by replicating filtering logic.
    
    This recreates the zero-variance voxel filtering that was applied during model fitting.
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    practice_metadata_path : Path
    fmriprep_path : Path
        
    Returns:
    --------
    valid_voxels_mask : boolean array of shape (91282,)
        True for voxels that were kept during training
    """
    import json
    # Remove this broken import - define functions inline instead
    
    # Load all training data
    Y_list = []
    run_onsets_list = [0]
    cumulative_samples = 0
    
    for session in train_sessions:
        # Load metadata
        metadata_path = practice_metadata_path / f'sub-{subject:02d}_ses-{session:03d}_practice_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        practice_runs = metadata['practice_runs']
        
        # Load fMRI for each run and concatenate
        for run in practice_runs:
            filepath = (fmriprep_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func' /
                       f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run}_space-fsLR_den-91k_bold.dtseries.nii')
            if not filepath.exists():
                raise FileNotFoundError(f"CIFTI file not found: {filepath}")
            cifti = nib.load(str(filepath))
            fmri_data = cifti.get_fdata()
            Y_list.append(fmri_data)
        
        # Track run onsets
        run_onsets_sess = np.array(metadata['run_onsets']) + cumulative_samples
        run_onsets_list.extend(run_onsets_sess[1:].tolist())
        cumulative_samples += metadata['n_samples']
    
    Y_train = np.vstack(Y_list).astype('float32')
    run_onsets_train = np.array(run_onsets_list)
    
    # Replicate the zero-variance filtering logic from fit_encoding_model.py
    run_splits_train = np.split(Y_train, run_onsets_train[1:])
    zero_var_in_any_run = np.zeros(Y_train.shape[1], dtype=bool)
    
    for run_data in run_splits_train:
        run_var = run_data.var(axis=0)
        zero_var_this_run = (run_var < 1e-10)
        zero_var_in_any_run |= zero_var_this_run
    
    valid_voxels_mask = ~zero_var_in_any_run
    
    print(f"Reconstructed voxel mask:")
    print(f"  Original voxels: {len(valid_voxels_mask)}")
    print(f"  Zero-variance voxels: {zero_var_in_any_run.sum()}")
    print(f"  Kept voxels: {valid_voxels_mask.sum()}")
    
    return valid_voxels_mask


def create_cifti_from_scores(scores, voxel_mask, template_path, output_path):
    """
    Create a CIFTI scalar file from model scores (e.g., R² values).
    
    Parameters:
    -----------
    scores : array of shape (n_kept_voxels,)
        Values for each kept voxel (e.g., test R²)
    voxel_mask : boolean array of shape (91282,)
        Mask indicating which voxels were kept
    template_path : str or Path
        Path to template CIFTI file for brain structure info
    output_path : str or Path
        Path to save output CIFTI file
        
    Returns:
    --------
    output_path : Path
    """
    from nibabel.cifti2 import Cifti2Header, Cifti2Image
    from nibabel.cifti2.cifti2_axes import ScalarAxis, BrainModelAxis
    # Load template to get brain structure info
    template = nib.load(str(template_path))
    # Create full array with NaNs for dropped voxels
    n_total_voxels = len(voxel_mask)
    scores_full = np.full(n_total_voxels, np.nan, dtype=np.float32)
    scores_full[voxel_mask] = scores
    # Create CIFTI with single scalar map
    data = scores_full[np.newaxis, :]  # Shape: (1, 91282)
    # Get the brain model axis from template (second axis, the spatial one)
    brain_model_axis = template.header.get_axis(1)
    # Create scalar axis (first axis, replaces time)
    scalar_axis = ScalarAxis(['R2'])
    # Create new header with scalar + brain model axes
    new_header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    # Create new CIFTI image
    new_cifti = Cifti2Image(data, header=new_header)
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_cifti.to_filename(str(output_path))
    
    print(f"Saved CIFTI to: {output_path}")
    return output_path


def create_all_cifti_maps(subject, train_sessions, test_sessions,
                          fmriprep_path=PATHS['fmriprep_data'],
                          figures_path=PATHS['figures']):
    """
    Create all CIFTI visualization maps for a fitted model.
    
    Creates:
    - Test R² map
    - CV R² map  
    - Best alpha map
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    fmriprep_path : Path
    figures_path : Path
        
    Returns:
    --------
    dict with paths to created CIFTI files
    """
    print(f"Creating CIFTI maps for sub-{subject:02d}")
    print(f"  Train sessions: {train_sessions}")
    print(f"  Test sessions: {test_sessions}")
    
    # Load results
    results = load_results(subject, train_sessions, test_sessions)
    base_name = results['paths']['base_name']
    

    # Load voxel mask directly instead of reconstructing
    print("\nLoading voxel mask...")
    voxel_mask_file = PATHS['cv_scores'] / f'{base_name}_voxel_mask.npy'
    if not voxel_mask_file.exists():
        raise FileNotFoundError(f"Voxel mask not found: {voxel_mask_file}")
    voxel_mask = np.load(voxel_mask_file)
    print(f"  Loaded mask: {voxel_mask.sum()} kept voxels")

    # Reconstruct voxel mask
    # print("\nReconstructing voxel mask...")
    # voxel_mask = reconstruct_voxel_mask(subject, train_sessions)
    
    # Get template CIFTI
    template_session = train_sessions[0]
    template_path = (fmriprep_path / f'sub-{subject:02d}' / f'ses-{template_session:03d}' / 'func' /
                    f'sub-{subject:02d}_ses-{template_session:03d}_task-mario_run-1_space-fsLR_den-91k_bold.dtseries.nii')
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template CIFTI not found: {template_path}")
    
    # Create output directory
    output_dir = figures_path / 'cifti_maps'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CIFTIs
    print("\nCreating CIFTI files...")
    
    test_r2_path = create_cifti_from_scores(
        results['test_scores'],
        voxel_mask,
        template_path,
        output_dir / f'{base_name}_test_r2.dscalar.nii'
    )
    
    cv_r2_path = create_cifti_from_scores(
        results['cv_scores'],
        voxel_mask,
        template_path,
        output_dir / f'{base_name}_cv_r2.dscalar.nii'
    )
    
    # Log best alphas for visualization
    log_alphas = np.log10(results['best_alphas'])
    alpha_path = create_cifti_from_scores(
        log_alphas,
        voxel_mask,
        template_path,
        output_dir / f'{base_name}_log10_best_alphas.dscalar.nii'
    )
    
    print("\n" + "="*80)
    print("CIFTI maps created successfully!")
    print(f"View with: wb_view {test_r2_path}")
    print("="*80)
    
    return {
        'test_r2': test_r2_path,
        'cv_r2': cv_r2_path,
        'best_alphas': alpha_path
    }


def print_model_summary(subject, train_sessions, test_sessions):
    """
    Print summary statistics for a fitted model.
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    """
    results = load_results(subject, train_sessions, test_sessions)
    
    cv_scores = results['cv_scores']
    test_scores = results['test_scores']
    best_alphas = results['best_alphas']
    
    print("="*80)
    print(f"MODEL SUMMARY: sub-{subject:02d}")
    print(f"Train sessions: {train_sessions}")
    print(f"Test sessions: {test_sessions}")
    print("="*80)
    
    print(f"\nVoxels: {len(cv_scores)}")
    
    print(f"\nCV R² distribution:")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Median: {np.median(cv_scores):.4f}")
    print(f"  Std: {cv_scores.std():.4f}")
    print(f"  Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
    print(f"  Positive: {(cv_scores > 0).sum()}/{len(cv_scores)} ({(cv_scores > 0).mean()*100:.1f}%)")
    print(f"  R² > 0.1: {(cv_scores > 0.1).sum()} ({(cv_scores > 0.1).mean()*100:.1f}%)")
    print(f"  R² > 0.3: {(cv_scores > 0.3).sum()} ({(cv_scores > 0.3).mean()*100:.1f}%)")
    
    print(f"\nTest R² distribution:")
    print(f"  Mean: {test_scores.mean():.4f}")
    print(f"  Median: {np.median(test_scores):.4f}")
    print(f"  Std: {test_scores.std():.4f}")
    print(f"  Range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")
    print(f"  Positive: {(test_scores > 0).sum()}/{len(test_scores)} ({(test_scores > 0).mean()*100:.1f}%)")
    
    print(f"\nCV-Test correlation: {np.corrcoef(cv_scores, test_scores)[0,1]:.3f}")
    
    print(f"\nBest alphas:")
    print(f"  Median: {np.median(best_alphas):.2e}")
    print(f"  Range: [{best_alphas.min():.2e}, {best_alphas.max():.2e}]")
    
    print(f"\nTop 10 voxels (by test R²):")
    top_10_idx = np.argsort(test_scores)[-10:][::-1]
    for i, idx in enumerate(top_10_idx, 1):
        print(f"  {i}. Voxel {idx}: CV={cv_scores[idx]:.3f}, Test={test_scores[idx]:.3f}, alpha={best_alphas[idx]:.2e}")


if __name__ == '__main__':
    subject = 1
    train_sessions = [7, 8, 9, 10, 12, 13, 14, 15, 16]
    test_sessions = [17]
    
    # Print summary
    print_model_summary(subject, train_sessions, test_sessions)
    # Create CIFTI maps
    print("\n")
    cifti_paths = create_all_cifti_maps(subject, train_sessions, test_sessions)
