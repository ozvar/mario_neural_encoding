"""
Create CIFTI maps for variance partitioning models.

Adapted version of create_model_cifti.py that works with variance partitioning
output structure (which only has test_scores.npy, not cv_scores or best_alphas).
"""
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from nibabel.cifti2 import Cifti2Header, Cifti2Image
from nibabel.cifti2.cifti2_axes import ScalarAxis

from mario_encoding.config import PATHS


def get_template_cifti_path(subject, fmriprep_path=PATHS['fmriprep_data']):
    """Get path to a template CIFTI file for brain structure information."""
    subject_dir = fmriprep_path / f'sub-{subject:02d}'
    cifti_files = sorted(subject_dir.rglob('*_space-fsLR_den-91k_bold.dtseries.nii'))
    
    if not cifti_files:
        raise FileNotFoundError(f"No CIFTI files found for subject {subject}")
    
    return cifti_files[0]


def create_cifti_from_scores(scores, voxel_mask, template_path, output_path, 
                             map_name='R2'):
    """
    Create a CIFTI scalar file from model scores.
    
    Parameters:
    -----------
    scores : array of shape (n_kept_voxels,)
    voxel_mask : boolean array of shape (91282,)
    template_path : Path
    output_path : Path
    map_name : str
    
    Returns:
    --------
    output_path : Path
    """
    # Load template
    template = nib.load(str(template_path))
    
    # Create full array with NaNs for dropped voxels
    n_total_voxels = len(voxel_mask)
    scores_full = np.full(n_total_voxels, np.nan, dtype=np.float32)
    scores_full[voxel_mask] = scores
    
    # Create CIFTI data (1, n_grayordinates)
    data = scores_full[np.newaxis, :]
    
    # Get brain model axis from template
    brain_model_axis = template.header.get_axis(1)
    
    # Create scalar axis
    scalar_axis = ScalarAxis([map_name])
    
    # Create new header and CIFTI
    new_header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    new_cifti = Cifti2Image(data, header=new_header)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_cifti.to_filename(str(output_path))
    
    print(f"Saved CIFTI to: {output_path}")
    
    return output_path


def create_vp_model_ciftis(subject, train_sessions, test_sessions,
                           fmriprep_path=PATHS['fmriprep_data'],
                           vp_results_path=None,
                           figures_path=PATHS['figures']):
    """
    Create CIFTI maps for variance partitioning full model performance.
    
    Creates test_scores CIFTI for the full model (equivalent to regular encoding model).
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    fmriprep_path : Path
    vp_results_path : Path or None
    figures_path : Path
        
    Returns:
    --------
    dict : Path to created CIFTI file
    """
    print("="*80)
    print("CREATING VARIANCE PARTITIONING MODEL CIFTI")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Train sessions: {train_sessions}")
    print(f"Test sessions: {test_sessions}")
    print()
    
    # Construct variance partitioning directory
    if vp_results_path is None:
        vp_results_path = PATHS.get('variance_partitioning',
                                     PATHS['models'].parent / 'variance_partitioning')
    
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dir_name = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    vp_output_dir = vp_results_path / dir_name
    
    if not vp_output_dir.exists():
        raise FileNotFoundError(f"Variance partitioning results not found: {vp_output_dir}")
    
    print(f"Loading from: {vp_output_dir}")
    print()
    
    # Load test scores
    test_scores_file = vp_output_dir / 'test_scores.npy'
    if not test_scores_file.exists():
        raise FileNotFoundError(f"test_scores.npy not found: {test_scores_file}")
    
    test_scores = np.load(test_scores_file)
    print(f"Loaded test scores: {test_scores.shape}")
    
    # Load voxel mask
    voxel_mask_file = vp_output_dir / 'valid_voxels_mask.npy'
    if not voxel_mask_file.exists():
        raise FileNotFoundError(f"Voxel mask not found: {voxel_mask_file}")
    
    voxel_mask = np.load(voxel_mask_file)
    print(f"Loaded voxel mask: {voxel_mask.shape}, kept: {voxel_mask.sum()}")
    print()
    
    # Get template CIFTI
    print("Getting template CIFTI...")
    template_path = get_template_cifti_path(subject, fmriprep_path)
    print(f"  Using: {template_path}")
    print()
    
    # Create output directory
    cifti_output_dir = figures_path / 'model_ciftis' / dir_name
    cifti_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving CIFTIs to: {cifti_output_dir}")
    print()
    
    # Create test R² CIFTI
    print("Creating test R² CIFTI...")
    test_r2_path = cifti_output_dir / f'{dir_name}_test_r2.dscalar.nii'
    create_cifti_from_scores(
        test_scores, voxel_mask, template_path, test_r2_path,
        map_name='test_R2'
    )
    print()
    
    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTest R² distribution:")
    print(f"  Mean: {test_scores.mean():.4f}")
    print(f"  Median: {np.median(test_scores):.4f}")
    print(f"  Std: {test_scores.std():.4f}")
    print(f"  Range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")
    print(f"  Positive: {(test_scores > 0).sum()}/{len(test_scores)} ({(test_scores > 0).mean()*100:.1f}%)")
    print(f"  R² > 0.1: {(test_scores > 0.1).sum()} ({(test_scores > 0.1).mean()*100:.1f}%)")
    print(f"  R² > 0.3: {(test_scores > 0.3).sum()} ({(test_scores > 0.3).mean()*100:.1f}%)")
    
    print()
    print("="*80)
    print("VISUALIZATION COMMAND")
    print("="*80)
    print(f"\nwb_view {test_r2_path}")
    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)
    
    return {'test_r2': test_r2_path}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create CIFTI maps for variance partitioning full model'
    )
    parser.add_argument('--subject', type=int, required=True, help='Subject number')
    parser.add_argument('--train-sessions', type=int, nargs='+', required=True,
                       help='Training session numbers (e.g., 6 7 8 9)')
    parser.add_argument('--test-sessions', type=int, nargs='+', required=True,
                       help='Test session numbers (e.g., 20 21)')
    
    args = parser.parse_args()
    
    # Create CIFTIs
    created_files = create_vp_model_ciftis(
        args.subject, args.train_sessions, args.test_sessions
    )
    
    return created_files


if __name__ == '__main__':
    main()
