"""
Create CIFTI maps for variance partitioning models.

Adapted version of create_model_cifti.py that works with variance partitioning
output structure (loads from R2_scores.npz and significance results).
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
    template = nib.load(str(template_path))
    n_total_voxels = len(voxel_mask)
    scores_full = np.full(n_total_voxels, np.nan, dtype=np.float32)
    scores_full[voxel_mask] = scores
    data = scores_full[np.newaxis, :]
    brain_model_axis = template.header.get_axis(1)
    scalar_axis = ScalarAxis([map_name])
    new_header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    new_cifti = Cifti2Image(data, header=new_header)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_cifti.to_filename(str(output_path))
    print(f"Saved CIFTI to: {output_path}")
    return output_path


def create_vp_model_ciftis(subject, train_sessions, test_sessions, experiment_id,
                           fmriprep_path=PATHS['fmriprep_data'],
                           vp_results_path=None,
                           figures_path=PATHS['figures']):
    """
    Create CIFTI maps for variance partitioning full model performance.
    
    Creates CIFTIs for:
    - Test R2 scores
    - Uncorrected p-values
    - FDR-corrected q-values
    - Significance masks
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    experiment_id : str
        Timestamp-based experiment identifier
    fmriprep_path : Path
    vp_results_path : Path or None
    figures_path : Path
        
    Returns:
    --------
    dict : Paths to created CIFTI files
    """
    print("="*80)
    print("CREATING VARIANCE PARTITIONING MODEL CIFTI")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Train sessions: {train_sessions}")
    print(f"Test sessions: {test_sessions}")
    print(f"Experiment ID: {experiment_id}")
    print()
    
    # Construct variance partitioning directory with experiment ID
    if vp_results_path is None:
        vp_results_path = PATHS.get('variance_partitioning',
                                     PATHS['models'].parent / 'variance_partitioning')
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    vp_output_dir = vp_results_path / dataset_dir / experiment_id
    
    if not vp_output_dir.exists():
        raise FileNotFoundError(f"Variance partitioning results not found: {vp_output_dir}")
    print(f"Loading from: {vp_output_dir}")
    print()
    
    # Load R2 scores from compressed npz file
    r2_scores_file = vp_output_dir / 'R2_scores.npz'
    if not r2_scores_file.exists():
        raise FileNotFoundError(f"R2_scores.npz not found: {r2_scores_file}")
    r2_data = np.load(r2_scores_file)
    test_scores = r2_data['R2_full']
    print(f"Loaded test R2 scores: {test_scores.shape}")
    print(f"  Available keys in R2_scores.npz: {list(r2_data.keys())}")
    print()
    
    # Load voxel mask
    voxel_mask_file = vp_output_dir / 'valid_voxels_mask.npy'
    if not voxel_mask_file.exists():
        raise FileNotFoundError(f"Voxel mask not found: {voxel_mask_file}")
    voxel_mask = np.load(voxel_mask_file)
    print(f"Loaded voxel mask: {voxel_mask.shape}, kept: {voxel_mask.sum()}")
    print()
    
    # Load significance results
    p_uncorrected_file = vp_output_dir / 'p_values_uncorrected.npy'
    p_fdr_file = vp_output_dir / 'p_values_fdr.npy'
    sig_uncorrected_file = vp_output_dir / 'significant_uncorrected_p0.05.npy'
    sig_fdr_file = vp_output_dir / 'significant_fdr_q0.05.npy'
    significance_available = all([
        p_uncorrected_file.exists(),
        p_fdr_file.exists(),
        sig_uncorrected_file.exists(),
        sig_fdr_file.exists()
    ])
    if significance_available:
        p_uncorrected = np.load(p_uncorrected_file)
        p_fdr = np.load(p_fdr_file)
        sig_uncorrected = np.load(sig_uncorrected_file)
        sig_fdr = np.load(sig_fdr_file)
        print("Loaded significance results:")
        print(f"  Uncorrected p-values: {p_uncorrected.shape}")
        print(f"  FDR q-values: {p_fdr.shape}")
        print(f"  Significant (p < 0.05): {sig_uncorrected.sum()}/{len(sig_uncorrected)} ({sig_uncorrected.mean()*100:.1f}%)")
        print(f"  Significant (q < 0.05): {sig_fdr.sum()}/{len(sig_fdr)} ({sig_fdr.mean()*100:.1f}%)")
        print()
    else:
        print("Warning: Significance results not found. Skipping p-value CIFTIs.")
        print("  Run compute_vp_significance.py first to generate significance results.")
        print()
    
    # Get template CIFTI
    print("Getting template CIFTI...")
    template_path = get_template_cifti_path(subject, fmriprep_path)
    print(f"  Using: {template_path}")
    print()
    
    # Create output directory with experiment ID
    cifti_output_dir = figures_path / 'model_ciftis' / dataset_dir / experiment_id
    cifti_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving CIFTIs to: {cifti_output_dir}")
    print()
    
    created_files = {}
    
    # Create test R2 CIFTI
    print("Creating test R2 CIFTI...")
    test_r2_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_test_r2.dscalar.nii'
    create_cifti_from_scores(
        test_scores, voxel_mask, template_path, test_r2_path,
        map_name='test_R2'
    )
    created_files['test_r2'] = test_r2_path
    print()
    
    # Create significance CIFTIs if available
    if significance_available:
        print("Creating uncorrected p-value CIFTI...")
        p_uncorrected_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_p_uncorrected.dscalar.nii'
        create_cifti_from_scores(
            p_uncorrected, voxel_mask, template_path, p_uncorrected_path,
            map_name='p_uncorrected'
        )
        created_files['p_uncorrected'] = p_uncorrected_path
        print()
        print("Creating FDR q-value CIFTI...")
        p_fdr_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_q_fdr.dscalar.nii'
        create_cifti_from_scores(
            p_fdr, voxel_mask, template_path, p_fdr_path,
            map_name='q_fdr'
        )
        created_files['q_fdr'] = p_fdr_path
        print()
        print("Creating uncorrected significance mask CIFTI...")
        sig_uncorrected_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_sig_p0.05.dscalar.nii'
        create_cifti_from_scores(
            sig_uncorrected.astype(np.float32), voxel_mask, template_path, sig_uncorrected_path,
            map_name='sig_p<0.05'
        )
        created_files['sig_uncorrected'] = sig_uncorrected_path
        print()
        print("Creating FDR significance mask CIFTI...")
        sig_fdr_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_sig_q0.05.dscalar.nii'
        create_cifti_from_scores(
            sig_fdr.astype(np.float32), voxel_mask, template_path, sig_fdr_path,
            map_name='sig_q<0.05'
        )
        created_files['sig_fdr'] = sig_fdr_path
        print()
    
    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTest R2 distribution:")
    print(f"  Mean: {test_scores.mean():.4f}")
    print(f"  Median: {np.median(test_scores):.4f}")
    print(f"  Std: {test_scores.std():.4f}")
    print(f"  Range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")
    print(f"  Positive: {(test_scores > 0).sum()}/{len(test_scores)} ({(test_scores > 0).mean()*100:.1f}%)")
    print(f"  R2 > 0.1: {(test_scores > 0.1).sum()} ({(test_scores > 0.1).mean()*100:.1f}%)")
    print(f"  R2 > 0.3: {(test_scores > 0.3).sum()} ({(test_scores > 0.3).mean()*100:.1f}%)")
    if significance_available:
        print(f"\nSignificance:")
        print(f"  Uncorrected (p < 0.05): {sig_uncorrected.sum()} ({sig_uncorrected.mean()*100:.1f}%)")
        print(f"  FDR (q < 0.05): {sig_fdr.sum()} ({sig_fdr.mean()*100:.1f}%)")
    print(f"\nCreated {len(created_files)} CIFTI files:")
    for key, path in created_files.items():
        print(f"  {key}: {path.name}")
    print()
    print("="*80)
    print("VISUALIZATION COMMAND")
    print("="*80)
    print(f"\nwb_view {test_r2_path}")
    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)
    return created_files


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
    parser.add_argument('--experiment-id', type=str, required=True,
                       help='Experiment ID (timestamp format: YYYYmmdd_HHMMSS)')
    args = parser.parse_args()
    
    # Create CIFTIs
    created_files = create_vp_model_ciftis(
        args.subject, args.train_sessions, args.test_sessions, args.experiment_id
    )
    return created_files


if __name__ == '__main__':
    main()
