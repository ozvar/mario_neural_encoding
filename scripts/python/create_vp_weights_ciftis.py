"""
Create CIFTI visualization maps for extracted feature space weights.

This script loads extracted weights from extract_variance_partitioning_weights.py
and creates CIFTI brain maps for visualizing weight patterns across voxels.
"""
import argparse
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from nibabel.cifti2 import Cifti2Header, Cifti2Image
from nibabel.cifti2.cifti2_axes import ScalarAxis

from mario_encoding.config import PATHS


def load_extracted_weights(output_dir, feature_space, threshold_suffix='unique_thresh0p010', 
                          logger_func=print):
    """
    Load extracted feature space weights and metadata.
    
    Parameters:
    -----------
    output_dir : Path
        Variance partitioning results directory
    feature_space : str
        Name of feature space (e.g., 'activity', 'motor')
    threshold_suffix : str
        Suffix indicating threshold type: 'unique_thresh0p010', 'full_thresh0p300', or 'all'
    logger_func : callable
        
    Returns:
    --------
    weights_selective : array of shape (n_features, n_selective_voxels)
    selective_mask : boolean array of shape (n_voxels,)
    valid_voxels_mask : boolean array of shape (91282,)
    metadata : dict
    """
    logger_func(f"Loading {feature_space} weights from: {output_dir}")
    logger_func(f"  Threshold suffix: {threshold_suffix}")
    
    # Load selective weights
    weights_file = output_dir / f'{feature_space}_weights_{threshold_suffix}.npy'
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights not found: {weights_file}")
    weights_selective = np.load(weights_file)
    
    # Load selective mask
    mask_file = output_dir / f'{feature_space}_selective_mask_{threshold_suffix}.npy'
    if not mask_file.exists():
        raise FileNotFoundError(f"Selective mask not found: {mask_file}")
    selective_mask = np.load(mask_file)
    
    # Load valid voxels mask (from variance partitioning)
    valid_voxels_file = output_dir / 'valid_voxels_mask.npy'
    if not valid_voxels_file.exists():
        raise FileNotFoundError(f"Valid voxels mask not found: {valid_voxels_file}")
    valid_voxels_mask = np.load(valid_voxels_file)
    
    # Load metadata
    metadata_file = output_dir / f'{feature_space}_weights_metadata_{threshold_suffix}.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger_func(f"  Weights shape: {weights_selective.shape}")
    logger_func(f"  Features: {metadata['n_features']}")
    logger_func(f"  Selective voxels: {metadata['n_selective_voxels']}")
    logger_func(f"  Total voxels (after filtering): {len(selective_mask)}")
    logger_func(f"  Total grayordinates: {len(valid_voxels_mask)}")
    logger_func(f"  Threshold type: {metadata.get('threshold_type', 'unique')}")
    logger_func(f"  Threshold value: {metadata.get('threshold_value', 0.01)}")
    
    return weights_selective, selective_mask, valid_voxels_mask, metadata


def get_template_cifti_path(subject, fmriprep_path=PATHS['fmriprep_data']):
    """
    Get path to a template CIFTI file for brain structure information.
    
    Parameters:
    -----------
    subject : int
    fmriprep_path : Path
        
    Returns:
    --------
    template_path : Path
    """
    subject_dir = fmriprep_path / f'sub-{subject:02d}'
    
    # Find first available CIFTI file
    cifti_files = sorted(subject_dir.rglob('*_space-fsLR_den-91k_bold.dtseries.nii'))
    
    if not cifti_files:
        raise FileNotFoundError(f"No CIFTI files found for subject {subject}")
    
    return cifti_files[0]


def create_weight_cifti(weights_selective, selective_mask, valid_voxels_mask,
                       template_path, output_path, map_names, logger_func=print):
    """
    Create multi-map CIFTI with one map per feature's weights.
    
    Parameters:
    -----------
    weights_selective : array of shape (n_features, n_selective_voxels)
    selective_mask : boolean array of shape (n_voxels_after_filtering,)
    valid_voxels_mask : boolean array of shape (91282,)
    template_path : Path
    output_path : Path
    map_names : list of str
        Name for each feature
    logger_func : callable
        
    Returns:
    --------
    output_path : Path
    """
    n_features, n_selective = weights_selective.shape
    n_voxels_filtered = len(selective_mask)
    n_total_voxels = len(valid_voxels_mask)
    
    logger_func(f"Creating weight CIFTI with {n_features} maps...")
    
    # Load template
    template = nib.load(str(template_path))
    brain_model_axis = template.header.get_axis(1)
    
    # Initialize data array (n_features, 91282)
    data = np.full((n_features, n_total_voxels), np.nan, dtype=np.float32)
    
    # For each feature, expand weights to full brain
    for feat_idx in range(n_features):
        # Create array for filtered voxels
        weights_filtered = np.full(n_voxels_filtered, np.nan, dtype=np.float32)
        weights_filtered[selective_mask] = weights_selective[feat_idx, :]
        
        # Expand to full 91282 grayordinates
        data[feat_idx, valid_voxels_mask] = weights_filtered
    
    # Create scalar axis with feature names
    scalar_axis = ScalarAxis(map_names)
    
    # Create CIFTI
    new_header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    new_cifti = Cifti2Image(data, header=new_header)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_cifti.to_filename(str(output_path))
    
    logger_func(f"  Saved: {output_path.name}")
    
    return output_path


def create_summary_weight_ciftis(weights_selective, selective_mask, valid_voxels_mask,
                                 template_path, output_dir, feature_space, logger_func=print):
    """
    Create summary weight maps (L1 norm, L2 norm, max absolute).
    
    Parameters:
    -----------
    weights_selective : array of shape (n_features, n_selective_voxels)
    selective_mask : boolean array
    valid_voxels_mask : boolean array
    template_path : Path
    output_dir : Path
    feature_space : str
    logger_func : callable
        
    Returns:
    --------
    dict : Paths to created files
    """
    logger_func("Creating summary weight maps...")
    
    n_voxels_filtered = len(selective_mask)
    n_total_voxels = len(valid_voxels_mask)
    
    # Load template
    template = nib.load(str(template_path))
    brain_model_axis = template.header.get_axis(1)
    
    created_files = {}
    
    # Compute summaries
    summaries = {
        'L1_norm': np.abs(weights_selective).sum(axis=0),
        'L2_norm': np.sqrt((weights_selective**2).sum(axis=0)),
        'max_abs': np.abs(weights_selective).max(axis=0)
    }
    
    for summary_name, summary_values in summaries.items():
        # Expand to full brain
        summary_full = np.full(n_total_voxels, np.nan, dtype=np.float32)
        summary_filtered = np.full(n_voxels_filtered, np.nan, dtype=np.float32)
        summary_filtered[selective_mask] = summary_values
        summary_full[valid_voxels_mask] = summary_filtered
        
        # Create CIFTI
        data = summary_full[np.newaxis, :]
        scalar_axis = ScalarAxis([f'{feature_space}_{summary_name}'])
        new_header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))
        new_cifti = Cifti2Image(data, header=new_header)
        
        # Save
        output_path = output_dir / f'{feature_space}_weights_{summary_name}.dscalar.nii'
        new_cifti.to_filename(str(output_path))
        logger_func(f"  Saved: {output_path.name}")
        created_files[summary_name] = output_path
    
    return created_files


def create_all_weight_ciftis(subject, train_sessions, test_sessions, feature_space,
                             threshold_suffix='unique_thresh0p010',
                             fmriprep_path=PATHS['fmriprep_data'],
                             vp_results_path=None,
                             figures_path=PATHS['figures']):
    """
    Create all CIFTI weight maps for extracted feature space weights.
    
    Creates:
    - Per-feature weight maps (one map per feature)
    - Summary maps (L1 norm, L2 norm, max absolute)
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    feature_space : str
        Name of feature space (e.g., 'activity', 'motor')
    fmriprep_path : Path
    vp_results_path : Path or None
    figures_path : Path
        
    Returns:
    --------
    dict : Paths to created CIFTI files
    """
    print("="*80)
    print(f"CREATING {feature_space.upper()} WEIGHT CIFTI MAPS")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Train sessions: {train_sessions}")
    print(f"Test sessions: {test_sessions}")
    print(f"Feature space: {feature_space}")
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
    print(f"Threshold suffix: {threshold_suffix}")
    print()
    
    # Load extracted weights
    weights_selective, selective_mask, valid_voxels_mask, metadata = load_extracted_weights(
        vp_output_dir, feature_space, threshold_suffix
    )
    print()
    
    # Get template CIFTI
    print("Getting template CIFTI...")
    template_path = get_template_cifti_path(subject, fmriprep_path)
    print(f"  Using: {template_path}")
    print()
    
    # Create output directory
    cifti_output_dir = figures_path / 'weight_ciftis' / dir_name
    cifti_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving CIFTIs to: {cifti_output_dir}")
    print()
    
    created_files = {}
    
    # Create per-feature weight maps
    print("Creating per-feature weight maps...")
    feature_names = metadata['feature_names']
    weights_path = cifti_output_dir / f'{feature_space}_weights_per_feature.dscalar.nii'
    create_weight_cifti(
        weights_selective, selective_mask, valid_voxels_mask,
        template_path, weights_path, feature_names
    )
    created_files['per_feature'] = weights_path
    print()
    
    # Create summary weight maps
    print("Creating summary weight maps...")
    summary_files = create_summary_weight_ciftis(
        weights_selective, selective_mask, valid_voxels_mask,
        template_path, cifti_output_dir, feature_space
    )
    created_files.update(summary_files)
    print()
    
    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nFeature space: {feature_space}")
    print(f"Features: {len(feature_names)}")
    print(f"  {', '.join(feature_names)}")
    print(f"\nSelective voxels: {metadata['n_selective_voxels']} ({metadata['pct_selective']:.1f}%)")
    print(f"Total voxels (after filtering): {len(selective_mask)}")
    
    print(f"\nWeight statistics (across selective voxels):")
    print(f"  Mean L1 norm: {np.abs(weights_selective).sum(axis=0).mean():.4f}")
    print(f"  Mean L2 norm: {np.sqrt((weights_selective**2).sum(axis=0)).mean():.4f}")
    print(f"  Max absolute weight: {np.abs(weights_selective).max():.4f}")
    
    print()
    print("="*80)
    print("VISUALIZATION COMMANDS")
    print("="*80)
    print(f"\nView per-feature weights:")
    print(f"  wb_view {created_files['per_feature']}")
    print(f"\nView summary maps:")
    for summary_type in ['L1_norm', 'L2_norm', 'max_abs']:
        print(f"  wb_view {created_files[summary_type]}")
    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)
    
    return created_files


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create CIFTI weight maps from extracted feature space weights'
    )
    parser.add_argument('--subject', type=int, required=True, help='Subject number')
    parser.add_argument('--train-sessions', type=int, nargs='+', required=True,
                       help='Training session numbers (e.g., 6 7 8 9)')
    parser.add_argument('--test-sessions', type=int, nargs='+', required=True,
                       help='Test session numbers (e.g., 20 21)')
    parser.add_argument('--feature-space', type=str, required=True,
                       choices=['perception', 'motor', 'action', 'scene', 'activity'],
                       help='Which feature space to visualize')
    parser.add_argument('--threshold-suffix', type=str, default='unique_thresh0p010',
                       help='Threshold suffix (e.g., unique_thresh0p010, full_thresh0p300, all)')
    
    args = parser.parse_args()
    
    # Create CIFTIs
    created_files = create_all_weight_ciftis(
        args.subject, args.train_sessions, args.test_sessions, args.feature_space,
        threshold_suffix=args.threshold_suffix
    )
    
    return created_files


if __name__ == '__main__':
    main()
