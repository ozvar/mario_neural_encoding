"""
Create CIFTI maps for variance partitioning models.

Adapted version of create_model_cifti.py that works with variance partitioning
output structure (loads from R2_scores.npz and significance results).
"""
import argparse
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from nibabel.cifti2 import Cifti2Header, Cifti2Image
from nibabel.cifti2.cifti2_axes import ScalarAxis

from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.utils.data_loading import get_template_cifti_path


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


def construct_vp_directory(subject, train_sessions, test_sessions, experiment_id, vp_results_path):
    """Construct variance partitioning directory path."""
    if vp_results_path is None:
        vp_results_path = PATHS['variance_partitioning']
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    vp_output_dir = vp_results_path / dataset_dir / experiment_id
    if not vp_output_dir.exists():
        raise FileNotFoundError(f"Variance partitioning results not found: {vp_output_dir}")
    print(f"Loading from: {vp_output_dir}")
    print()
    return vp_output_dir, dataset_dir


def load_metadata(vp_output_dir):
    """Load metadata to check what metrics were computed."""
    metadata_file = vp_output_dir / 'metadata.json'
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    computed_metrics = metadata.get('computed_metrics', {})
    feature_spaces = list(metadata.get('feature_spaces', {}).keys())
    print("Computed metrics:")
    print(f"  Product measure: {computed_metrics.get('product_measure', False)}")
    print(f"  Unique variance: {computed_metrics.get('unique_variance', False)}")
    print(f"  Fisher z: {computed_metrics.get('fisher_z', False)}")
    print(f"Feature spaces: {feature_spaces}")
    print()
    return metadata, computed_metrics, feature_spaces


def load_r2_data(vp_output_dir):
    """Load R2 scores from compressed npz file."""
    r2_scores_file = vp_output_dir / 'R2_scores.npz'
    if not r2_scores_file.exists():
        raise FileNotFoundError(f"R2_scores.npz not found: {r2_scores_file}")
    r2_data = np.load(r2_scores_file)
    test_scores = r2_data['R2_full']
    print(f"Loaded test R2 scores: {test_scores.shape}")
    print(f"  Available keys in R2_scores.npz: {list(r2_data.keys())}")
    print()
    return r2_data, test_scores


def load_voxel_mask(vp_output_dir):
    """Load voxel mask."""
    voxel_mask_file = vp_output_dir / 'valid_voxels_mask.npy'
    if not voxel_mask_file.exists():
        raise FileNotFoundError(f"Voxel mask not found: {voxel_mask_file}")
    voxel_mask = np.load(voxel_mask_file)
    print(f"Loaded voxel mask: {voxel_mask.shape}, kept: {voxel_mask.sum()}")
    print()
    return voxel_mask


def load_significance_results(vp_output_dir):
    """Load significance testing results if available."""
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
        return {
            'p_uncorrected': p_uncorrected,
            'p_fdr': p_fdr,
            'sig_uncorrected': sig_uncorrected,
            'sig_fdr': sig_fdr,
            'available': True
        }
    else:
        print("Warning: Significance results not found. Skipping p-value CIFTIs.")
        print("  Run compute_vp_significance.py first to generate significance results.")
        print()
        return {'available': False}


def setup_template_and_output(subject, train_sessions, test_sessions, experiment_id,
                               fmri_path, pipeline, figures_path, dataset_dir):
    """Get template CIFTI and create output directory."""
    print("Getting template CIFTI...")
    template_path = get_template_cifti_path(subject, fmri_path, pipeline)
    print(f"  Using: {template_path}")
    print()
    cifti_output_dir = figures_path / 'model_ciftis' / dataset_dir / experiment_id
    cifti_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving CIFTIs to: {cifti_output_dir}")
    print()
    return template_path, cifti_output_dir


def prepare_data_and_paths(subject, train_sessions, test_sessions, experiment_id,
                           fmri_path, pipeline, vp_results_path, figures_path):
    """
    Load all data and prepare paths for CIFTI creation.
    
    Returns:
    --------
    data : dict
        Contains all loaded data and paths needed for CIFTI creation
    """
    vp_output_dir, dataset_dir = construct_vp_directory(
        subject, train_sessions, test_sessions, experiment_id, vp_results_path
    )
    metadata, computed_metrics, feature_spaces = load_metadata(vp_output_dir)
    r2_data, test_scores = load_r2_data(vp_output_dir)
    voxel_mask = load_voxel_mask(vp_output_dir)
    significance = load_significance_results(vp_output_dir)
    template_path, cifti_output_dir = setup_template_and_output(
        subject, train_sessions, test_sessions, experiment_id,
        fmri_path, pipeline, figures_path, dataset_dir
    )
    return {
        'vp_output_dir': vp_output_dir,
        'dataset_dir': dataset_dir,
        'experiment_id': experiment_id,
        'metadata': metadata,
        'computed_metrics': computed_metrics,
        'feature_spaces': feature_spaces,
        'r2_data': r2_data,
        'test_scores': test_scores,
        'voxel_mask': voxel_mask,
        'significance': significance,
        'template_path': template_path,
        'cifti_output_dir': cifti_output_dir
    }


def create_base_r2_ciftis(r2_data, voxel_mask, template_path, 
                          cifti_output_dir, dataset_dir, experiment_id):
    """Create CIFTIs for base R2 metrics (R2_full and Fisher z)."""
    created_files = {}
    print("Creating test R2 CIFTI...")
    test_scores = r2_data['R2_full']
    test_r2_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_test_r2.dscalar.nii'
    create_cifti_from_scores(
        test_scores, voxel_mask, template_path, test_r2_path,
        map_name='test_R2'
    )
    created_files['test_r2'] = test_r2_path
    print()
    if 'fisher_z_R2_full' in r2_data:
        print("Creating Fisher z CIFTI...")
        fisher_z_scores = r2_data['fisher_z_R2_full']
        fisher_z_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_fisher_z_r2.dscalar.nii'
        create_cifti_from_scores(
            fisher_z_scores, voxel_mask, template_path, fisher_z_path,
            map_name='fisher_z_R2'
        )
        created_files['fisher_z_r2'] = fisher_z_path
        print()
    return created_files


def create_product_measure_ciftis(r2_data, computed_metrics, feature_spaces,
                                  voxel_mask, template_path, cifti_output_dir,
                                  dataset_dir, experiment_id):
    """Create CIFTIs for product measure decomposition."""
    created_files = {}
    if not computed_metrics.get('product_measure', False):
        return created_files
    print("Creating product measure CIFTIs...")
    for space in feature_spaces:
        key = f'product_measure_{space}'
        if key in r2_data:
            pm_scores = r2_data[key]
            pm_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_product_measure_{space}.dscalar.nii'
            create_cifti_from_scores(
                pm_scores, voxel_mask, template_path, pm_path,
                map_name=f'product_measure_{space}'
            )
            created_files[f'product_measure_{space}'] = pm_path
            print(f"  {space}: mean={pm_scores.mean():.6f}, min={pm_scores.min():.6f}, max={pm_scores.max():.6f}")
    print()
    return created_files


def create_unique_variance_ciftis(r2_data, computed_metrics, feature_spaces,
                                  voxel_mask, template_path, cifti_output_dir,
                                  dataset_dir, experiment_id):
    """Create CIFTIs for unique variance per feature space."""
    created_files = {}
    if not computed_metrics.get('unique_variance', False):
        return created_files
    print("Creating unique variance CIFTIs...")
    for space in feature_spaces:
        unique_key = f'R2_unique_{space}'
        if unique_key in r2_data:
            unique_scores = r2_data[unique_key]
            unique_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_unique_variance_{space}.dscalar.nii'
            create_cifti_from_scores(
                unique_scores, voxel_mask, template_path, unique_path,
                map_name=f'unique_R2_{space}'
            )
            created_files[f'unique_variance_{space}'] = unique_path
            print(f"  {space}: mean={unique_scores.mean():.6f}, positive={100*(unique_scores>0).mean():.1f}%")
        fisher_z_unique_key = f'fisher_z_R2_unique_{space}'
        if fisher_z_unique_key in r2_data:
            fisher_z_unique = r2_data[fisher_z_unique_key]
            fisher_z_unique_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_fisher_z_unique_{space}.dscalar.nii'
            create_cifti_from_scores(
                fisher_z_unique, voxel_mask, template_path, fisher_z_unique_path,
                map_name=f'fisher_z_unique_{space}'
            )
            created_files[f'fisher_z_unique_{space}'] = fisher_z_unique_path
    print()
    return created_files


def create_derived_metrics_ciftis(r2_data, voxel_mask, template_path,
                                  cifti_output_dir, dataset_dir, experiment_id):
    """Create CIFTIs for derived metrics (shared variance, integration index)."""
    created_files = {}
    if 'R2_shared' in r2_data:
        print("Creating shared variance CIFTI...")
        shared_scores = r2_data['R2_shared']
        shared_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_shared_variance.dscalar.nii'
        create_cifti_from_scores(
            shared_scores, voxel_mask, template_path, shared_path,
            map_name='R2_shared'
        )
        created_files['shared_variance'] = shared_path
        print(f"  mean={shared_scores.mean():.6f}, positive={100*(shared_scores>0).mean():.1f}%")
        print()
    if 'fisher_z_R2_shared' in r2_data:
        print("Creating Fisher z shared variance CIFTI...")
        fisher_z_shared = r2_data['fisher_z_R2_shared']
        fisher_z_shared_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_fisher_z_shared.dscalar.nii'
        create_cifti_from_scores(
            fisher_z_shared, voxel_mask, template_path, fisher_z_shared_path,
            map_name='fisher_z_R2_shared'
        )
        created_files['fisher_z_shared'] = fisher_z_shared_path
        print()
    if 'integration_index' in r2_data:
        print("Creating integration index CIFTI...")
        seg_index = r2_data['integration_index']
        seg_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_integration_index.dscalar.nii'
        create_cifti_from_scores(
            seg_index, voxel_mask, template_path, seg_path,
            map_name='integration_index'
        )
        created_files['integration_index'] = seg_path
        print(f"  mean={seg_index.mean():.6f}, range=[{seg_index.min():.6f}, {seg_index.max():.6f}]")
        print()
    return created_files


def create_significance_ciftis(significance, voxel_mask, template_path,
                               cifti_output_dir, dataset_dir, experiment_id):
    """Create CIFTIs for significance testing results."""
    created_files = {}
    if not significance['available']:
        return created_files
    print("Creating uncorrected p-value CIFTI...")
    p_uncorrected_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_p_uncorrected.dscalar.nii'
    create_cifti_from_scores(
        significance['p_uncorrected'], voxel_mask, template_path, p_uncorrected_path,
        map_name='p_uncorrected'
    )
    created_files['p_uncorrected'] = p_uncorrected_path
    print()
    print("Creating FDR q-value CIFTI...")
    p_fdr_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_q_fdr.dscalar.nii'
    create_cifti_from_scores(
        significance['p_fdr'], voxel_mask, template_path, p_fdr_path,
        map_name='q_fdr'
    )
    created_files['q_fdr'] = p_fdr_path
    print()
    print("Creating uncorrected significance mask CIFTI...")
    sig_uncorrected_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_sig_p0.05.dscalar.nii'
    create_cifti_from_scores(
        significance['sig_uncorrected'].astype(np.float32), voxel_mask, template_path, sig_uncorrected_path,
        map_name='sig_p<0.05'
    )
    created_files['sig_uncorrected'] = sig_uncorrected_path
    print()
    print("Creating FDR significance mask CIFTI...")
    sig_fdr_path = cifti_output_dir / f'{dataset_dir}_{experiment_id}_sig_q0.05.dscalar.nii'
    create_cifti_from_scores(
        significance['sig_fdr'].astype(np.float32), voxel_mask, template_path, sig_fdr_path,
        map_name='sig_q<0.05'
    )
    created_files['sig_fdr'] = sig_fdr_path
    print()
    return created_files


def generate_ciftis(data):
    """
    Generate all CIFTI files from loaded data.
    
    Parameters:
    -----------
    data : dict
        Dictionary from prepare_data_and_paths()
        
    Returns:
    --------
    created_files : dict
        Paths to all created CIFTI files
    """
    created_files = {}
    created_files.update(create_base_r2_ciftis(
        data['r2_data'], data['voxel_mask'], data['template_path'],
        data['cifti_output_dir'], data['dataset_dir'], data['experiment_id']
    ))
    created_files.update(create_product_measure_ciftis(
        data['r2_data'], data['computed_metrics'], data['feature_spaces'],
        data['voxel_mask'], data['template_path'], data['cifti_output_dir'],
        data['dataset_dir'], data['experiment_id']
    ))
    created_files.update(create_unique_variance_ciftis(
        data['r2_data'], data['computed_metrics'], data['feature_spaces'],
        data['voxel_mask'], data['template_path'], data['cifti_output_dir'],
        data['dataset_dir'], data['experiment_id']
    ))
    created_files.update(create_derived_metrics_ciftis(
        data['r2_data'], data['voxel_mask'], data['template_path'],
        data['cifti_output_dir'], data['dataset_dir'], data['experiment_id']
    ))
    created_files.update(create_significance_ciftis(
        data['significance'], data['voxel_mask'], data['template_path'],
        data['cifti_output_dir'], data['dataset_dir'], data['experiment_id']
    ))
    return created_files


def print_results_summary(data, created_files):
    """Print summary of results and created files."""
    test_scores = data['test_scores']
    r2_data = data['r2_data']
    computed_metrics = data['computed_metrics']
    feature_spaces = data['feature_spaces']
    significance = data['significance']
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
    if computed_metrics.get('product_measure', False):
        print(f"\nProduct measure:")
        for space in feature_spaces:
            key = f'product_measure_{space}'
            if key in r2_data:
                pm = r2_data[key]
                print(f"  {space}: mean={pm.mean():.4f}, negative={100*(pm<0).mean():.1f}%")
    if computed_metrics.get('unique_variance', False):
        print(f"\nUnique variance:")
        for space in feature_spaces:
            key = f'R2_unique_{space}'
            if key in r2_data:
                uv = r2_data[key]
                print(f"  {space}: mean={uv.mean():.4f}, positive={100*(uv>0).mean():.1f}%")
    if 'R2_shared' in r2_data:
        print(f"\nDerived metrics:")
        shared = r2_data['R2_shared']
        print(f"  Shared variance: mean={shared.mean():.4f}, positive={100*(shared>0).mean():.1f}%")
        if 'integration_index' in r2_data:
            seg = r2_data['integration_index']
            print(f"  Segregation index: mean={seg.mean():.4f}, range=[{seg.min():.4f}, {seg.max():.4f}]")
    if significance['available']:
        print(f"\nSignificance:")
        print(f"  Uncorrected (p < 0.05): {significance['sig_uncorrected'].sum()} ({significance['sig_uncorrected'].mean()*100:.1f}%)")
        print(f"  FDR (q < 0.05): {significance['sig_fdr'].sum()} ({significance['sig_fdr'].mean()*100:.1f}%)")
    print(f"\nCreated {len(created_files)} CIFTI files:")
    for key, path in created_files.items():
        print(f"  {key}: {path.name}")
    print()
    print("="*80)
    print("VISUALIZATION COMMAND")
    print("="*80)
    if 'test_r2' in created_files:
        print(f"\nwb_view {created_files['test_r2']}")
    print()
    print("="*80)
    print("COMPLETE")
    print("="*80)


def create_vp_model_ciftis(subject, train_sessions, test_sessions, experiment_id,
                           fmri_path=None,
                           pipeline=None,
                           vp_results_path=None,
                           figures_path=PATHS['figures']):
    """
    Create CIFTI maps for variance partitioning results.
    
    Creates CIFTIs for:
    - Test R2 scores
    - Fisher z-transformed R2
    - Product measure per feature space (if computed)
    - Unique variance per feature space (if computed)
    - Fisher z of unique variance (if computed)
    - Shared variance (if computed)
    - Segregation index (if computed)
    - Fisher z of shared variance (if computed)
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
    fmri_path : Path or None
        Defaults to config-resolved path if None
    pipeline : str or None
        'fmriprep' or 'hcp'. Defaults to PARAMETERS['preprocessing_pipeline'] if None
    vp_results_path : Path or None
    figures_path : Path
        
    Returns:
    --------
    dict : Paths to created CIFTI files
    """
    if pipeline is None:
        pipeline = PARAMETERS['preprocessing_pipeline']
    if fmri_path is None:
        fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']

    print("="*80)
    print("CREATING VARIANCE PARTITIONING MODEL CIFTI")
    print("="*80)
    print(f"Subject: {subject}")
    print(f"Train sessions: {train_sessions}")
    print(f"Test sessions: {test_sessions}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Pipeline: {pipeline}")
    print()
    data = prepare_data_and_paths(
        subject, train_sessions, test_sessions, experiment_id,
        fmri_path, pipeline, vp_results_path, figures_path
    )
    created_files = generate_ciftis(data)
    print_results_summary(data, created_files)
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
    created_files = create_vp_model_ciftis(
        args.subject, args.train_sessions, args.test_sessions, args.experiment_id
    )
    return created_files


if __name__ == '__main__':
    main()
