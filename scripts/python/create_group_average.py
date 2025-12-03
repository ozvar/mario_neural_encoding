"""
Create group-average maps from individual subject variance partitioning results.

This script loads variance partitioning results from multiple subjects (potentially
with different train/test splits and experiment IDs) and computes group averages
for specified metrics.

Output structure:
    {PATHS['variance_partitioning']}/
        group_average/
            {experiment_id}/
                config.json
                group_averages.npz
                group_metadata.json
                group_R2_full_mean.dscalar.nii
                group_R2_unique_motor_mean.dscalar.nii
                ...
"""
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from nibabel.cifti2 import Cifti2Header, Cifti2Image
from nibabel.cifti2.cifti2_axes import ScalarAxis

from mario_encoding.config import PATHS


def setup_logging(output_dir):
    """Configure logging to both file and console."""
    log_file = output_dir / 'group_average.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)


def construct_subject_path(subject, train_sessions, test_sessions, experiment_id, base_path):
    """
    Construct path to subject's variance partitioning results.
    
    Parameters:
    -----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int
    experiment_id : str
    base_path : Path
        
    Returns:
    --------
    subject_dir : Path
    """
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    subject_dir = base_path / dataset_dir / experiment_id
    return subject_dir


def load_subject_results(subject_spec, base_path, logger):
    """
    Load variance partitioning results for a single subject.
    
    Parameters:
    -----------
    subject_spec : dict
    base_path : Path
    logger : logging.Logger
        
    Returns:
    --------
    results : dict
    """
    subject = subject_spec['subject']
    train_sessions = subject_spec['train_sessions']
    test_sessions = subject_spec['test_sessions']
    experiment_id = subject_spec['experiment_id']
    subject_dir = construct_subject_path(
        subject, train_sessions, test_sessions, experiment_id, base_path
    )
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject results not found: {subject_dir}")
    r2_scores_path = subject_dir / 'R2_scores.npz'
    metadata_path = subject_dir / 'metadata.json'
    if not r2_scores_path.exists():
        raise FileNotFoundError(f"R2 scores not found: {r2_scores_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    r2_data = dict(np.load(r2_scores_path))
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return {
        'r2_data': r2_data,
        'metadata': metadata,
        'subject': subject,
        'path': subject_dir
    }


def check_grayordinates_match(subjects_data, n_expected):
    """Check all subjects have matching number of grayordinates."""
    for subj_data in subjects_data[1:]:
        subj_meta = subj_data['metadata']
        subj_num = subj_data['subject']
        n_voxels = subj_meta['n_voxels_original']
        if n_voxels != n_expected:
            raise ValueError(
                f"Subject {subj_num} has {n_voxels} grayordinates, expected {n_expected}"
            )


def check_feature_spaces_match(subjects_data, logger):
    """Check all subjects have matching feature spaces and features."""
    first_meta = subjects_data[0]['metadata']
    feature_spaces = set(first_meta['feature_spaces'].keys())
    warnings = []
    for subj_data in subjects_data[1:]:
        subj_meta = subj_data['metadata']
        subj_num = subj_data['subject']
        subj_spaces = set(subj_meta['feature_spaces'].keys())
        if subj_spaces != feature_spaces:
            warnings.append(
                f"Subject {subj_num} has different feature spaces: {subj_spaces} vs {feature_spaces}"
            )
        for space_name in feature_spaces:
            if space_name not in subj_meta['feature_spaces']:
                continue
            first_features = set(first_meta['feature_spaces'][space_name])
            subj_features = set(subj_meta['feature_spaces'][space_name])
            if first_features != subj_features:
                warnings.append(
                    f"Subject {subj_num} has different features in {space_name}"
                )
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    return warnings


def check_experiment_parameters(subjects_data):
    """Check for differences in experiment parameters, return warnings."""
    warnings = []
    first_meta = subjects_data[0]['metadata']
    first_config = first_meta.get('variance_partitioning', {})
    for subj_data in subjects_data[1:]:
        subj_meta = subj_data['metadata']
        subj_num = subj_data['subject']
        subj_config = subj_meta.get('variance_partitioning', {})
        if first_config.get('delays') != subj_config.get('delays'):
            warnings.append(
                f"Subject {subj_num} has different delays: "
                f"{subj_config.get('delays')} vs {first_config.get('delays')}"
            )
        if first_config.get('solver') != subj_config.get('solver'):
            warnings.append(
                f"Subject {subj_num} has different solver: "
                f"{subj_config.get('solver')} vs {first_config.get('solver')}"
            )
    return warnings


def validate_subjects_compatibility(subjects_data, logger):
    """
    Validate that all subjects have compatible data for averaging.
    
    Parameters:
    -----------
    subjects_data : list of dict
    logger : logging.Logger
        
    Returns:
    --------
    validation_report : dict
    """
    first_meta = subjects_data[0]['metadata']
    n_grayordinates = first_meta['n_voxels_original']
    if n_grayordinates != 91282:
        logger.warning(f"Expected 91282 grayordinates, got {n_grayordinates}")
    check_grayordinates_match(subjects_data, n_grayordinates)
    feature_warnings = check_feature_spaces_match(subjects_data, logger)
    param_warnings = check_experiment_parameters(subjects_data)
    all_warnings = feature_warnings + param_warnings
    if param_warnings:
        logger.warning(f"Parameter differences detected:")
        for warning in param_warnings:
            logger.warning(f"  {warning}")
    return {
        'compatible': True,
        'warnings': all_warnings,
        'n_grayordinates': n_grayordinates
    }


def load_metric_for_subject(subj_data, metric_name, n_grayordinates):
    """
    Load a metric for a single subject and expand to full grayordinate array.
    
    Parameters:
    -----------
    subj_data : dict
    metric_name : str
    n_grayordinates : int
        
    Returns:
    --------
    full_values : array of shape (n_grayordinates,)
    voxel_mask : boolean array
    """
    subj_num = subj_data['subject']
    r2_data = subj_data['r2_data']
    if metric_name not in r2_data:
        raise ValueError(
            f"Metric '{metric_name}' not found in subject {subj_num} results. "
            f"Available: {list(r2_data.keys())}"
        )
    kept_values = r2_data[metric_name]
    voxel_mask = np.load(subj_data['path'] / 'valid_voxels_mask.npy')
    if len(voxel_mask) != n_grayordinates:
        raise ValueError(
            f"Subject {subj_num} voxel mask wrong length: {len(voxel_mask)} vs {n_grayordinates}"
        )
    if voxel_mask.sum() != len(kept_values):
        raise ValueError(
            f"Subject {subj_num} mask sum ({voxel_mask.sum()}) != data length ({len(kept_values)})"
        )
    full_values = np.full(n_grayordinates, np.nan, dtype=np.float32)
    full_values[voxel_mask] = kept_values
    return full_values, voxel_mask


def average_metric_across_subjects(subjects_data, metric_name, n_grayordinates):
    """Average a single metric across subjects."""
    all_values = []
    all_masks = []
    for subj_data in subjects_data:
        full_values, voxel_mask = load_metric_for_subject(
            subj_data, metric_name, n_grayordinates
        )
        all_values.append(full_values)
        all_masks.append(voxel_mask)
    all_values = np.array(all_values)
    all_masks = np.array(all_masks)
    with np.errstate(invalid='ignore'):
        mean_values = np.nanmean(all_values, axis=0)
    n_valid = all_masks.sum(axis=0)
    return {
        'mean': mean_values,
        'n_valid': n_valid
    }


def compute_group_averages(subjects_data, metrics_to_average, n_grayordinates, logger):
    """
    Compute group averages for specified metrics.
    
    Parameters:
    -----------
    subjects_data : list of dict
    metrics_to_average : list of str
    n_grayordinates : int
    logger : logging.Logger
        
    Returns:
    --------
    group_averages : dict
    """
    group_averages = {}
    for metric_name in metrics_to_average:
        group_averages[metric_name] = average_metric_across_subjects(
            subjects_data, metric_name, n_grayordinates
        )
    return group_averages


def save_group_results(group_averages, output_dir, config, subjects_metadata, logger):
    """
    Save group average results to disk.
    
    Parameters:
    -----------
    group_averages : dict
    output_dir : Path
    config : dict
    subjects_metadata : list of dict
    logger : logging.Logger
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    save_dict = {}
    for metric_name, data in group_averages.items():
        save_dict[f'{metric_name}_mean'] = data['mean']
        save_dict[f'{metric_name}_n_valid'] = data['n_valid']
    np.savez_compressed(output_dir / 'group_averages.npz', **save_dict)
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_subjects': len(subjects_metadata),
        'subjects': [m['subject'] for m in subjects_metadata],
        'output_label': config['output_label'],
        'metrics_averaged': list(group_averages.keys()),
        'subject_details': [
            {
                'subject': subj_data['subject'],
                'train_sessions': subj_data['train_sessions'],
                'test_sessions': subj_data['test_sessions'],
                'experiment_id': subj_data['experiment_id']
            }
            for subj_data in config['subjects']
        ]
    }
    with open(output_dir / 'group_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def create_group_ciftis(group_averages, output_dir, template_cifti_path, logger):
    """
    Create CIFTI files for group-averaged metrics.
    
    Parameters:
    -----------
    group_averages : dict
    output_dir : Path
    template_cifti_path : Path
    logger : logging.Logger
    """
    template = nib.load(str(template_cifti_path))
    brain_model_axis = template.header.get_axis(1)
    for metric_name, data in group_averages.items():
        mean_values = data['mean']
        cifti_data = mean_values[np.newaxis, :]
        map_name = f'{metric_name}_mean'
        scalar_axis = ScalarAxis([map_name])
        new_header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))
        new_cifti = Cifti2Image(cifti_data, header=new_header)
        output_path = output_dir / f'group_{metric_name}_mean.dscalar.nii'
        new_cifti.to_filename(str(output_path))


def get_template_cifti_path(subject, fmriprep_path):
    """Get path to a template CIFTI file from first subject."""
    subject_dir = fmriprep_path / f'sub-{subject:02d}'
    cifti_files = sorted(subject_dir.rglob('*_space-fsLR_den-91k_bold.dtseries.nii'))
    if not cifti_files:
        raise FileNotFoundError(
            f"No CIFTI files found for subject {subject} in {subject_dir}"
        )
    return cifti_files[0]


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create group-average maps from individual variance partitioning results'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Name of config from mario_encoding.group_configs (e.g., GROUP_N3_PRIMARY)')
    args = parser.parse_args()
    from mario_encoding import group_configs
    if not hasattr(group_configs, args.config):
        raise ValueError(
            f"Config '{args.config}' not found in mario_encoding.group_configs. "
            f"Available: {[name for name in dir(group_configs) if name.isupper()]}"
        )
    config = getattr(group_configs, args.config)
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PATHS['variance_partitioning'] / 'group_average' / experiment_id
    logger = setup_logging(output_dir)
    logger.info("="*80)
    logger.info("GROUP AVERAGE ANALYSIS")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output label: {config['output_label']}")
    logger.info(f"Subjects: {[s['subject'] for s in config['subjects']]}")
    logger.info(f"Metrics: {len(config['metrics_to_average'])}")
    logger.info("")
    subjects_data = []
    for subj_spec in config['subjects']:
        subj_results = load_subject_results(
            subj_spec, PATHS['variance_partitioning'], logger
        )
        subjects_data.append(subj_results)
    validation = validate_subjects_compatibility(subjects_data, logger)
    group_averages = compute_group_averages(
        subjects_data, config['metrics_to_average'], 
        validation['n_grayordinates'], logger
    )
    save_group_results(
        group_averages, output_dir, config,
        [s['metadata'] for s in subjects_data], logger
    )
    first_subject = config['subjects'][0]['subject']
    template_cifti = get_template_cifti_path(first_subject, PATHS['fmriprep_data'])
    create_group_ciftis(group_averages, output_dir, template_cifti, logger)
    logger.info("")
    logger.info("="*80)
    logger.info("COMPLETE")
    logger.info("="*80)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Files: config.json, group_averages.npz, group_metadata.json")
    logger.info(f"CIFTIs: {len(group_averages)} mean maps")
    logger.info("")


if __name__ == '__main__':
    main()
