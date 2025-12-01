"""
Analyze variance partitioning results across Yeo functional networks.

This script loads R² maps (full, unique variances, computed shared) and quantifies
how different variance components are distributed across functional brain networks.

Usage:
    python analyze_variance_by_network.py --subject 1 --train-sessions 7 8 9 10 11 12 \
        --test-sessions 14 --experiment-id 20251121_210306 --r2-threshold 0.05
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib

from mario_encoding.config import PATHS


def load_experiment_config(results_dir):
    """
    Load experiment configuration from config.json.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing config.json
        
    Returns
    -------
    config : dict
        Experiment configuration
    feature_space_names : list of str
        Names of feature spaces in this experiment
    """
    import json
    
    config_path = results_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract feature space names
    if 'feature_spaces' not in config:
        raise KeyError("'feature_spaces' not found in config.json")
    
    feature_space_names = list(config['feature_spaces'].keys())
    
    return config, feature_space_names


def load_yeo_networks(yeo_path):
    """
    Load Yeo network labels from dlabel.nii file.
    
    Parameters
    ----------
    yeo_path : Path
        Path to Yeo dlabel.nii file
        
    Returns
    -------
    network_labels : np.ndarray (n_grayordinates,)
        Network assignment for each grayordinate (0 = unassigned, 1-7 = networks)
    network_names : list of str
        Names of the 7 Yeo networks in order
    """
    cifti = nib.load(yeo_path)
    network_labels = cifti.get_fdata().squeeze().astype(int)
    
    # Yeo 7 network names in order (ID 1-7)
    network_names = [
        'Visual',
        'Somatomotor', 
        'Dorsal Attention',
        'Ventral Attention',
        'Limbic',
        'Frontoparietal',
        'Default'
    ]
    
    return network_labels, network_names


def load_variance_maps(results_dir, valid_voxels_mask, feature_space_names):
    """
    Load R² maps from variance partitioning results.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing R2_scores.npz
    valid_voxels_mask : np.ndarray (n_grayordinates,)
        Boolean mask of valid voxels
    feature_space_names : list of str
        Names of feature spaces from config (e.g., ['motor', 'scene', 'activity'])
        
    Returns
    -------
    variance_maps : dict
        Dictionary with keys: 'R2_full', 'unique_{space}', 'shared'
        Each value is np.ndarray (n_grayordinates,) with NaN for invalid voxels
    """
    # Load R² scores
    r2_path = results_dir / 'R2_scores.npz'
    if not r2_path.exists():
        raise FileNotFoundError(f"R2_scores.npz not found at {r2_path}")
    
    data = np.load(r2_path)
    
    # Initialize full-size arrays with NaN
    n_grayordinates = len(valid_voxels_mask)
    variance_maps = {}
    
    # Load R2_full
    if 'R2_full' not in data:
        raise KeyError(f"R2_full not found in R2_scores.npz. Available keys: {list(data.keys())}")
    
    full_map = np.full(n_grayordinates, np.nan)
    full_map[valid_voxels_mask] = data['R2_full']
    variance_maps['R2_full'] = full_map
    
    # Load unique variance for each feature space
    for space_name in feature_space_names:
        npz_key = f'R2_unique_{space_name}'
        if npz_key not in data:
            raise KeyError(f"{npz_key} not found in R2_scores.npz. Available keys: {list(data.keys())}")
        
        full_map = np.full(n_grayordinates, np.nan)
        full_map[valid_voxels_mask] = data[npz_key]
        variance_maps[f'unique_{space_name}'] = full_map
    
    # Compute shared variance: R2_full - sum(unique variances)
    unique_sum = np.zeros(n_grayordinates)
    for space_name in feature_space_names:
        unique_sum += variance_maps[f'unique_{space_name}']
    
    variance_maps['shared'] = variance_maps['R2_full'] - unique_sum
    
    return variance_maps


def compute_network_statistics(variance_maps, network_labels, network_names, 
                               valid_voxels_mask, feature_space_names, r2_threshold=0.05):
    """
    Compute mean and std of variance components per network.
    
    Parameters
    ----------
    variance_maps : dict
        Dictionary of variance maps (R2_full, unique_*, shared)
    network_labels : np.ndarray
        Network assignment for each grayordinate
    network_names : list of str
        Names of networks
    valid_voxels_mask : np.ndarray
        Boolean mask of valid voxels
    feature_space_names : list of str
        Names of feature spaces in this experiment
    r2_threshold : float
        Minimum R²_full to include voxel in statistics
        
    Returns
    -------
    stats : dict
        Nested dict: stats[network_name][variance_type] = {'mean': float, 'std': float, 'n_voxels': int}
    """
    stats = {}
    
    # Get R²_full for thresholding
    r2_full = variance_maps['R2_full']
    
    # Build list of variance types to compute
    variance_types = ['R2_full'] + [f'unique_{space}' for space in feature_space_names] + ['shared']
    
    for net_id, net_name in enumerate(network_names, start=1):
        # Get voxels in this network
        network_mask = (network_labels == net_id) & valid_voxels_mask
        
        # Further threshold by R²_full
        network_mask_thresholded = network_mask & (r2_full > r2_threshold)
        n_voxels = network_mask_thresholded.sum()
        
        stats[net_name] = {'n_voxels': n_voxels}
        
        if n_voxels == 0:
            # No voxels pass threshold in this network
            for var_type in variance_types:
                stats[net_name][var_type] = {'mean': np.nan, 'std': np.nan}
            continue
        
        # Compute statistics for each variance type
        for var_type in variance_types:
            values = variance_maps[var_type][network_mask_thresholded]
            # Remove NaN values if any slipped through
            values = values[~np.isnan(values)]
            
            stats[net_name][var_type] = {
                'mean': np.mean(values) if len(values) > 0 else np.nan,
                'std': np.std(values) if len(values) > 0 else np.nan
            }
    
    return stats


def print_network_statistics(stats, network_names, feature_space_names):
    """
    Print formatted statistics table.
    
    Parameters
    ----------
    stats : dict
        Network statistics from compute_network_statistics()
    network_names : list of str
        Names of networks in order
    feature_space_names : list of str
        Names of feature spaces in this experiment
    """
    # Build header dynamically
    n_spaces = len(feature_space_names)
    # Base width: Network(20) + N_Voxels(12) + R²_Full(15) + Shared(15) = 62
    # Add 15 per unique space
    total_width = 62 + (n_spaces * 15)
    
    print("\n" + "="*total_width)
    print("VARIANCE DECOMPOSITION BY FUNCTIONAL NETWORK")
    print("="*total_width)
    print()
    
    # Build header
    header = f"{'Network':<20} {'N Voxels':<12} {'R² Full':<15} "
    for space_name in feature_space_names:
        header += f"{'Unique ' + space_name.capitalize():<15} "
    header += f"{'Shared':<15}"
    print(header)
    print("-"*total_width)
    
    # Print each network
    for net_name in network_names:
        n_vox = stats[net_name]['n_voxels']
        
        # Format R2_full
        r2_str = f"{stats[net_name]['R2_full']['mean']:.3f}±{stats[net_name]['R2_full']['std']:.3f}" if n_vox > 0 else "---"
        
        # Start building row
        row = f"{net_name:<20} {n_vox:<12} {r2_str:<15} "
        
        # Add unique variance for each feature space
        for space_name in feature_space_names:
            var_key = f'unique_{space_name}'
            var_str = f"{stats[net_name][var_key]['mean']:.3f}±{stats[net_name][var_key]['std']:.3f}" if n_vox > 0 else "---"
            row += f"{var_str:<15} "
        
        # Add shared variance
        shared_str = f"{stats[net_name]['shared']['mean']:.3f}±{stats[net_name]['shared']['std']:.3f}" if n_vox > 0 else "---"
        row += f"{shared_str:<15}"
        
        print(row)
    
    print("="*total_width)
    print()


def get_results_directory(subject, train_sessions, test_sessions, experiment_id):
    """
    Construct path to variance partitioning results directory.
    
    Parameters
    ----------
    subject : int
    train_sessions : list of int
    test_sessions : list of int  
    experiment_id : str
        
    Returns
    -------
    results_dir : Path
    """
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    results_dir = PATHS['variance_partitioning'] / dataset_dir / experiment_id
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    return results_dir


def main():
    parser = argparse.ArgumentParser(
        description='Analyze variance partitioning results across Yeo functional networks'
    )
    parser.add_argument('--subject', type=int, required=True,
                       help='Subject ID')
    parser.add_argument('--train-sessions', type=int, nargs='+', required=True,
                       help='Training session(s)')
    parser.add_argument('--test-sessions', type=int, nargs='+', required=True,
                       help='Test session(s)')
    parser.add_argument('--experiment-id', type=str, required=True,
                       help='Experiment ID (timestamp)')
    parser.add_argument('--r2-threshold', type=float, default=0.05,
                       help='Minimum R² threshold for including voxels (default: 0.05)')
    
    args = parser.parse_args()
    
    # Get results directory
    results_dir = get_results_directory(
        args.subject, args.train_sessions, args.test_sessions, args.experiment_id
    )
    
    # Load experiment config to get feature spaces
    config, feature_space_names = load_experiment_config(results_dir)
    
    print("="*90)
    print("YEO NETWORK ANALYSIS")
    print("="*90)
    print(f"Subject: {args.subject}")
    print(f"Train sessions: {args.train_sessions}")
    print(f"Test sessions: {args.test_sessions}")
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Feature spaces: {feature_space_names}")
    print(f"R² threshold: {args.r2_threshold}")
    print()
    
    # Get results directory
    print("Loading data...")
    print(f"Results directory: {results_dir}")
    
    # Load valid voxels mask
    mask_path = results_dir / 'valid_voxels_mask.npy'
    if not mask_path.exists():
        raise FileNotFoundError(f"Valid voxels mask not found: {mask_path}")
    valid_voxels_mask = np.load(mask_path)
    print(f"Valid voxels: {valid_voxels_mask.sum()} / {len(valid_voxels_mask)}")
    
    # Load Yeo networks
    yeo_path = PATHS['Yeo7_networks']
    print(f"Loading Yeo networks from: {yeo_path}")
    network_labels, network_names = load_yeo_networks(yeo_path)
    print(f"Loaded {len(network_names)} networks")
    
    # Load variance maps
    print("Loading variance maps...")
    variance_maps = load_variance_maps(results_dir, valid_voxels_mask, feature_space_names)
    print(f"Loaded maps: {list(variance_maps.keys())}")
    
    # Compute statistics per network
    print(f"Computing statistics (R² > {args.r2_threshold})...")
    stats = compute_network_statistics(
        variance_maps, network_labels, network_names, 
        valid_voxels_mask, feature_space_names, args.r2_threshold
    )
    
    # Print results
    print_network_statistics(stats, network_names, feature_space_names)
    
    print("Analysis complete.")


if __name__ == '__main__':
    main()
