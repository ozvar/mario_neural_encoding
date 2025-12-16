"""
Visualize variance partitioning results across functional brain networks.

Creates stacked bar charts showing variance decomposition (shared, unique motor, unique scene)
across Yeo or Cole-Anticevic functional networks.

Usage:
    # Subject-level
    python plot_variance_by_network.py \
        --subject 3 --train-sessions 11 12 13 14 15 16 --test-sessions 17 \
        --experiment-id 20251125_125937 \
        --parcellation cole-anticevic
    
    # Group-level
    python plot_variance_by_network.py \
        --group-config GROUP_N3_PRIMARY \
        --experiment-id 20251208_134311 \
        --parcellation cole-anticevic
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

from mario_encoding.config import PATHS
from mario_encoding.utils import sns_styleset
from mario_encoding import group_configs


# ============================================================================
# Network Configurations
# ============================================================================

def get_network_config(parcellation_name):
    """
    Get network labels, colors, and anatomical ordering for a parcellation.
    
    Parameters
    ----------
    parcellation_name : str
        'yeo' or 'cole-anticevic'
        
    Returns
    -------
    config : dict
        Contains 'labels', 'colors', 'order' (indices for anatomical hierarchy)
    """
    if parcellation_name == 'yeo':
        return {
            'labels': [
                'Visual',
                'Somatomotor', 
                'Dorsal Attention',
                'Ventral Attention',
                'Limbic',
                'Frontoparietal',
                'Default'
            ],
            'colors': [
                '#E79523',  # Visual - orange
                '#CD3E4E',  # Somatomotor - red
                '#00760F',  # Dorsal Attention - green
                '#DCF8A4',  # Ventral Attention - light green
                '#C43BFA',  # Limbic - purple
                '#4682B4',  # Frontoparietal - blue
                '#781286'   # Default - dark purple
            ],
            'order': [0, 1, 2, 3, 4, 5, 6]  # Anatomical: sensory → association → default
        }
    
    elif parcellation_name == 'cole-anticevic':
        return {
            'labels': [
                'Primary Visual',
                'Visual2',
                'Somatomotor',
                'Auditory',
                'Cingulo-Opercular',
                'Dorsal Attention',
                'Language',
                'Frontoparietal',
                'Ventral Multimodal',
                'Posterior Multimodal',
                'Orbito-Affective',
                'Default'
            ],
            'colors': [
                '#0000FF',  # Primary Visual - blue
                '#6400FF',  # Visual2 - purple
                '#00FFFF',  # Somatomotor - cyan
                '#FA3EFB',  # Auditory - pink
                '#990099',  # Cingulo-Opercular - violet
                '#00FF00',  # Dorsal Attention - light green
                '#009B9B',  # Language - teal
                '#FFFF00',  # Frontoparietal - yellow
                '#FF9D00',  # Ventral Multimodal - orange
                '#B15928',  # Posterior Multimodal - brown
                '#417D00',  # Orbito-Affective - olive green
                '#FF0000'   # Default - red
            ],
            'order': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Anatomical hierarchy
        }
    
    else:
        raise ValueError(f"Unknown parcellation: {parcellation_name}")


def get_variance_colors():
    """
    Get color scheme for variance components.
    
    Returns
    -------
    colors : dict
        Maps variance type to hex color
    """
    return {
        'shared': '#c44e52',   # deep red/magenta
        'motor': '#dd8452',    # orange
        'scene': '#4c72b0'     # blue
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_network_parcellation(parcellation_path):
    """
    Load network labels from dlabel.nii file.
    
    Parameters
    ----------
    parcellation_path : Path
        
    Returns
    -------
    network_labels : np.ndarray (n_grayordinates,)
        Network ID for each grayordinate (0 = unassigned, 1-N = networks)
    """
    cifti = nib.load(str(parcellation_path))
    network_labels = cifti.get_fdata().squeeze().astype(int)
    return network_labels


def load_experiment_config(results_dir):
    """
    Load experiment configuration from config.json.
    
    Parameters
    ----------
    results_dir : Path
        
    Returns
    -------
    config : dict
    feature_space_names : list of str
    """
    import json
    
    config_path = results_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if 'feature_spaces' not in config:
        raise KeyError("'feature_spaces' not found in config.json")
    
    feature_space_names = list(config['feature_spaces'].keys())
    
    return config, feature_space_names


def load_variance_maps(results_dir, valid_voxels_mask, feature_space_names):
    """
    Load R² maps from variance partitioning results.
    
    Parameters
    ----------
    results_dir : Path
    valid_voxels_mask : np.ndarray (n_grayordinates,)
        Boolean mask of valid voxels
    feature_space_names : list of str
        
    Returns
    -------
    variance_maps : dict
        Keys: 'R2_full', 'unique_{space}', 'shared'
        Values: np.ndarray (n_grayordinates,) with NaN for invalid voxels
    """
    r2_path = results_dir / 'R2_scores.npz'
    if not r2_path.exists():
        raise FileNotFoundError(f"R2_scores.npz not found at {r2_path}")
    
    data = np.load(r2_path)
    n_grayordinates = len(valid_voxels_mask)
    variance_maps = {}
    
    # Check if data is already in full grayordinate space (group-level)
    # or needs to be expanded from masked values (subject-level)
    is_group_level = len(data['R2_full']) == n_grayordinates
    
    # Load R2_full
    if 'R2_full' not in data:
        raise KeyError(f"R2_full not found in R2_scores.npz")
    
    if is_group_level:
        variance_maps['R2_full'] = data['R2_full'].astype(np.float64)
    else:
        full_map = np.full(n_grayordinates, np.nan)
        full_map[valid_voxels_mask] = data['R2_full']
        variance_maps['R2_full'] = full_map
    
    # Load unique variance for each feature space
    for space_name in feature_space_names:
        npz_key = f'R2_unique_{space_name}'
        if npz_key not in data:
            raise KeyError(f"{npz_key} not found in R2_scores.npz")
        
        if is_group_level:
            variance_maps[f'unique_{space_name}'] = data[npz_key].astype(np.float64)
        else:
            full_map = np.full(n_grayordinates, np.nan)
            full_map[valid_voxels_mask] = data[npz_key]
            variance_maps[f'unique_{space_name}'] = full_map
    
    # Load R2_shared directly from npz
    if 'R2_shared' not in data:
        raise KeyError(f"R2_shared not found in R2_scores.npz")
    
    if is_group_level:
        variance_maps['shared'] = data['R2_shared'].astype(np.float64)
    else:
        shared_map = np.full(n_grayordinates, np.nan)
        shared_map[valid_voxels_mask] = data['R2_shared']
        variance_maps['shared'] = shared_map
    
    return variance_maps


# ============================================================================
# Network Statistics
# ============================================================================

def compute_network_means(variance_maps, network_labels, network_config, 
                         valid_voxels_mask, variance_types, r2_threshold=0.05):
    """
    Compute mean variance per network for specified variance types.
    
    Parameters
    ----------
    variance_maps : dict
    network_labels : np.ndarray
    network_config : dict
        Contains 'labels', 'colors', 'order'
    valid_voxels_mask : np.ndarray
    variance_types : list of str
        e.g., ['shared', 'unique_motor', 'unique_scene']
    r2_threshold : float
        
    Returns
    -------
    means : np.ndarray (n_networks, n_variance_types)
    network_names_ordered : list of str
    """
    n_networks = len(network_config['labels'])
    n_types = len(variance_types)
    means = np.zeros((n_networks, n_types))
    
    r2_full = variance_maps['R2_full']
    
    for i, net_idx in enumerate(network_config['order']):
        net_id = net_idx + 1  # Network IDs are 1-indexed
        
        # Get voxels in this network above threshold
        network_mask = (network_labels == net_id) & valid_voxels_mask
        network_mask_thresh = network_mask & (r2_full > r2_threshold)
        
        # Compute mean for each variance type
        for j, var_type in enumerate(variance_types):
            values = variance_maps[var_type][network_mask_thresh]
            values = values[~np.isnan(values)]
            means[i, j] = np.mean(values) if len(values) > 0 else 0.0
    
    # Get network names in anatomical order
    network_names_ordered = [network_config['labels'][idx] for idx in network_config['order']]
    
    return means, network_names_ordered


# ============================================================================
# Plotting
# ============================================================================

def prepare_violin_data(variance_maps, network_labels, network_config, 
                        valid_voxels_mask, variance_types, r2_threshold=0.05):
    """
    Prepare data for distribution plots - extract voxel-level values per network.
    
    Parameters
    ----------
    variance_maps : dict
    network_labels : np.ndarray
    network_config : dict
    valid_voxels_mask : np.ndarray
    variance_types : list of str
    r2_threshold : float
        
    Returns
    -------
    violin_data : list of dict
        Each dict has: 'network', 'variance_type', 'values'
    network_names_ordered : list of str
    """
    r2_full = variance_maps['R2_full']
    violin_data = []
    
    for net_idx in network_config['order']:
        net_id = net_idx + 1
        net_name = network_config['labels'][net_idx]
        
        # Get voxels in this network above threshold
        network_mask = (network_labels == net_id) & valid_voxels_mask
        network_mask_thresh = network_mask & (r2_full > r2_threshold)
        
        # Extract values for each variance type
        for var_type in variance_types:
            values = variance_maps[var_type][network_mask_thresh]
            values = values[~np.isnan(values)]
            
            if len(values) > 0:
                violin_data.append({
                    'network': net_name,
                    'variance_type': var_type,
                    'values': values
                })
    
    network_names_ordered = [network_config['labels'][idx] for idx in network_config['order']]
    
    return violin_data, network_names_ordered


def plot_ridge_distributions_shared(violin_data, network_names, network_config, 
                                   variance_colors, output_path):
    """
    Create ridge plot for shared variance with network-specific colors.
    
    Parameters
    ----------
    violin_data : list of dict
        Each dict has 'network', 'variance_type', 'values'
    network_names : list of str
    network_config : dict
        Contains 'labels', 'colors', 'order'
    variance_colors : dict
        Maps variance_type to hex color
    output_path : Path
    """
    import pandas as pd
    from scipy import stats
    
    # Filter to only shared variance
    filtered_data = [item for item in violin_data 
                     if item['variance_type'] == 'shared']
    
    # Convert to DataFrame
    rows = []
    for item in filtered_data:
        for val in item['values']:
            rows.append({
                'Network': item['network'],
                'R²': val
            })
    
    df = pd.DataFrame(rows)
    
    # Compute median R² per network for ordering
    network_medians = df.groupby('Network')['R²'].median().sort_values(ascending=False)
    network_order = network_medians.index.tolist()
    
    # Create network color mapping from config
    network_color_map = {}
    for net_name in network_order:
        net_idx = network_config['labels'].index(net_name)
        network_color_map[net_name] = network_config['colors'][net_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 10))
    
    # Set global x-axis limits
    x_min = df['R²'].min() - 0.005
    x_max = df['R²'].max() * 1.1
    
    # Y positions for networks
    n_networks = len(network_order)
    y_spacing = 1.0 / (n_networks + 1)
    y_positions = np.arange(n_networks) * y_spacing + y_spacing
    network_y_map = {net: y for net, y in zip(network_order[::-1], y_positions)}
    
    # Draw ridges for each network
    for net in network_order[::-1]:
        y_base = network_y_map[net]
        color = network_color_map[net]
        
        # Get data for this network
        net_data = df[df['Network'] == net]['R²'].values
        
        if len(net_data) > 1:
            # Compute KDE with full range
            kde = stats.gaussian_kde(net_data, bw_method=0.5)
            x_range = np.linspace(x_min, x_max, 500)
            density = kde(x_range)
            
            # Normalize density for ridge height
            density_norm = density / density.max() * (y_spacing * 0.8)
            
            # Draw baseline
            ax.plot([x_min, x_max], [y_base, y_base], 
                   color='black', linewidth=0.5, alpha=0.3, zorder=1)
            
            # Draw filled ridge with network color
            ax.fill_between(x_range, y_base, y_base + density_norm,
                           color=color, alpha=0.6, zorder=2)
            
            # Draw black outline
            ax.plot(x_range, y_base + density_norm, 
                   color='black', linewidth=2, zorder=3)
    
    # Formatting
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    # Increase axis and tick styling
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='both', which='major', 
                   width=2, length=8,  # Thicker, longer ticks
                   labelsize=25,        # Larger labels
                   pad=10)              # More space between ticks and labels
    ax.set_xlabel('R² Shared', fontsize=28)
    ax.tick_params(axis='x', labelsize=28)
    
    # Set y-ticks to network names
    ax.set_yticks(y_positions)
    ax.set_yticklabels(network_order[::-1], fontsize=28)
    ax.tick_params(left=False)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ridge_distributions_overlapping(violin_data, network_names, network_config, 
                                        variance_colors, output_path):
    """
    Create single ridge plot with motor and scene overlapping on same axis.
    
    Parameters
    ----------
    violin_data : list of dict
        Each dict has 'network', 'variance_type', 'values'
    network_names : list of str
    network_config : dict
        Contains 'labels', 'colors', 'order'
    variance_colors : dict
        Maps variance_type to hex color
    output_path : Path
    """
    import pandas as pd
    from scipy import stats
    
    # Filter to only unique_motor and unique_scene
    filtered_data = [item for item in violin_data 
                     if item['variance_type'] in ['unique_motor', 'unique_scene']]
    
    # Convert to DataFrame
    rows = []
    for item in filtered_data:
        for val in item['values']:
            rows.append({
                'Network': item['network'],
                'variance_type': item['variance_type'],
                'R²': val
            })
    
    df = pd.DataFrame(rows)
    
    # Order by median scene variance only
    #scene_df = df[df['variance_type'] == 'unique_scene']
    #network_medians = scene_df.groupby('Network')['R²'].median().sort_values(ascending=False)

    # Order by median shared variance (to match shared plot)
    shared_data = [item for item in violin_data if item['variance_type'] == 'shared']
    shared_medians = {}
    for item in shared_data:
        net = item['network']
        if net not in shared_medians:
            shared_medians[net] = np.median(item['values'])
    network_medians = pd.Series(shared_medians).sort_values(ascending=False)
    network_order = network_medians.index.tolist()
    
    # Set up colors for variance types
    motor_color = '#188bb7'  # Blue 
    scene_color = '#ec8e48'  # Orange 
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 10))
    
    # Set global x-axis limits
    x_min = df['R²'].min() - 0.005
    x_max = df['R²'].max() * 1.1
    
    # Y positions for networks
    n_networks = len(network_order)
    y_spacing = 1.0 / (n_networks + 1)
    y_positions = np.arange(n_networks) * y_spacing + y_spacing
    network_y_map = {net: y for net, y in zip(network_order[::-1], y_positions)}
    
    # Draw ridges for each network
    for net in network_order[::-1]:
        y_base = network_y_map[net]
        
        # Draw baseline
        ax.plot([x_min, x_max], [y_base, y_base], 
               color='black', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Draw motor ridge (red)
        motor_data = df[(df['Network'] == net) & (df['variance_type'] == 'unique_motor')]['R²'].values
        if len(motor_data) > 1:
            kde = stats.gaussian_kde(motor_data, bw_method=0.5)
            x_range = np.linspace(x_min, x_max, 500)
            density = kde(x_range)
            density_norm = density / density.max() * (y_spacing * 0.8)
            
            # Fill
            ax.fill_between(x_range, y_base, y_base + density_norm,
                           color=motor_color, alpha=0.5, zorder=2)
            # Black outline
            ax.plot(x_range, y_base + density_norm, 
                   color='black', linewidth=2, zorder=4)
        
        # Draw scene ridge (green) on same baseline
        scene_data = df[(df['Network'] == net) & (df['variance_type'] == 'unique_scene')]['R²'].values
        if len(scene_data) > 1:
            kde = stats.gaussian_kde(scene_data, bw_method=0.5)
            x_range = np.linspace(x_min, x_max, 500)
            density = kde(x_range)
            density_norm = density / density.max() * (y_spacing * 0.8)
            
            # Fill
            ax.fill_between(x_range, y_base, y_base + density_norm,
                           color=scene_color, alpha=0.5, zorder=3)
            # Black outline
            ax.plot(x_range, y_base + density_norm, 
                   color='black', linewidth=2, zorder=5)
    
    # Formatting
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    # Increase axis and tick styling
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='both', which='major', 
                   width=2, length=8,  # Thicker, longer ticks
                   labelsize=28,        # Larger labels
                   pad=10)              # More space between ticks and labels
    ax.set_xlabel('R² Unique', fontsize=28)
    ax.tick_params(axis='x', labelsize=28)
    
    # Set y-ticks to network names
    ax.set_yticks(y_positions)
    #ax.set_yticklabels(network_order[::-1], fontsize=18)
    ax.set_yticklabels([])
    ax.tick_params(left=False)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=motor_color, alpha=0.5, edgecolor='black', 
              label='Unique Motor', linewidth=2),
        Patch(facecolor=scene_color, alpha=0.5, edgecolor='black',
              label='Unique Scene', linewidth=2)
    ]
    #ax.legend(handles=legend_elements, loc='upper right', fontsize=16, frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_grouped_bars(means, network_names, variance_type_labels, variance_colors, output_path):
    """
    Create grouped bar chart of variance decomposition per network.
    
    Parameters
    ----------
    means : np.ndarray (n_networks, n_variance_types)
    network_names : list of str
    variance_type_labels : list of str
        e.g., ['Shared', 'Motor', 'Scene']
    variance_colors : list of str
        Hex colors for each variance type
    output_path : Path
    """
    n_networks = len(network_names)
    n_types = len(variance_type_labels)
    
    # Compute total R² per network and sort descending
    total_r2 = means.sum(axis=1)
    sort_idx = np.argsort(total_r2)[::-1]
    means_sorted = means[sort_idx]
    network_names_sorted = [network_names[i] for i in sort_idx]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create grouped bars
    x = np.arange(n_networks)
    bar_width = 0.25
    
    for i in range(n_types):
        offset = (i - n_types/2 + 0.5) * bar_width
        ax.bar(x + offset, means_sorted[:, i], bar_width,
               label=variance_type_labels[i], color=variance_colors[i],
               edgecolor='white', linewidth=1)
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(network_names_sorted, rotation=45, ha='right', fontsize=14)
    ax.set_ylabel('Mean R²', fontsize=16)
    ax.set_xlabel('Functional Network', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(frameon=False, loc='upper right', fontsize=14)
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_stacked_bars(means, network_names, variance_type_labels, variance_colors, output_path):
    """
    Create stacked bar chart of variance decomposition per network.
    
    Parameters
    ----------
    means : np.ndarray (n_networks, n_variance_types)
    network_names : list of str
    variance_type_labels : list of str
        e.g., ['Shared', 'Motor', 'Scene']
    variance_colors : list of str
        Hex colors for each variance type
    output_path : Path
    """
    n_networks = len(network_names)
    n_types = len(variance_type_labels)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create stacked bars
    x = np.arange(n_networks)
    bar_width = 0.7
    
    bottom = np.zeros(n_networks)
    for i in range(n_types):
        ax.bar(x, means[:, i], bar_width, bottom=bottom, 
               label=variance_type_labels[i], color=variance_colors[i],
               edgecolor='white', linewidth=1)
        bottom += means[:, i]
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(network_names, rotation=45, ha='right')
    ax.set_ylabel('Mean R²', fontsize=12)
    ax.set_xlabel('Functional Network', fontsize=12)
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Path Utilities
# ============================================================================

def get_results_directory(subject, train_sessions, test_sessions, experiment_id):
    """Construct path to variance partitioning results directory."""
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    results_dir = PATHS['variance_partitioning'] / dataset_dir / experiment_id
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    return results_dir


def get_parcellation_path(parcellation_name):
    """Get path to network parcellation file."""
    if parcellation_name == 'yeo':
        return PATHS['Yeo7_networks']
    elif parcellation_name == 'cole-anticevic':
        return PATHS['ColeAnticevic_networks']
    else:
        raise ValueError(f"Unknown parcellation: {parcellation_name}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot variance decomposition across functional networks'
    )
    
    # Data source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--subject', type=int,
                             help='Subject ID for individual analysis')
    source_group.add_argument('--group-config', type=str,
                             help='Group config name from mario_encoding.group_configs')
    
    # Subject-level arguments
    parser.add_argument('--train-sessions', type=int, nargs='+',
                       help='Training sessions (required for --subject)')
    parser.add_argument('--test-sessions', type=int, nargs='+',
                       help='Test sessions (required for --subject)')
    
    # Common arguments
    parser.add_argument('--experiment-id', type=str, required=True,
                       help='Experiment ID (timestamp)')
    parser.add_argument('--parcellation', type=str, required=True,
                       choices=['yeo', 'cole-anticevic'],
                       help='Network parcellation to use')
    parser.add_argument('--r2-threshold', type=float, default=0.05,
                       help='Minimum R² threshold for including voxels (default: 0.05)')
    
    args = parser.parse_args()
    
    # Validate subject-level arguments
    if args.subject and (not args.train_sessions or not args.test_sessions):
        parser.error("--subject requires --train-sessions and --test-sessions")
    
    # ========================================================================
    # Load data and config
    # ========================================================================
    
    sns_styleset()
    if args.group_config:
        # Group-level analysis
        if not hasattr(group_configs, args.group_config):
            raise ValueError(
                f"Config '{args.group_config}' not found in mario_encoding.group_configs"
            )
        
        config = getattr(group_configs, args.group_config)
        feature_space_names = list(config['feature_spaces'].keys())
        results_dir = PATHS['variance_partitioning'] / 'group_average' / args.experiment_id
        
        print(f"Group analysis: {args.group_config}")
        print(f"Subjects: {[s['subject'] for s in config['subjects']]}")
    
    else:
        # Subject-level analysis
        results_dir = get_results_directory(
            args.subject, args.train_sessions, args.test_sessions, args.experiment_id
        )
        config, feature_space_names = load_experiment_config(results_dir)
        
        print(f"Subject: {args.subject}")
        print(f"Train: {args.train_sessions}, Test: {args.test_sessions}")
    
    print(f"Results directory: {results_dir}")
    print(f"Feature spaces: {feature_space_names}")
    print(f"Parcellation: {args.parcellation}")
    print()
    
    # ========================================================================
    # Load network parcellation
    # ========================================================================
    
    parcellation_path = get_parcellation_path(args.parcellation)
    print(f"Loading parcellation from: {parcellation_path}")
    network_labels = load_network_parcellation(parcellation_path)
    network_config = get_network_config(args.parcellation)
    print(f"Loaded {len(network_config['labels'])} networks")
    
    # ========================================================================
    # Load variance maps
    # ========================================================================
    
    mask_path = results_dir / 'valid_voxels_mask.npy'
    if mask_path.exists():
        # Subject-level: load explicit mask
        valid_voxels_mask = np.load(mask_path)
        print(f"Valid voxels: {valid_voxels_mask.sum()} / {len(valid_voxels_mask)}")
    else:
        # Group-level: create mask from R2_full (load temporarily to get shape)
        r2_path = results_dir / 'R2_scores.npz'
        temp_data = np.load(r2_path)
        r2_full_values = temp_data['R2_full']
        n_grayordinates = 91282  # Expected for HCP surface space
        
        # All voxels are "valid" for group average - create full mask
        valid_voxels_mask = np.ones(n_grayordinates, dtype=bool)
        print(f"Group-level: using all {n_grayordinates} grayordinates")
    
    print("Loading variance maps...")
    variance_maps = load_variance_maps(results_dir, valid_voxels_mask, feature_space_names)
    print(f"Loaded: {list(variance_maps.keys())}")
    
    # ========================================================================
    # Compute network statistics
    # ========================================================================
    
    # Focus on shared, motor, scene
    variance_types = ['shared', 'unique_motor', 'unique_scene']
    variance_type_labels_list = ['Shared', 'Motor', 'Scene']
    variance_type_labels_dict = {
        'shared': 'Shared',
        'unique_motor': 'Motor', 
        'unique_scene': 'Scene'
    }
    variance_color_map = get_variance_colors()
    variance_colors_list = [variance_color_map[vt] for vt in ['shared', 'motor', 'scene']]
    variance_colors_dict = {
        'shared': variance_color_map['shared'],
        'unique_motor': variance_color_map['motor'],
        'unique_scene': variance_color_map['scene']
    }
    
    print(f"Computing network means (R² > {args.r2_threshold})...")
    means, network_names_ordered = compute_network_means(
        variance_maps, network_labels, network_config,
        valid_voxels_mask, variance_types, args.r2_threshold
    )
    
    # ========================================================================
    # Create grouped bar plot
    # ========================================================================
    
    output_filename = f'variance_by_network_{args.parcellation}_grouped.svg'
    output_path = results_dir / output_filename
    print(f"Creating grouped bar plot: {output_path}")
    
    plot_grouped_bars(
        means, network_names_ordered, variance_type_labels_list,
        variance_colors_list, output_path
    )
    
    # ========================================================================
    # Create ridge plots
    # ========================================================================
    
    print(f"Preparing ridge plot data...")
    violin_data, _ = prepare_violin_data(
        variance_maps, network_labels, network_config,
        valid_voxels_mask, variance_types, args.r2_threshold
    )

    # Overlapping ridge plot (motor + scene on same axis)
    output_filename = f'variance_by_network_{args.parcellation}_ridges_overlapping.svg'
    output_path = results_dir / output_filename
    print(f"Creating overlapping ridge plot: {output_path}")
    
    plot_ridge_distributions_overlapping(
        violin_data, network_names_ordered, network_config,
        variance_colors_dict, output_path
    )
    
    # Shared variance ridge plot
    output_filename = f'variance_by_network_{args.parcellation}_ridges_shared.svg'
    output_path = results_dir / output_filename
    print(f"Creating shared variance ridge plot: {output_path}")
    
    plot_ridge_distributions_shared(
        violin_data, network_names_ordered, network_config,
        variance_colors_dict, output_path
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
