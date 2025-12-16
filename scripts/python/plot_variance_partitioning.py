"""
Visualize variance partitioning results: R2 distribution and variance component breakdown.
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from mario_encoding.utils import sns_styleset, save_figure


def load_config(experiment_dir: Path) -> Dict:
    """Load experiment configuration."""
    config_file = experiment_dir / 'config.json'
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {experiment_dir}")
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def load_variance_data(experiment_dir: Path, feature_spaces: list) -> Dict[str, np.ndarray]:
    """
    Load R2 scores from variance partitioning results.
    
    Parameters:
    -----------
    experiment_dir : Path
        Directory containing R2_scores.npz file
    feature_spaces : list
        List of feature space names from config
        
    Returns:
    --------
    r2_data : dict
        Dictionary with R2_full, R2_unique_* for each space, R2_shared (if available)
    """
    r2_file = experiment_dir / 'R2_scores.npz'
    if not r2_file.exists():
        raise FileNotFoundError(f"R2_scores.npz not found in {experiment_dir}")
    
    data = np.load(r2_file)
    r2_data = {'R2_full': data['R2_full']}
    
    # Load unique variance for each feature space
    for space in feature_spaces:
        key = f'R2_unique_{space}'
        if key in data:
            r2_data[key] = data[key]
        else:
            print(f"Warning: {key} not found in R2_scores.npz")
    
    # Load shared variance if available
    if 'R2_shared' in data:
        r2_data['R2_shared'] = data['R2_shared']
    
    return r2_data


def compute_shared_variance(r2_data: Dict[str, np.ndarray], feature_spaces: list) -> np.ndarray:
    """
    Compute shared variance across all feature spaces.
    
    Shared = R2_full - sum(unique variances)
    
    Parameters:
    -----------
    r2_data : dict
        Must contain R2_full and R2_unique_* arrays
    feature_spaces : list
        List of feature space names
        
    Returns:
    --------
    R2_shared : array
        Shared variance per voxel
    """
    sum_unique = np.zeros_like(r2_data['R2_full'])
    for space in feature_spaces:
        key = f'R2_unique_{space}'
        if key in r2_data:
            sum_unique += r2_data[key]
    
    R2_shared = r2_data['R2_full'] - sum_unique
    return R2_shared


def add_clean_end_ticks(ax=None, y_margin=0.05):
   # Get current axes if none specified
   if ax is None:
       ax = plt.gca()
   # Get current axis limits and data ranges
   ymin, ymax = ax.get_ylim()
   # Function to round up to next nice number
   def round_up_to_nice(x, margin=0.1):
       # Add margin to max value
       x_with_margin = x * (1 + margin)
       # Get magnitude
       magnitude = 10 ** np.floor(np.log10(x_with_margin))
       # Round up to next nice number at this magnitude
       normalized = np.ceil(x_with_margin / magnitude)
       return normalized * magnitude
   # Calculate nice end value for y-axis only
   y_end = round_up_to_nice(ymax, y_margin)
   # Set new y limit only
   ax.set_ylim(ymin, y_end)
   # Generate evenly spaced ticks for y-axis only
   ax.yaxis.set_major_locator(plt.AutoLocator())


def plot_full_r2_histogram(r2_full: np.ndarray, output_path: Path, color: str) -> None:
    """
    Create histogram of R2_full distribution across voxels.
    
    Parameters:
    -----------
    r2_full : array
        R2 values per voxel
    output_path : Path
        Where to save figure
    color : str
        Hex color code for bars
    """
    # Filter NaN values
    r2_valid = r2_full[~np.isnan(r2_full)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(r2_valid, bins=50, color=color, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('R$^2$')
    ax.set_ylabel('Number of Voxels')
    
    # Add summary stats as text
    mean_r2 = r2_valid.mean()
    median_r2 = np.median(r2_valid)
    textstr = f'Mean: {mean_r2:.3f}\nMedian: {median_r2:.3f}'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            fontsize=24, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


def plot_shared_r2_histogram(r2_shared: np.ndarray, output_path: Path, color: str) -> None:
    """
    Create histogram of R2_shared distribution across voxels.
    
    Parameters:
    -----------
    r2_shared : array
        R2 values per voxel
    output_path : Path
        Where to save figure
    color : str
        Hex color code for bars
    """
    # Filter NaN values
    r2_valid = r2_shared[~np.isnan(r2_shared)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(r2_valid, bins=50, color=color, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('R$^2$')
    ax.set_ylabel('Number of Voxels')
    
    # Add summary stats as text
    mean_r2 = r2_valid.mean()
    median_r2 = np.median(r2_valid)
    textstr = f'Mean: {mean_r2:.3f}\nMedian: {median_r2:.3f}'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            fontsize=24, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


def plot_overlaid_r2_histograms(r2_full: np.ndarray, r2_shared: np.ndarray, 
                                output_path: Path, 
                                color_full: str, color_shared: str) -> None:
    """
    Create overlaid histograms of R2_full and R2_shared distributions.
    
    Parameters:
    -----------
    r2_full : array
        Full model R2 values per voxel
    r2_shared : array
        Shared variance R2 values per voxel
    output_path : Path
        Where to save figure
    color_full : str
        Hex color code for full R2 bars
    color_shared : str
        Hex color code for shared R2 bars
    """
    # Filter NaN values
    r2_full_valid = r2_full[~np.isnan(r2_full)]
    r2_shared_valid = r2_shared[~np.isnan(r2_shared)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine shared bin edges for both histograms
    bins = 50
    bin_range = (min(r2_full_valid.min(), r2_shared_valid.min()), 
                 max(r2_full_valid.max(), r2_shared_valid.max()))
    
    # Plot overlaid histograms with transparency
    ax.hist(r2_full_valid, bins=bins, range=bin_range, color=color_full, 
            alpha=0.6, edgecolor='white', linewidth=0.5, label='Full model')
    ax.hist(r2_shared_valid, bins=bins, range=bin_range, color=color_shared, 
            alpha=0.6, edgecolor='white', linewidth=0.5, label='Shared variance')
    
    # Add vertical lines for means
    mean_full = r2_full_valid.mean()
    mean_shared = r2_shared_valid.mean()
    ax.axvline(mean_full, color=color_full, linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(mean_shared, color=color_shared, linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('R$^2$')
    ax.set_ylabel('Number of Voxels')
    
    # Create legend with mean values
    legend_labels = [
        f'Full model',
        f'Shared variance'
    ]
    ax.legend(legend_labels, frameon=False, bbox_to_anchor=(0.98, 0.98), loc='upper right')
    
    add_clean_end_ticks(ax, y_margin=0.05)
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


def plot_full_vs_shared_kde(r2_full: np.ndarray, r2_shared: np.ndarray,
                            output_path: Path, 
                            color_full: str = '#8c8c8c',
                            color_shared: str = '#c44e52',
                            alpha_full: float = 0.5,
                            alpha_shared: float = 0.7,
                            bw_method: str = 'scott') -> None:
    """
    Create KDE plot comparing R2_full and R2_shared distributions.
    
    Parameters:
    -----------
    r2_full : array
        Full model R2 values per voxel
    r2_shared : array
        Shared variance R2 values per voxel
    output_path : Path
        Where to save figure
    color_full : str
        Hex color code for full R2 curve (default gray)
    color_shared : str
        Hex color code for shared R2 curve (default red)
    alpha_full : float
        Transparency for full R2 fill (default 0.5)
    alpha_shared : float
        Transparency for shared R2 fill (default 0.7)
    bw_method : str or float
        Bandwidth selection method for KDE (default 'scott')
    """
    from scipy import stats
    
    # Filter out NaN values
    r2_full_valid = r2_full[~np.isnan(r2_full)]
    r2_shared_valid = r2_shared[~np.isnan(r2_shared)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute KDEs
    kde_full = stats.gaussian_kde(r2_full_valid, bw_method=bw_method)
    kde_shared = stats.gaussian_kde(r2_shared_valid, bw_method=bw_method)
    
    # Create evaluation points spanning the full range
    x_min = min(r2_full_valid.min(), r2_shared_valid.min())
    x_max = max(r2_full_valid.max(), r2_shared_valid.max())
    x_eval = np.linspace(x_min, x_max, 500)
    
    # Evaluate KDEs
    density_full = kde_full(x_eval)
    density_shared = kde_shared(x_eval)
    
    # Plot filled KDEs
    ax.fill_between(x_eval, density_full, alpha=alpha_full, 
                     color=color_full, label='Full model')
    ax.fill_between(x_eval, density_shared, alpha=alpha_shared, 
                     color=color_shared, label='Shared variance')
    
    # Labels and styling
    ax.set_xlabel('R$^2$')
    ax.set_ylabel('Density')
    ax.legend(frameon=False, loc='upper right')
    
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


def plot_variance_components(r2_data: Dict[str, np.ndarray], R2_shared: np.ndarray,
                             feature_spaces: list, output_path: Path, 
                             colors: Dict[str, str]) -> None:
    """
    Create bar chart showing mean variance explained by each component.
    
    Parameters:
    -----------
    r2_data : dict
        Contains R2_unique_* arrays for each feature space
    R2_shared : array
        Shared variance per voxel
    feature_spaces : list
        List of feature space names (preserves order)
    output_path : Path
        Where to save figure
    colors : dict
        Maps component names to hex color codes
    """
    # Build components dict in order: shared, then feature spaces (use nanmean)
    components = {'Shared': np.nanmean(R2_shared)}
    for space in feature_spaces:
        key = f'R2_unique_{space}'
        if key in r2_data:
            components[space.capitalize()] = np.nanmean(r2_data[key])
    
    labels = list(components.keys())
    values = list(components.values())
    bar_colors = [colors.get(label, '#888888') for label in labels]  # Default gray if not in colors
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color=bar_colors, edgecolor='white', linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel('Mean R$^2$')
    ax.set_title('Variance Decomposition')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=20)
    
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


def plot_variance_pie_chart(r2_data: Dict[str, np.ndarray], R2_shared: np.ndarray,
                           output_path: Path,
                           colors: Dict[str, str]) -> None:
    """
    Create pie chart showing proportion of total variance by component.
    
    Lumps actions and activity unique variance with shared.
    
    Parameters:
    -----------
    r2_data : dict
        Contains R2_full and R2_unique_* arrays
    R2_shared : array
        Shared variance per voxel
    output_path : Path
        Where to save figure
    colors : dict
        Maps component names to hex color codes
    """
    # Calculate mean R² for each component (filter NaN)
    mean_shared = np.nanmean(R2_shared)
    mean_motor = np.nanmean(r2_data.get('R2_unique_motor', np.array([0.0])))
    mean_scene = np.nanmean(r2_data.get('R2_unique_scene', np.array([0.0])))
    mean_actions = np.nanmean(r2_data.get('R2_unique_actions', np.array([0.0])))
    mean_activity = np.nanmean(r2_data.get('R2_unique_activity', np.array([0.0])))
    
    # Lump shared + actions + activity
    mean_shared_lumped = mean_shared + mean_actions + mean_activity
    
    # Components for pie chart
    labels = ['Shared', 'Motor unique', 'Scene unique']
    sizes = [mean_shared_lumped, mean_motor, mean_scene]
    pie_colors = ['#aa52c7', '#188bb7', '#ec8e48']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create pie chart
    wedges, texts = ax.pie(
        sizes, 
        labels=None,
        colors=pie_colors,
        autopct=None,
        startangle=90,
        textprops={'fontsize': 18},
        wedgeprops={'edgecolor': 'black', 'linewidth': 4}
    )
    
    # Make percentage text bold
    # for autotext in autotexts:
    #     autotext.set_color('white')
    #     autotext.set_fontweight('bold')
    #     autotext.set_fontsize(20)
    
    ax.axis('equal')
    
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


def visualize_variance_partitioning(experiment_dir, 
                                   color_palette: list) -> None:
    """
    Generate all variance partitioning visualizations for one experiment.
    
    Creates:
    1. Histogram of R2_full distribution
    2. Bar chart of variance components
    
    Parameters:
    -----------
    experiment_dir : Path or str
        Directory containing R2_scores.npz and config.json
    color_palette : list
        List of hex color codes (at least 6 colors)
    """
    experiment_dir = Path(experiment_dir)
    print(f"Loading data from {experiment_dir}...")
    
    # Load config to get feature spaces
    config = load_config(experiment_dir)
    feature_spaces = list(config['feature_spaces'].keys())
    print(f"Feature spaces from config: {feature_spaces}")
    
    # Load variance data
    r2_data = load_variance_data(experiment_dir, feature_spaces)
    
    # Compute or load shared variance
    if 'R2_shared' in r2_data:
        print("Using pre-computed shared variance...")
        R2_shared = r2_data['R2_shared']
    else:
        print("Computing shared variance...")
        R2_shared = compute_shared_variance(r2_data, feature_spaces)
    
    # Print summary stats (filter NaN for means)
    print(f"  Mean shared variance: {np.nanmean(R2_shared):.6f}")
    for space in feature_spaces:
        key = f'R2_unique_{space}'
        if key in r2_data:
            print(f"  Mean unique {space}: {np.nanmean(r2_data[key]):.6f}")
    
    # Map colors to components (capitalize to match bar chart labels)
    colors = {
        'Shared': color_palette[1],      # gray
        'Motor': color_palette[3],       # yellow
        'Actions': color_palette[5],     # orange
        'Activity': color_palette[2],    # brown
        'Scene': color_palette[4]        # blue
    }
    
    # Create histograms
    print("Creating full model R2 histogram...")
    hist_path = experiment_dir / 'R2_full_histogram.png'
    plot_full_r2_histogram(r2_data['R2_full'], hist_path, color_palette[0])  # deep red

    print("Creating shared variance R2 histogram...")
    hist_path = experiment_dir / 'R2_shared_histogram.png'
    plot_shared_r2_histogram(r2_data['R2_shared'], hist_path, color_palette[1])  # gray
    print(f"  Saved to {hist_path}")
    
    print("Creating overlaid R2 histograms...")
    overlaid_path = experiment_dir / 'R2_overlaid_histogram.png'
    plot_overlaid_r2_histograms(
        r2_data['R2_full'], R2_shared, overlaid_path,
        color_palette[0], color_palette[1]  # red and gray
    )
    print(f"  Saved to {overlaid_path}")

    # Create KDE plot
    print("Creating full vs shared KDE plot...")
    kde_path = experiment_dir / 'R2_full_vs_shared_kde.png'
    plot_full_vs_shared_kde(
        r2_data['R2_full'], R2_shared, kde_path,
        color_full=color_palette[1],    # gray
        color_shared=color_palette[0]   # red
    )
    print(f"  Saved to {kde_path}")

    # Create bar chart
    print("Creating variance components bar chart...")
    bar_path = experiment_dir / 'variance_components.png'
    plot_variance_components(r2_data, R2_shared, feature_spaces, bar_path, colors)
    print(f"  Saved to {bar_path}")
    
    # Create pie chart
    print("Creating variance components pie chart...")
    pie_path = experiment_dir / 'variance_components_pie.png'
    plot_variance_pie_chart(r2_data, R2_shared, pie_path, colors)
    print(f"  Saved to {pie_path}")


if __name__ == '__main__':
    # Configure plotting style
    sns_styleset()
    
    # Color palette
    color_palette = [
        '#8609b0',  # gray/purple (full variance)
        '#54355e',  # purple (shared variance)
        '#188bb7',  # blue (motor variance)
        '#ec8e48',  # orange (scene variance)
        '#4c72b0',  # blue
        '#dd8452'   # orange
    ]
    
    # Example usage - replace with your actual experiment directory
    experiment_dir = '/home/ozvar/Git/cneuromod/mario_neural_encoding/results/variance_partitioning/group_average/20251208_134311'
    
    visualize_variance_partitioning(experiment_dir, color_palette)
