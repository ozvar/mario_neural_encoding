"""
Extract feature space weights from variance partitioning full model.

This script loads the full GroupRidgeCV model from variance partitioning,
extracts weights for a specified feature space, handles delays, and prepares
data for PCA analysis on selective voxels.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from mario_encoding.config import PATHS, FEATURE_SPACES


def load_variance_partitioning_results(output_dir, logger_func=print):
    """
    Load variance partitioning results.
    
    Parameters:
    -----------
    output_dir : Path
        Directory containing variance partitioning results
    logger_func : callable
        
    Returns:
    --------
    model_full : fitted GroupRidgeCV
    R2_scores : dict with R2_full, R2_unique_*, etc.
    metadata : dict
    """
    logger_func(f"Loading variance partitioning results from: {output_dir}")
    
    # Load full model
    model_path = output_dir / 'model_full.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Full model not found: {model_path}")
    
    logger_func(f"  Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_full = pickle.load(f)
    
    # Load R² scores
    r2_path = output_dir / 'R2_scores.npz'
    if not r2_path.exists():
        raise FileNotFoundError(f"R² scores not found: {r2_path}")
    
    logger_func(f"  Loading R² scores from: {r2_path}")
    R2_data = np.load(r2_path)
    R2_scores = {key: R2_data[key] for key in R2_data.files}
    
    # Load metadata
    metadata_path = output_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    logger_func(f"  Loading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger_func(f"  Model type: {type(model_full)}")
    logger_func(f"  R² arrays: {list(R2_scores.keys())}")
    logger_func(f"  Voxels: {metadata['n_voxels_kept']}")
    logger_func(f"  Features: {metadata['n_features_kept']}")
    
    return model_full, R2_scores, metadata


def extract_group_coefficients(model, logger_func=print):
    """
    Extract coefficients from GroupRidgeCV.
    
    GroupRidgeCV stores coef_ attribute with shape (n_features_total, n_targets).
    Coefficients are concatenated across groups in the order groups were provided.
    
    Parameters:
    -----------
    model : fitted GroupRidgeCV
    logger_func : callable
        
    Returns:
    --------
    coef : array of shape (n_features_total, n_voxels)
    """
    logger_func("Extracting coefficients from GroupRidgeCV...")
    
    if not hasattr(model, 'coef_'):
        raise AttributeError("Model does not have coef_ attribute. Was it fitted?")
    
    coef = model.coef_
    
    # Convert to numpy if needed (torch tensor)
    if hasattr(coef, 'cpu'):
        coef = coef.cpu().numpy()
    elif hasattr(coef, 'to_numpy'):
        coef = coef.to_numpy()
    
    logger_func(f"  Coefficient shape: {coef.shape}")
    logger_func(f"  Coefficient dtype: {coef.dtype}")
    
    return coef


def identify_feature_space_indices(metadata, feature_space_name, n_delays, logger_func=print):
    """
    Identify which indices in the delayed feature matrix correspond to specified feature space.
    
    Parameters:
    -----------
    metadata : dict
        Contains feature_spaces with all features
    feature_space_name : str
        Name of feature space (e.g., 'activity', 'scene')
    n_delays : int
        Number of FIR delays
    logger_func : callable
        
    Returns:
    --------
    feature_indices : array of int
        Indices of features in the delayed coefficient matrix
    feature_names : list of str
        Names of features in this space
    """
    logger_func(f"Identifying {feature_space_name} feature indices...")
    
    # Get features from metadata
    space_features = metadata['feature_spaces'][feature_space_name]
    n_features = len(space_features)
    
    logger_func(f"  {feature_space_name} features: {n_features}")
    logger_func(f"  Feature names: {space_features}")
    
    # Get all kept features to find positions
    all_kept_features = metadata['kept_feature_names']
    
    # Find positions of features in the kept features list
    feature_positions = []
    for feat in space_features:
        if feat in all_kept_features:
            feature_positions.append(all_kept_features.index(feat))
        else:
            logger_func(f"  WARNING: Feature '{feat}' not in kept features")
    
    feature_positions = np.array(feature_positions)
    logger_func(f"  Feature positions in original X: {feature_positions}")
    
    # Expand for delays
    # After Delayer: [feat1_d1, feat1_d2, feat1_d3, feat1_d4, feat2_d1, ...]
    # Total features after delays
    n_features_original = len(all_kept_features)
    
    feature_indices_delayed = []
    for delay_idx in range(n_delays):
        offset = delay_idx * n_features_original
        feature_indices_delayed.extend((feature_positions + offset).tolist())
    
    feature_indices_delayed = np.array(feature_indices_delayed)
    
    logger_func(f"  Total delayed features: {n_features_original * n_delays}")
    logger_func(f"  {feature_space_name} indices after delays: {len(feature_indices_delayed)}")
    logger_func(f"  Expected: {n_features * n_delays}")
    
    return feature_indices_delayed, space_features


def extract_feature_space_coefficients(coef, feature_indices, logger_func=print):
    """
    Extract feature space coefficients from full coefficient matrix.
    
    Parameters:
    -----------
    coef : array of shape (n_features_total, n_voxels)
    feature_indices : array of int
        Indices of features from target space
    logger_func : callable
        
    Returns:
    --------
    space_coef : array of shape (n_space_features_delayed, n_voxels)
    """
    logger_func("Extracting feature space coefficients...")
    
    space_coef = coef[feature_indices, :]
    
    logger_func(f"  Feature space coefficient shape: {space_coef.shape}")
    logger_func(f"  Value range: [{space_coef.min():.4f}, {space_coef.max():.4f}]")
    
    return space_coef


def average_across_delays(space_coef, n_features, n_delays, logger_func=print):
    """
    Reshape feature space coefficients by delays and average.
    
    Parameters:
    -----------
    space_coef : array of shape (n_features * n_delays, n_voxels)
    n_features : int
    n_delays : int
    logger_func : callable
        
    Returns:
    --------
    space_coef_avg : array of shape (n_features, n_voxels)
    space_coef_per_delay : array of shape (n_delays, n_features, n_voxels)
    """
    logger_func("Averaging coefficients across delays...")
    
    n_voxels = space_coef.shape[1]
    
    # Reshape to (n_delays, n_features, n_voxels)
    space_coef_per_delay = space_coef.reshape(n_delays, n_features, n_voxels)
    
    # Average across delays
    space_coef_avg = space_coef_per_delay.mean(axis=0)
    
    logger_func(f"  Reshaped to: {space_coef_per_delay.shape}")
    logger_func(f"  Averaged to: {space_coef_avg.shape}")
    
    return space_coef_avg, space_coef_per_delay


def identify_selective_voxels(R2_scores, feature_space_name, threshold=0.01, 
                             threshold_type='unique', logger_func=print):
    """
    Identify voxels above threshold for specified feature space.
    
    Parameters:
    -----------
    R2_scores : dict
        Contains R2_full and R2_unique_* for each feature space
    feature_space_name : str
        Name of feature space
    threshold : float
        Minimum R^2 to consider
    threshold_type : str
        Type of threshold: 'unique', 'full', or 'none'
        - 'unique': Use R2_unique_{feature_space_name} > threshold
        - 'full': Use R2_full > threshold
        - 'none': Include all voxels (no thresholding)
    logger_func : callable
        
    Returns:
    --------
    selective_mask : boolean array
    """
    logger_func(f"Identifying selective voxels for {feature_space_name}...")
    logger_func(f"  Threshold type: {threshold_type}")
    logger_func(f"  Threshold value: {threshold}")
    
    if threshold_type == 'none':
        # Include all voxels
        n_voxels = len(R2_scores['R2_full'])
        selective_mask = np.ones(n_voxels, dtype=bool)
        logger_func(f"  No thresholding - using all {n_voxels} voxels")
        
    elif threshold_type == 'full':
        # Threshold on R2_full
        if 'R2_full' not in R2_scores:
            raise KeyError("R2_full not found in R^2 scores")
        
        R2_full = R2_scores['R2_full']
        selective_mask = R2_full > threshold
        n_selective = selective_mask.sum()
        pct_selective = n_selective / len(selective_mask) * 100
        
        logger_func(f"  Threshold: R2_full > {threshold}")
        logger_func(f"  Selected voxels: {n_selective} ({pct_selective:.2f}%)")
        logger_func(f"  Mean R2_full (selected voxels): {R2_full[selective_mask].mean():.6f}")
        logger_func(f"  Median R2_full (selected voxels): {np.median(R2_full[selective_mask]):.6f}")
        
    elif threshold_type == 'unique':
        # Threshold on R2_unique for this feature space
        R2_unique_key = f'R2_unique_{feature_space_name}'
        if R2_unique_key not in R2_scores:
            raise KeyError(f"{R2_unique_key} not found in R^2 scores. Available: {list(R2_scores.keys())}")
        
        R2_unique = R2_scores[R2_unique_key]
        selective_mask = R2_unique > threshold
        n_selective = selective_mask.sum()
        pct_selective = n_selective / len(selective_mask) * 100
        
        logger_func(f"  Threshold: R2_unique_{feature_space_name} > {threshold}")
        logger_func(f"  Selected voxels: {n_selective} ({pct_selective:.2f}%)")
        logger_func(f"  Mean R2_unique (selected voxels): {R2_unique[selective_mask].mean():.6f}")
        logger_func(f"  Median R2_unique (selected voxels): {np.median(R2_unique[selective_mask]):.6f}")
        
    else:
        raise ValueError(f"Invalid threshold_type: {threshold_type}. Must be 'unique', 'full', or 'none'")
    
    return selective_mask


def select_selective_weights(space_coef_avg, selective_mask, logger_func=print):
    """
    Select weights for selective voxels only.
    
    Parameters:
    -----------
    space_coef_avg : array of shape (n_features, n_voxels)
    selective_mask : boolean array
    logger_func : callable
        
    Returns:
    --------
    weights_selective : array of shape (n_features, n_selective_voxels)
    """
    logger_func("Selecting selective voxel weights...")
    
    weights_selective = space_coef_avg[:, selective_mask]
    
    logger_func(f"  Selected weights shape: {weights_selective.shape}")
    
    return weights_selective


def save_results(output_dir, feature_space_name, weights_selective, space_coef_per_delay, 
                selective_mask, feature_names, metadata, threshold_type, threshold_value, 
                logger_func=print):
    """
    Save extracted feature space weights and metadata.
    
    Parameters:
    -----------
    output_dir : Path
    feature_space_name : str
    weights_selective : array of shape (n_features, n_selective_voxels)
    space_coef_per_delay : array of shape (n_delays, n_features, n_voxels)
    selective_mask : boolean array
    feature_names : list of str
    metadata : dict (original VP metadata)
    threshold_type : str
        'unique', 'full', or 'none'
    threshold_value : float
    logger_func : callable
    """
    logger_func(f"Saving {feature_space_name} weights...")
    
    # Create filename suffix based on threshold type
    if threshold_type == 'none':
        suffix = 'all'
    else:
        suffix = f'{threshold_type}_thresh{threshold_value:.3f}'.replace('.', 'p')
    
    # Save averaged weights for selective voxels (for PCA)
    weights_file = output_dir / f'{feature_space_name}_weights_{suffix}.npy'
    np.save(weights_file, weights_selective)
    logger_func(f"  Saved selective weights to: {weights_file}")
    
    # Save per-delay weights (all voxels, for HRF analysis if needed)
    weights_per_delay_file = output_dir / f'{feature_space_name}_weights_per_delay_{suffix}.npy'
    np.save(weights_per_delay_file, space_coef_per_delay)
    logger_func(f"  Saved per-delay weights to: {weights_per_delay_file}")
    
    # Save selective mask
    mask_file = output_dir / f'{feature_space_name}_selective_mask_{suffix}.npy'
    np.save(mask_file, selective_mask)
    logger_func(f"  Saved {feature_space_name}-selective mask to: {mask_file}")
    
    # Save metadata
    space_metadata = {
        'feature_space': feature_space_name,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_selective_voxels': int(selective_mask.sum()),
        'n_total_voxels': int(len(selective_mask)),
        'pct_selective': float(selective_mask.sum() / len(selective_mask) * 100),
        'weights_shape': list(weights_selective.shape),
        'threshold_type': threshold_type,
        'threshold_value': float(threshold_value),
        'original_vp_metadata': metadata
    }
    
    metadata_file = output_dir / f'{feature_space_name}_weights_metadata_{suffix}.json'
    with open(metadata_file, 'w') as f:
        json.dump(space_metadata, f, indent=2)
    logger_func(f"  Saved metadata to: {metadata_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract feature space weights from variance partitioning full model'
    )
    parser.add_argument('--subject', type=int, required=True, help='Subject number')
    parser.add_argument('--train-sessions', type=int, nargs='+', required=True,
                       help='Training session numbers (e.g., 7 8 9)')
    parser.add_argument('--test-sessions', type=int, nargs='+', required=True,
                       help='Test session numbers (e.g., 10)')
    parser.add_argument('--feature-space', type=str, required=True,
                       choices=['perception', 'motor', 'action', 'scene', 'activity'],
                       help='Which feature space to extract')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Minimum R^2 threshold (default: 0.01)')
    parser.add_argument('--threshold-type', type=str, default='unique',
                       choices=['unique', 'full', 'none'],
                       help='Threshold type: unique (R2_unique), full (R2_full), or none (all voxels)')
    parser.add_argument('--n-delays', type=int, default=4,
                       help='Number of FIR delays used (default: 4)')
    
    args = parser.parse_args()
    
    # Construct output directory path
    train_str = f"{min(args.train_sessions):03d}-{max(args.train_sessions):03d}"
    test_str = f"{min(args.test_sessions):03d}-{max(args.test_sessions):03d}"
    dir_name = f'sub-{args.subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    output_dir = PATHS['variance_partitioning'] / dir_name
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Variance partitioning results not found: {output_dir}")
    
    print("="*80)
    print(f"EXTRACTING {args.feature_space.upper()} FEATURE WEIGHTS")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Train sessions: {args.train_sessions}")
    print(f"Test sessions: {args.test_sessions}")
    print(f"Feature space: {args.feature_space}")
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {args.threshold_type} > {args.threshold}")
    print()
    
    # Load variance partitioning results
    model_full, R2_scores, metadata = load_variance_partitioning_results(output_dir)
    print()
    
    # Extract coefficients from GroupRidgeCV
    coef = extract_group_coefficients(model_full)
    print()
    
    # Identify feature space indices
    feature_indices, feature_names = identify_feature_space_indices(
        metadata, args.feature_space, args.n_delays
    )
    print()
    
    # Extract feature space coefficients
    space_coef = extract_feature_space_coefficients(coef, feature_indices)
    print()
    
    # Average across delays
    n_features = len(feature_names)
    space_coef_avg, space_coef_per_delay = average_across_delays(
        space_coef, n_features, args.n_delays
    )
    print()
    
    # Identify selective voxels
    selective_mask = identify_selective_voxels(
        R2_scores, args.feature_space, threshold=args.threshold,
        threshold_type=args.threshold_type
    )
    print()
    
    # Select weights for selective voxels
    weights_selective = select_selective_weights(
        space_coef_avg, selective_mask
    )
    print()
    
    # Save results (filename includes threshold type)
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_results(
        output_dir, args.feature_space, weights_selective, space_coef_per_delay,
        selective_mask, feature_names, metadata, args.threshold_type, args.threshold
    )
    print()
    
    print("="*80)
    print("COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Run permutation testing for significance (optional)")
    print(f"2. Run PCA analysis on {args.feature_space} weights")


if __name__ == '__main__':
    main()
