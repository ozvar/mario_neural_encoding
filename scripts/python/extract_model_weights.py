"""
Extract model weights from fitted voxelwise encoding models.

This script loads a fitted model, extracts primal coefficients, reshapes by delays,
averages across delays, and rescales by prediction accuracy following Huth et al. (2012).
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from mario_encoding.config import PATHS


def load_model_and_metadata(model_path, cv_scores_path, feature_names_path, logger_func=print):
    """
    Load fitted model, CV scores, and feature names.
    
    Parameters:
    -----------
    model_path : Path
    cv_scores_path : Path
    feature_names_path : Path
    logger_func : callable
        
    Returns:
    --------
    pipeline : fitted sklearn Pipeline
    cv_scores : array of shape (n_voxels,)
    feature_names : list of str
    """
    logger_func(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    logger_func(f"Loading CV scores from: {cv_scores_path}")
    cv_scores = np.load(cv_scores_path)

    logger_func(f"  CV scores shape: {cv_scores.shape}")
    logger_func(f"  Mean CV R²: {cv_scores.mean():.4f}")
    logger_func(f"Loading feature names from: {feature_names_path}")
    with open(feature_names_path, 'r') as f:
        feature_metadata = json.load(f)
    
    feature_names = feature_metadata['kept_features']
    logger_func(f"  Kept features: {len(feature_names)}")
    
    return pipeline, cv_scores, feature_names


def extract_primal_weights(pipeline, logger_func=print):
    """
    Extract primal coefficients from KernelRidgeCV.
    
    Parameters:
    -----------
    pipeline : fitted sklearn Pipeline with Delayer -> KernelRidgeCV
    logger_func : callable
        
    Returns:
    --------
    primal_coef : array of shape (n_features * n_delays, n_voxels)
    """
    logger_func("Extracting primal coefficients from model...")
    # Get primal weights from kernel ridge
    primal_coef = pipeline[-1].get_primal_coef()
    # Convert to numpy if needed
    if hasattr(primal_coef, 'cpu'):
        primal_coef = primal_coef.cpu().numpy()
    
    logger_func(f"  Primal coefficient shape: {primal_coef.shape}")
    
    return primal_coef


def rescale_weights_by_scores(primal_coef, cv_scores, logger_func=print):
    """
    Rescale regression coefficients by prediction accuracy.
    
    Following Huth et al. (2012): normalize weights to unit norm, then scale by sqrt(R²).
    This addresses the bias introduced by per-voxel regularization.
    
    Parameters:
    -----------
    primal_coef : array of shape (n_features * n_delays, n_voxels)
    cv_scores : array of shape (n_voxels,)
    logger_func : callable
        
    Returns:
    --------
    rescaled_coef : array of shape (n_features * n_delays, n_voxels)
    """
    logger_func("Rescaling weights by prediction accuracy...")
    # Normalize to unit norm
    norms = np.linalg.norm(primal_coef, axis=0)
    primal_coef = primal_coef / (norms[None, :] + 1e-10)
    # Scale by sqrt(max(0, R²))
    scale_factors = np.sqrt(np.maximum(0, cv_scores))
    rescaled_coef = primal_coef * scale_factors[None, :]
    
    logger_func(f"  Weight range before rescaling: [{norms.min():.3f}, {norms.max():.3f}]")
    logger_func(f"  Weight range after rescaling: [{np.abs(rescaled_coef).max():.3f}]")
    
    return rescaled_coef


def reshape_and_average_delays(primal_coef, pipeline, logger_func=print):
    """
    Reshape weights by delays and average across delays.
    
    Parameters:
    -----------
    primal_coef : array of shape (n_features * n_delays, n_voxels)
    pipeline : fitted Pipeline with Delayer step
    logger_func : callable
        
    Returns:
    --------
    average_coef : array of shape (n_features, n_voxels)
    primal_coef_per_delay : array of shape (n_delays, n_features, n_voxels)
    """
    logger_func("Reshaping by delays...")
    # Get Delayer from pipeline
    delayer = pipeline.named_steps['delayer']
    logger_func(f"  Delays used: {delayer.delays}")
    # Reshape to (n_delays, n_features, n_voxels)
    primal_coef_per_delay = delayer.reshape_by_delays(primal_coef, axis=0)
    logger_func(f"  Reshaped to: {primal_coef_per_delay.shape}")
    # Average across delays
    average_coef = primal_coef_per_delay.mean(axis=0)
    logger_func(f"  Averaged to: {average_coef.shape}")
    
    return average_coef, primal_coef_per_delay


def save_weights(output_path, average_coef, primal_coef_per_delay, feature_names, logger_func=print):
    """
    Save extracted weights and metadata.
    
    Parameters:
    -----------
    output_path : Path
        Base path for output files (without extension)
    average_coef : array of shape (n_features, n_voxels)
    primal_coef_per_delay : array of shape (n_delays, n_features, n_voxels)
    feature_names : list of str
    logger_func : callable
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save averaged weights (for PCA)
    weights_file = output_path.with_name(output_path.stem + '_weights.npy')
    np.save(weights_file, average_coef)
    logger_func(f"Saved averaged weights to: {weights_file}")
    # Save per-delay weights (for HRF analysis)
    weights_per_delay_file = output_path.with_name(output_path.stem + '_weights_per_delay.npy')
    np.save(weights_per_delay_file, primal_coef_per_delay)
    logger_func(f"Saved per-delay weights to: {weights_per_delay_file}")
    # Save feature names
    feature_names_file = output_path.with_name(output_path.stem + '_weights_features.json')
    with open(feature_names_file, 'w') as f:
        json.dump({'features': feature_names, 'n_features': len(feature_names)}, f, indent=2)
    logger_func(f"Saved feature names to: {feature_names_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Extract model weights from fitted encoding model')
    parser.add_argument('--subject', type=int, required=True, help='Subject number')
    parser.add_argument('--train-sessions', type=int, nargs=2, metavar=('START', 'END'),
                       required=True, help='Training session range (e.g., 7 7)')
    parser.add_argument('--test-sessions', type=int, nargs=2, metavar=('START', 'END'),
                       required=True, help='Test session range (e.g., 20 20)')
    
    args = parser.parse_args()
    
    # Construct file paths
    train_str = f"{args.train_sessions[0]:03d}-{args.train_sessions[1]:03d}"
    test_str = f"{args.test_sessions[0]:03d}-{args.test_sessions[1]:03d}"
    base_name = f'sub-{args.subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    
    model_path = PATHS['models'] / f'{base_name}_model.pkl'
    cv_scores_path = PATHS['cv_scores'] / f'{base_name}_cv_scores.npy'
    feature_names_path = PATHS['cv_scores'] / f'{base_name}_feature_names.json'
    
    # Validate files exist
    for path in [model_path, cv_scores_path, feature_names_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    
    print("="*80)
    print("EXTRACTING MODEL WEIGHTS")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Model: {base_name}")
    print()
    
    # Load model and metadata
    pipeline, cv_scores, feature_names = load_model_and_metadata(
        model_path, cv_scores_path, feature_names_path
    )
    print()
    
    # Extract primal weights
    primal_coef = extract_primal_weights(pipeline)
    print()
    
    # Rescale by prediction accuracy
    primal_coef = rescale_weights_by_scores(primal_coef, cv_scores)
    print()
    
    # Reshape by delays and average
    average_coef, primal_coef_per_delay = reshape_and_average_delays(primal_coef, pipeline)
    print()
    
    # Save results
    print("="*80)
    print("SAVING WEIGHTS")
    print("="*80)
    output_path = PATHS['pca'] / base_name
    save_weights(output_path, average_coef, primal_coef_per_delay, feature_names)
    print()
    
    print("="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
