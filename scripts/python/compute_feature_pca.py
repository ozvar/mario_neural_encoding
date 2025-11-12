"""
Compute PCA on model weights to identify feature space structure.

This script performs PCA on the weight matrix (features × voxels), following
Huth et al. (2012) methodology to identify low-dimensional feature representations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from mario_encoding.config import PATHS


def load_weights_and_features(weights_path, features_path, logger_func=print):
    """
    Load averaged model weights and feature names.
    
    Parameters:
    -----------
    weights_path : Path
    features_path : Path
    logger_func : callable
        
    Returns:
    --------
    weights : array of shape (n_features, n_voxels)
    feature_names : list of str
    """
    logger_func(f"Loading weights from: {weights_path}")
    weights = np.load(weights_path)
    logger_func(f"  Weights shape: {weights.shape}")
    
    logger_func(f"Loading feature names from: {features_path}")
    with open(features_path, 'r') as f:
        feature_metadata = json.load(f)
    
    feature_names = feature_metadata['features']
    logger_func(f"  Feature names: {len(feature_names)}")
    
    # Validate shapes match
    if weights.shape[0] != len(feature_names):
        raise ValueError(f"Weight shape {weights.shape} doesn't match {len(feature_names)} features")
    
    return weights, feature_names


def compute_pca(weights, n_components=None, logger_func=print):
    """
    Perform PCA on weight matrix.
    
    Treats voxels as samples and features as variables, following Huth et al. (2012).
    
    Parameters:
    -----------
    weights : array of shape (n_features, n_voxels)
    n_components : int or None
        Number of components to compute. If None, compute all components.
    logger_func : callable
        
    Returns:
    --------
    pca : fitted PCA object
    voxel_projections : array of shape (n_voxels, n_components)
        Projection of each voxel onto PCs
    """
    logger_func("Computing PCA on weight matrix...")
    logger_func(f"  Input: {weights.shape[0]} features × {weights.shape[1]} voxels")
    
    if n_components is None:
        n_components = min(weights.shape)
    # PCA: voxels as samples, features as variables
    # Transpose so shape is (n_voxels, n_features)
    pca = PCA(n_components=n_components)
    voxel_projections = pca.fit_transform(weights.T)
    
    logger_func(f"  Computed {pca.n_components_} principal components")
    logger_func(f"  Explained variance (first 10 PCs): {pca.explained_variance_ratio_[:10]}")
    logger_func(f"  Cumulative variance (first 10 PCs): {pca.explained_variance_ratio_[:10].cumsum()[-1]:.3f}")
    
    return pca, voxel_projections


def create_scree_plot(pca, output_path, logger_func=print):
    """
    Create scree plot showing explained variance per PC.
    
    Parameters:
    -----------
    pca : fitted PCA object
    output_path : Path
    logger_func : callable
    """
    logger_func("Creating scree plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot explained variance
    n_show = min(20, len(pca.explained_variance_ratio_))
    ax1.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show])
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot')
    ax1.grid(True, alpha=0.3)
    # Plot cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, n_show + 1), cumvar[:n_show], marker='o')
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger_func(f"Saved scree plot to: {output_path}")


def create_factor_loadings_table(pca, feature_names, n_components=10, logger_func=print):
    """
    Create DataFrame with PC loadings for each feature.
    
    Parameters:
    -----------
    pca : fitted PCA object
    feature_names : list of str
    n_components : int
        Number of PCs to include in table
    logger_func : callable
        
    Returns:
    --------
    loadings_df : pd.DataFrame
        Columns: feature, PC1, PC2, ..., PCn
    """
    logger_func(f"Creating factor loadings table for {n_components} components...")
    
    n_components = min(n_components, pca.n_components_)
    
    # PC loadings are the components_ (n_components, n_features)
    # Transpose to (n_features, n_components)
    loadings = pca.components_[:n_components].T
    
    # Create DataFrame
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    loadings_df = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=pc_columns
    )
    loadings_df.index.name = 'feature'
    loadings_df = loadings_df.reset_index()
    
    # Add magnitude column (L2 norm across PCs)
    loadings_df['magnitude'] = np.linalg.norm(loadings, axis=1)
    
    logger_func(f"  Loadings table shape: {loadings_df.shape}")
    
    return loadings_df


def save_pca_results(output_path, pca, voxel_projections, loadings_df, logger_func=print):
    """
    Save PCA results to disk.
    
    Parameters:
    -----------
    output_path : Path
        Base path for output files (without extension)
    pca : fitted PCA object
    voxel_projections : array of shape (n_voxels, n_components)
    loadings_df : pd.DataFrame
    logger_func : callable
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save PCA object (for reconstruction)
    pca_file = output_path.with_name(output_path.stem + '_pca.pkl')
    import pickle
    with open(pca_file, 'wb') as f:
        pickle.dump(pca, f)
    logger_func(f"Saved PCA object to: {pca_file}")
    # Save voxel projections (for CIFTI visualization)
    projections_file = output_path.with_name(output_path.stem + '_voxel_projections.npy')
    np.save(projections_file, voxel_projections)
    logger_func(f"Saved voxel projections to: {projections_file}")
    # Save explained variance
    variance_file = output_path.with_name(output_path.stem + '_explained_variance.npy')
    np.save(variance_file, pca.explained_variance_ratio_)
    logger_func(f"Saved explained variance to: {variance_file}")
    # Save factor loadings table
    loadings_csv = output_path.with_name(output_path.stem + '_factor_loadings.csv')
    loadings_df.to_csv(loadings_csv, index=False, float_format='%.6f')
    logger_func(f"Saved factor loadings to: {loadings_csv}")
    # Save top features per PC
    top_features_file = output_path.with_name(output_path.stem + '_top_features_per_pc.txt')
    with open(top_features_file, 'w') as f:
        n_show = min(10, len(loadings_df))
        for i in range(min(5, pca.n_components_)):
            pc_col = f'PC{i+1}'
            f.write(f"\n{pc_col} (explains {pca.explained_variance_ratio_[i]:.3%} variance)\n")
            f.write("="*60 + "\n")
            # Top positive loadings
            top_pos = loadings_df.nlargest(n_show, pc_col)
            f.write("\nTop positive loadings:\n")
            for _, row in top_pos.iterrows():
                f.write(f"  {row['feature']:30s} {row[pc_col]:7.4f}\n")
            # Top negative loadings
            top_neg = loadings_df.nsmallest(n_show, pc_col)
            f.write("\nTop negative loadings:\n")
            for _, row in top_neg.iterrows():
                f.write(f"  {row['feature']:30s} {row[pc_col]:7.4f}\n")
    
    logger_func(f"Saved top features per PC to: {top_features_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Compute PCA on model weights')
    parser.add_argument('--subject', type=int, required=True, help='Subject number')
    parser.add_argument('--train-sessions', type=int, nargs=2, metavar=('START', 'END'),
                       required=True, help='Training session range (e.g., 7 7)')
    parser.add_argument('--test-sessions', type=int, nargs=2, metavar=('START', 'END'),
                       required=True, help='Test session range (e.g., 20 20)')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCs to compute (default: all)')
    parser.add_argument('--n-loadings', type=int, default=20,
                       help='Number of PCs to include in loadings table (default: 10)')
    
    args = parser.parse_args()
    
    # Construct file paths
    train_str = f"{args.train_sessions[0]:03d}-{args.train_sessions[1]:03d}"
    test_str = f"{args.test_sessions[0]:03d}-{args.test_sessions[1]:03d}"
    base_name = f'sub-{args.subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    
    weights_path = PATHS['pca'] / f'{base_name}_weights.npy'
    features_path = PATHS['pca'] / f'{base_name}_weights_features.json'
    
    # Validate files exist
    for path in [weights_path, features_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}\nRun extract_model_weights.py first.")
    
    print("="*80)
    print("COMPUTING PCA ON MODEL WEIGHTS")
    print("="*80)
    print(f"Subject: {args.subject}")
    print(f"Model: {base_name}")
    print()
    
    # Load weights and features
    weights, feature_names = load_weights_and_features(weights_path, features_path)
    print()
    
    # Compute PCA
    pca, voxel_projections = compute_pca(weights, n_components=args.n_components)
    print()
    
    # Create scree plot
    print("="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    scree_path = PATHS['pca'] / f'{base_name}_scree_plot.png'
    create_scree_plot(pca, scree_path)
    print()
    
    # Create factor loadings table
    loadings_df = create_factor_loadings_table(pca, feature_names, n_components=args.n_loadings)
    print()
    
    # Save results
    print("="*80)
    print("SAVING PCA RESULTS")
    print("="*80)
    output_path = PATHS['pca'] / base_name
    save_pca_results(output_path, pca, voxel_projections, loadings_df)
    print()
    
    # Summary statistics
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total components: {pca.n_components_}")
    print(f"Variance explained by first 5 PCs: {pca.explained_variance_ratio_[:5].sum():.3%}")
    print(f"Variance explained by first 10 PCs: {pca.explained_variance_ratio_[:10].sum():.3%}")
    
    # Find n_components for 90% variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.argmax(cumvar >= 0.9) + 1
    print(f"Components needed for 90% variance: {n_90}")
    print()
    
    print("="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
