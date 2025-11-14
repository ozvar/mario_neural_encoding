"""
Diagnostic checks for variance partitioning weights.
Run in REPL or as script.
"""
import numpy as np
import json
from pathlib import Path

try:
    from mario_encoding.config import FEATURE_SPACES
    SPACES = list(FEATURE_SPACES.keys())
except ImportError:
    print("Warning: Could not import FEATURE_SPACES from mario_encoding.config")
    print("Using default feature spaces")
    SPACES = ['perception', 'motor', 'action', 'scene', 'activity']

# CONFIGURATION
VP_DIR = Path('/home/ozvar/Git/cneuromod/mario_neural_encoding/results/variance_partitioning/sub-01_train-ses-007-012_test-ses-014-015')


def load_vp_results(vp_dir):
    """Load all variance partitioning results."""
    print("="*80)
    print("LOADING VARIANCE PARTITIONING RESULTS")
    print("="*80)
    
    # Load metadata
    with open(vp_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load R2 scores
    r2_data = np.load(vp_dir / 'R2_scores.npz')
    R2 = {key: r2_data[key] for key in r2_data.files}
    
    # Load weights for each space
    weights = {}
    masks = {}
    
    for space in SPACES:
        try:
            weights[space] = np.load(vp_dir / f'{space}_weights_selective.npy')
            masks[space] = np.load(vp_dir / f'{space}_selective_mask.npy')
            with open(vp_dir / f'{space}_weights_metadata.json', 'r') as f:
                weights[f'{space}_meta'] = json.load(f)
        except FileNotFoundError:
            print(f"  Warning: {space} weights not found")
            weights[space] = None
            masks[space] = None
    
    print(f"\nLoaded data for subject {metadata['subject']}")
    print(f"Train sessions: {metadata['train_sessions']}")
    print(f"Test sessions: {metadata['test_sessions']}")
    print(f"Voxels kept: {metadata['n_voxels_kept']}")
    print(f"Features kept: {metadata['n_features_kept']}")
    
    return metadata, R2, weights, masks


def check_r2_distributions(R2):
    """Check R2 score distributions."""
    print("\n" + "="*80)
    print("R2 SCORE DISTRIBUTIONS")
    print("="*80)
    
    print("\nFull model R2:")
    print(f"  Mean: {R2['R2_full'].mean():.6f}")
    print(f"  Median: {np.median(R2['R2_full']):.6f}")
    print(f"  Max: {R2['R2_full'].max():.6f}")
    print(f"  % positive: {(R2['R2_full'] > 0).sum() / len(R2['R2_full']) * 100:.1f}%")
    print(f"  % > 0.01: {(R2['R2_full'] > 0.01).sum() / len(R2['R2_full']) * 100:.1f}%")
    print(f"  % > 0.1: {(R2['R2_full'] > 0.1).sum() / len(R2['R2_full']) * 100:.1f}%")
    
    print("\nUnique R2 by feature space:")
    for space in SPACES:
        key = f'R2_unique_{space}'
        if key in R2:
            r2_unique = R2[key]
            print(f"\n  {space}:")
            print(f"    Mean: {r2_unique.mean():.6f}")
            print(f"    Median: {np.median(r2_unique):.6f}")
            print(f"    Max: {r2_unique.max():.6f}")
            print(f"    Min: {r2_unique.min():.6f}")
            print(f"    % positive: {(r2_unique > 0).sum() / len(r2_unique) * 100:.1f}%")
            print(f"    % > 0.01: {(r2_unique > 0.01).sum() / len(r2_unique) * 100:.1f}%")
            print(f"    % > 0.1: {(r2_unique > 0.1).sum() / len(r2_unique) * 100:.1f}%")
            print(f"    % negative: {(r2_unique < 0).sum() / len(r2_unique) * 100:.1f}%")


def check_unique_variance_sum(R2):
    """Check if sum of unique variances is reasonable."""
    print("\n" + "="*80)
    print("VARIANCE DECOMPOSITION CHECK")
    print("="*80)
    
    unique_keys = [f'R2_unique_{s}' for s in SPACES if f'R2_unique_{s}' in R2]
    
    sum_unique = np.sum([R2[key] for key in unique_keys], axis=0)
    
    print(f"\nR2_full mean: {R2['R2_full'].mean():.6f}")
    print(f"Sum(R2_unique) mean: {sum_unique.mean():.6f}")
    print(f"Ratio sum/full: {sum_unique.mean() / (R2['R2_full'].mean() + 1e-10):.3f}")
    
    print("\nVoxel-wise comparison:")
    n_sum_exceeds = (sum_unique > R2['R2_full']).sum()
    print(f"  Voxels where sum(unique) > R2_full: {n_sum_exceeds} ({n_sum_exceeds/len(sum_unique)*100:.1f}%)")


def check_weight_magnitudes(weights, masks):
    """Check weight magnitude distributions."""
    print("\n" + "="*80)
    print("WEIGHT MAGNITUDE DISTRIBUTIONS")
    print("="*80)
    
    for space in SPACES:
        if weights[space] is None:
            continue
            
        w = weights[space]
        mask = masks[space]
        
        print(f"\n{space}:")
        print(f"  Selective voxels: {mask.sum()} ({mask.sum()/len(mask)*100:.1f}%)")
        
        if w.size > 0:
            print(f"  Weight array shape: {w.shape}")
            print(f"  Weight magnitudes:")
            print(f"    Mean |w|: {np.abs(w).mean():.6f}")
            print(f"    Median |w|: {np.median(np.abs(w)):.6f}")
            print(f"    Max |w|: {np.abs(w).max():.6f}")
            print(f"    % zeros: {(w == 0).sum() / w.size * 100:.1f}%")
            
            small_threshold = 1e-6
            print(f"    % < {small_threshold}: {(np.abs(w) < small_threshold).sum() / w.size * 100:.1f}%")


def check_feature_space_sizes(metadata, weights):
    """Check how many features per space and per delay."""
    print("\n" + "="*80)
    print("FEATURE SPACE SIZES")
    print("="*80)
    
    for space in SPACES:
        key = f'{space}_meta'
        if key in weights and weights[key] is not None:
            meta = weights[key]
            print(f"\n{space}:")
            print(f"  Base features: {len(meta['feature_names'])}")
            print(f"  Features: {meta['feature_names']}")
            if 'n_features_with_delays' in meta:
                print(f"  After delays: {meta['n_features_with_delays']}")
            if 'delays' in meta:
                print(f"  Delays: {meta['delays']}")


def check_selective_overlap(masks):
    """Check overlap between selective masks."""
    print("\n" + "="*80)
    print("SELECTIVE VOXEL OVERLAP")
    print("="*80)
    
    available_spaces = [s for s in SPACES if masks[s] is not None]
    
    print("\nPairwise overlaps:")
    for i, space1 in enumerate(available_spaces):
        for space2 in available_spaces[i+1:]:
            overlap = (masks[space1] & masks[space2]).sum()
            union = (masks[space1] | masks[space2]).sum()
            jaccard = overlap / union if union > 0 else 0
            print(f"  {space1} & {space2}: {overlap} voxels (Jaccard: {jaccard:.3f})")
    
    n_spaces_per_voxel = np.sum([masks[s] for s in available_spaces], axis=0)
    print("\nVoxels by number of selective spaces:")
    for n in range(1, len(available_spaces)+1):
        count = (n_spaces_per_voxel == n).sum()
        print(f"  Selective for {n} space(s): {count} voxels")


def check_well_predicted_voxels(R2, masks, threshold=0.1):
    """Check selective masks among well-predicted voxels."""
    print("\n" + "="*80)
    print(f"SELECTIVITY IN WELL-PREDICTED VOXELS (R2 > {threshold})")
    print("="*80)
    
    well_predicted = R2['R2_full'] > threshold
    n_well = well_predicted.sum()
    
    print(f"\nWell-predicted voxels: {n_well} ({n_well/len(well_predicted)*100:.1f}%)")
    
    for space in SPACES:
        if masks[space] is None:
            continue
        
        selective_and_well = masks[space] & well_predicted
        
        print(f"\n{space}:")
        print(f"  Selective & well-predicted: {selective_and_well.sum()}")
        print(f"  % of well-predicted that are selective: {selective_and_well.sum()/n_well*100:.1f}%")
        print(f"  % of selective that are well-predicted: {selective_and_well.sum()/masks[space].sum()*100:.1f}%")


def check_activity_features_specifically(weights, R2):
    """Specific diagnostics for activity features."""
    print("\n" + "="*80)
    print("ACTIVITY FEATURE DIAGNOSTICS")
    print("="*80)
    
    if weights['activity'] is None:
        print("No activity weights found")
        return
    
    w_activity = weights['activity']
    r2_unique_activity = R2['R2_unique_activity']
    
    # Get feature names from metadata
    feature_names = weights['activity_meta']['feature_names']
    
    print(f"\nActivity-selective voxels: {w_activity.shape[1]}")
    print(f"Activity features: {w_activity.shape[0]}")
    
    print("\nWeight magnitude per feature (averaged across voxels):")
    mean_abs_weights = np.abs(w_activity).mean(axis=1)
    for i, mag in enumerate(mean_abs_weights):
        feat_name = feature_names[i] if i < len(feature_names) else f"Feature {i}"
        print(f"  {feat_name}: {mag:.6f}")
    
    print("\nFeature importance (sum of absolute weights):")
    feature_importance = np.abs(w_activity).sum(axis=1)
    feature_importance_norm = feature_importance / feature_importance.sum()
    
    # Sort by importance for better readability
    sorted_indices = np.argsort(feature_importance_norm)[::-1]
    for idx in sorted_indices:
        feat_name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
        print(f"  {feat_name}: {feature_importance_norm[idx]:.3f}")
    
    top_n = 10
    top_indices = np.argsort(r2_unique_activity)[-top_n:][::-1]
    print(f"\nTop {top_n} voxels by activity unique R2:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Voxel {idx}: R2_unique = {r2_unique_activity[idx]:.6f}, R2_full = {R2['R2_full'][idx]:.6f}")


def run_all_diagnostics(vp_dir=VP_DIR):
    """Run all diagnostic checks."""
    metadata, R2, weights, masks = load_vp_results(vp_dir)
    
    check_r2_distributions(R2)
    check_unique_variance_sum(R2)
    check_weight_magnitudes(weights, masks)
    check_feature_space_sizes(metadata, weights)
    check_selective_overlap(masks)
    check_well_predicted_voxels(R2, masks, threshold=0.1)
    check_activity_features_specifically(weights, R2)
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    
    return metadata, R2, weights, masks


if __name__ == '__main__':
    metadata, R2, weights, masks = run_all_diagnostics()
