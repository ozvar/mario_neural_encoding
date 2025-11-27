"""
Variance partitioning utilities for banded ridge regression.

Implements variance decomposition methods to quantify unique contributions
of different feature spaces to voxelwise encoding model predictions.
"""
import numpy as np
from sklearn.model_selection import check_cv
from himalaya.ridge import GroupRidgeCV
from voxelwise_tutorials.utils import generate_leave_one_run_out


def map_features_to_spaces(feature_names, valid_features_mask, feature_spaces_dict, logger):
    """
    Map surviving features (after filtering) to their feature spaces.
    
    NOTE: At this point, feature_names should already be a subset matching 
    FEATURE_SPACES (via select_and_validate_features). This function only
    handles the zero-variance filtering that happened after selection.
    
    Parameters:
    -----------
    feature_names : list of str
        Selected features (from config) before zero-variance filtering
    valid_features_mask : boolean array
        Mask indicating which selected features survived zero-variance filtering
    feature_spaces_dict : dict
        Keys are space names, values are lists of feature names in each space
    logger : logging.Logger
        
    Returns:
    --------
    feature_spaces_filtered : dict
        Keys are space names, values are lists of surviving feature names
    space_to_indices : dict
        Keys are space names, values are arrays of column indices in filtered X
    """
    logger.info("")
    logger.info("="*80)
    logger.info("FEATURE SPACE MAPPING (after zero-variance filtering)")
    logger.info("="*80)
    
    # Get list of kept features
    kept_features = [name for name, keep in zip(feature_names, valid_features_mask) if keep]
    
    feature_spaces_filtered = {}
    space_to_indices = {}
    
    for space_name in sorted(feature_spaces_dict.keys()):
        space_features = feature_spaces_dict[space_name]
        surviving_indices = []
        surviving_names = []
        dropped_names = []
        
        for feat in space_features:
            if feat in kept_features:
                idx = kept_features.index(feat)
                surviving_indices.append(idx)
                surviving_names.append(feat)
            elif feat in feature_names:  # Was selected but filtered out
                dropped_names.append(feat)
            # If feat not in feature_names, it was already caught in selection step
        
        feature_spaces_filtered[space_name] = surviving_names
        space_to_indices[space_name] = np.array(surviving_indices)
        
        logger.info(f"  {space_name}: {len(surviving_names)}/{len(space_features)} features survived")
        if dropped_names:
            logger.info(f"    Dropped (zero variance): {dropped_names}")
    
    logger.info("="*80)
    
    return feature_spaces_filtered, space_to_indices


def validate_all_spaces_nonempty(feature_spaces_filtered, logger):
    """
    Validate that all feature spaces have at least one surviving feature.
    
    Parameters:
    -----------
    feature_spaces_filtered : dict
        Keys are space names, values are lists of surviving feature names
    logger : logging.Logger
        
    Raises:
    -------
    ValueError if any feature space has zero features
    """
    logger.info("Validating feature spaces...")
    
    empty_spaces = [name for name, features in feature_spaces_filtered.items() if len(features) == 0]
    
    if len(empty_spaces) > 0:
        error_msg = f"Feature spaces with zero surviving features: {empty_spaces}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("  All feature spaces have >0 features ")


def expand_feature_space_indices_for_delays(space_to_indices, n_features_original, delays, logger):
    """
    Expand feature space indices to account for FIR delays.
    
    After applying Delayer, each feature is replicated across delays:
    [feat1, feat2, ..., featN] -> [feat1_d1, feat1_d2, ..., featN_d4]
    
    This function expands the original feature space indices to reference
    all delayed copies of features in each space.
    
    Parameters:
    -----------
    space_to_indices : dict
        Keys are space names, values are arrays of original feature indices
    n_features_original : int
        Number of features before applying delays
    delays : list of int
        FIR delay values
    logger : logging.Logger
        
    Returns:
    --------
    space_to_indices_delayed : dict
        Keys are space names, values are arrays of delayed feature indices
    """
    logger.info("Expanding feature space indices for delays...")
    logger.info(f"  Original features: {n_features_original}")
    logger.info(f"  Delays: {delays}")
    logger.info(f"  Total delayed features: {n_features_original * len(delays)}")
    
    space_to_indices_delayed = {}
    
    for space_name, original_indices in space_to_indices.items():
        delayed_indices = []
        
        for delay_idx in range(len(delays)):
            offset = delay_idx * n_features_original
            delayed_indices.extend((original_indices + offset).tolist())
        
        space_to_indices_delayed[space_name] = np.array(delayed_indices)
        
        logger.info(f"  {space_name}: {len(original_indices)} -> {len(delayed_indices)} features")
    
    return space_to_indices_delayed


def create_feature_space_arrays(X_delayed, space_to_indices_delayed, feature_spaces_filtered, logger):
    """
    Split delayed feature matrix into list of arrays for GroupRidgeCV.
    
    Parameters:
    -----------
    X_delayed : array of shape (n_samples, n_features_delayed)
        Feature matrix after applying Delayer
    space_to_indices_delayed : dict
        Keys are space names, values are arrays of delayed feature indices
    feature_spaces_filtered : dict
        Keys are space names (used to maintain consistent ordering)
    logger : logging.Logger
        
    Returns:
    --------
    X_list : list of arrays
        One array per feature space, in consistent order
    space_names_ordered : list of str
        Feature space names in the order they appear in X_list
    """
    logger.info("Creating feature space arrays for GroupRidgeCV...")
    
    # Use sorted order for consistency
    space_names_ordered = sorted(feature_spaces_filtered.keys())
    X_list = []
    
    for space_name in space_names_ordered:
        indices = space_to_indices_delayed[space_name]
        X_space = X_delayed[:, indices]
        X_list.append(X_space)
        logger.info(f"  {space_name}: {X_space.shape}")
    
    return X_list, space_names_ordered


def fit_full_model(X_train_list, Y_train, run_onsets_train, params, backend, logger):
    """
    Fit full GroupRidgeCV model with all feature spaces.
    
    Parameters:
    -----------
    X_train_list : list of arrays
        One array per feature space
    Y_train : array of shape (n_samples, n_voxels)
    run_onsets_train : array of int
    params : dict
        'solver', 'solver_params' from config
    backend : str
        'torch_cuda', 'numpy', etc.
    logger : logging.Logger
        
    Returns:
    --------
    model : fitted GroupRidgeCV
    """
    logger.info("Fitting full model with all feature spaces...")
    
    # Create CV splitter
    cv = generate_leave_one_run_out(Y_train.shape[0], run_onsets_train)
    cv = check_cv(cv)
    logger.info(f"  Using leave-one-run-out CV with {cv.get_n_splits()} folds")
    
    # Create model
    model = GroupRidgeCV(
        groups="input",  # X_train_list signals automatic grouping
        solver=params['solver'],
        solver_params=params['solver_params'],
        cv=cv,
        Y_in_cpu=False,
        force_cpu=False
    )
    
    logger.info(f"  Feature spaces: {len(X_train_list)}")
    logger.info(f"  Total features: {sum(x.shape[1] for x in X_train_list)}")
    logger.info(f"  Training samples: {Y_train.shape[0]}")
    logger.info(f"  Voxels: {Y_train.shape[1]}")
    
    # Fit
    model.fit(X_train_list, Y_train)
    logger.info("  Model fitting complete ")
    
    return model


def fit_restricted_model(X_train_list, Y_train, excluded_space_name, space_names_ordered, 
                        run_onsets_train, params, backend, logger):
    """
    Fit GroupRidgeCV model excluding one feature space.
    
    Parameters:
    -----------
    X_train_list : list of arrays
        One array per feature space (full list)
    Y_train : array of shape (n_samples, n_voxels)
    excluded_space_name : str
        Name of feature space to exclude
    space_names_ordered : list of str
        Ordered list of space names (to find index to exclude)
    run_onsets_train : array of int
    params : dict
    backend : str
    logger : logging.Logger
        
    Returns:
    --------
    model : fitted GroupRidgeCV
    """
    logger.info(f"Fitting restricted model (excluding {excluded_space_name})...")
    
    # Find index to exclude
    excluded_idx = space_names_ordered.index(excluded_space_name)
    
    # Create restricted feature list
    X_train_restricted = [X_train_list[i] for i in range(len(X_train_list)) if i != excluded_idx]
    
    logger.info(f"  Excluded space index: {excluded_idx}")
    logger.info(f"  Remaining feature spaces: {len(X_train_restricted)}")
    logger.info(f"  Total features: {sum(x.shape[1] for x in X_train_restricted)}")
    
    # Create CV splitter
    cv = generate_leave_one_run_out(Y_train.shape[0], run_onsets_train)
    cv = check_cv(cv)
    
    # Create and fit model
    model = GroupRidgeCV(
        groups="input",
        solver=params['solver'],
        solver_params=params['solver_params'],
        cv=cv,
        Y_in_cpu=False,
        force_cpu=False
    )
    
    model.fit(X_train_restricted, Y_train)
    logger.info("  Model fitting complete ")
    
    return model


def compute_unique_variance(R2_full, R2_restricted_dict, logger):
    """
    Compute unique variance explained by each feature space.
    
    Unique variance for space X = R2_full - R2_without_X
    
    Parameters:
    -----------
    R2_full : array of shape (n_voxels,)
        Test R2 scores from full model
    R2_restricted_dict : dict
        Keys are space names, values are test R2 scores from models without that space
    logger : logging.Logger
        
    Returns:
    --------
    R2_unique : dict
        Keys are space names, values are unique variance arrays
    """
    logger.info("Computing unique variance for each feature space...")
    
    R2_unique = {}
    
    for space_name, R2_restricted in R2_restricted_dict.items():
        R2_unique[space_name] = R2_full - R2_restricted
        
        mean_unique = R2_unique[space_name].mean()
        median_unique = np.median(R2_unique[space_name])
        pct_positive = (R2_unique[space_name] > 0).sum() / len(R2_unique[space_name]) * 100
        
        logger.info(f"  {space_name}:")
        logger.info(f"    Mean unique R2: {mean_unique:.6f}")
        logger.info(f"    Median unique R2: {median_unique:.6f}")
        logger.info(f"    % voxels with positive unique variance: {pct_positive:.1f}%")
    
    return R2_unique


def compute_shared_variance(R2_full, R2_unique_dict, logger):
    """
    Compute shared variance across all feature spaces.
    
    Shared variance = R2_full - sum(unique variances)
    
    Parameters:
    -----------
    R2_full : array of shape (n_voxels,)
        Test R2 scores from full model
    R2_unique_dict : dict
        Keys are space names, values are unique variance arrays
    logger : logging.Logger
        
    Returns:
    --------
    R2_shared : array of shape (n_voxels,)
        Shared variance per voxel
    """
    logger.info("Computing shared variance across all feature spaces...")
    sum_unique = np.sum([v for v in R2_unique_dict.values()], axis=0)
    R2_shared = R2_full - sum_unique
    mean_shared = R2_shared.mean()
    median_shared = np.median(R2_shared)
    pct_positive = (R2_shared > 0).sum() / len(R2_shared) * 100
    logger.info(f"  Mean shared R2: {mean_shared:.6f}")
    logger.info(f"  Median shared R2: {median_shared:.6f}")
    logger.info(f"  % voxels with positive shared variance: {pct_positive:.1f}%")
    logger.info(f"  Ratio shared/total: {mean_shared / R2_full.mean():.3f}")

    return R2_shared


def compute_segregation_index(R2_full, R2_unique_dict, logger):
    """
    Compute segregation index as proportion of unique variance.
    
    Segregation index = sum(unique variances) / R2_full
    Values near 0: Low segregation (mostly shared variance)
    Values near 1: High segregation (mostly unique variance)
    
    Parameters:
    -----------
    R2_full : array of shape (n_voxels,)
        Test R2 scores from full model
    R2_unique_dict : dict
        Keys are space names, values are unique variance arrays
    logger : logging.Logger
        
    Returns:
    --------
    segregation_index : array of shape (n_voxels,)
        Segregation index per voxel (0 to 1+, can exceed 1 due to negative unique)
    """
    logger.info("Computing segregation index...")
    sum_unique = np.sum([v for v in R2_unique_dict.values()], axis=0)
    segregation_index = sum_unique / (R2_full + 1e-10)
    mean_seg = segregation_index.mean()
    median_seg = np.median(segregation_index)
    pct_low_seg = (segregation_index < 0.3).sum() / len(segregation_index) * 100
    pct_high_seg = (segregation_index > 0.7).sum() / len(segregation_index) * 100
    logger.info(f"  Mean segregation index: {mean_seg:.6f}")
    logger.info(f"  Median segregation index: {median_seg:.6f}")
    logger.info(f"  % voxels with low segregation (<0.3): {pct_low_seg:.1f}%")
    logger.info(f"  % voxels with high segregation (>0.7): {pct_high_seg:.1f}%")

    return segregation_index


def compute_product_measure(model_full, X_test_list, Y_test, space_names_ordered, logger):
    """
    Compute product measure for each feature space from joint model.
    
    Product measure decomposes R^2 into contributions from each feature space:
    R_tilde_j = sum(y_pred_j * (2*y - y_pred_full)) / sum(y^2)
    
    where y_pred_j is the sub-prediction from feature space j using the joint model weights.
    
    Parameters:
    -----------
    model_full : fitted GroupRidgeCV
        Full model with all feature spaces
    X_test_list : list of arrays
        Test features, one array per feature space
    Y_test : array of shape (n_samples, n_voxels)
        Test targets
    space_names_ordered : list of str
        Feature space names in order matching X_test_list
    logger : logging.Logger
        
    Returns:
    --------
    product_measures : dict
        Keys are space names, values are product measure arrays (n_voxels,)
    """
    logger.info("Computing product measure for each feature space...")
    
    # Get full model prediction
    Y_pred_full = model_full.predict(X_test_list)
    if hasattr(Y_pred_full, 'cpu'):
        Y_pred_full = Y_pred_full.cpu().numpy()
    
    # Get weights from model - shape depends on delays
    # model_full.coef_ has shape (n_targets, n_features_total) or (n_targets, n_features_per_group, n_groups)
    weights = model_full.coef_
    if hasattr(weights, 'cpu'):
        weights = weights.cpu().numpy()
    
    # Compute sub-predictions for each feature space
    product_measures = {}
    
    # Determine if we need to handle grouped weights structure
    if weights.ndim == 3:
        # Shape: (n_voxels, n_features_per_delay, n_groups)
        # Need to index by group
        for idx, space_name in enumerate(space_names_ordered):
            X_space = X_test_list[idx]
            weights_space = weights[:, :, idx]  # (n_voxels, n_features_for_this_space)
            
            # Compute sub-prediction: X_space @ weights_space.T
            Y_pred_space = X_space @ weights_space.T  # (n_samples, n_voxels)

            # Compute product measure per voxel
            numerator = np.sum(Y_pred_space * (2 * Y_test - Y_pred_full), axis=0)  # (n_voxels,)
            denominator = np.sum(Y_test ** 2, axis=0)  # (n_voxels,)

            product_measures[space_name] = numerator / np.maximum(denominator, 1.0)
            
            mean_pm = product_measures[space_name].mean()
            n_negative = (product_measures[space_name] < 0).sum()
            pct_negative = n_negative / len(product_measures[space_name]) * 100
            
            logger.info(f"  {space_name}: mean={mean_pm:.6f}, {n_negative} negative ({pct_negative:.1f}%)")
    else:
        # Fallback: weights shape (n_voxels, n_features_total)
        # Need to manually slice by cumulative feature counts
        feature_start = 0
        for idx, space_name in enumerate(space_names_ordered):
            X_space = X_test_list[idx]
            n_features_space = X_space.shape[1]
            weights_space = weights[feature_start:feature_start + n_features_space, :]
            
            Y_pred_space = X_space @ weights_space

            numerator = np.sum(Y_pred_space * (2 * Y_test - Y_pred_full), axis=0)
            denominator = np.sum(Y_test ** 2, axis=0)
            
            product_measures[space_name] = numerator / np.maximum(denominator, 1.0)
            
            mean_pm = product_measures[space_name].mean()
            n_negative = (product_measures[space_name] < 0).sum()
            pct_negative = n_negative / len(product_measures[space_name]) * 100
            
            logger.info(f"  {space_name}: mean={mean_pm:.6f}, {n_negative} negative ({pct_negative:.1f}%)")
            
            feature_start += n_features_space
    
    # Verify decomposition: sum should equal R^2
    sum_product = np.sum([v for v in product_measures.values()], axis=0)
    R2_full = model_full.score(X_test_list, Y_test)
    if hasattr(R2_full, 'cpu'):
        R2_full = R2_full.cpu().numpy()
    
    logger.info(f"  Sum of product measures: mean={sum_product.mean():.6f}")
    logger.info(f"  R2_full: mean={R2_full.mean():.6f}")
    logger.info(f"  Difference: mean={np.abs(sum_product - R2_full).mean():.8f}")
    
    return product_measures


def fisher_z_transform(R2_values):
    """
    Apply Fisher z-transform to R^2 values.
    
    Transform: z = 0.5 * log((1 + r) / (1 - r))
    where r = sqrt(R^2) (taking positive root)
    
    Parameters:
    -----------
    R2_values : array or dict of arrays
        R^2 values to transform. Can be single array or dict of arrays.
        
    Returns:
    --------
    fisher_z : array or dict of arrays (same structure as input)
        Fisher z-transformed values
    """
    def transform_single(R2):
        # Clip R2 to valid range [0, 1)
        R2_clipped = np.clip(R2, 0, 0.9999)
        # Convert to correlation coefficient
        r = np.sqrt(R2_clipped)
        # Apply Fisher z transform
        z = np.arctanh(r)
        return z
    
    if isinstance(R2_values, dict):
        return {key: transform_single(val) for key, val in R2_values.items()}
    else:
        return transform_single(R2_values)


def validate_variance_partition(R2_full, R2_unique, logger):
    """
    Sanity checks on variance partitioning results.
    
    Parameters:
    -----------
    R2_full : array of shape (n_voxels,)
    R2_unique : dict
        Keys are space names, values are unique variance arrays
    logger : logging.Logger
    """
    logger.info("Validating variance partitioning results...")
    
    # 1. Check that sum of unique variances is reasonable
    sum_unique = np.sum([v for v in R2_unique.values()], axis=0)
    
    logger.info("  Relationship between sum(unique) and R2_full:")
    logger.info(f"    Mean R2_full: {R2_full.mean():.6f}")
    logger.info(f"    Mean sum(unique): {sum_unique.mean():.6f}")
    logger.info(f"    Ratio sum(unique)/R2_full: {(sum_unique.mean() / R2_full.mean()):.3f}")
    
    # Sum of unique can exceed R2_full due to overlapping explained variance
    # But it should be within reasonable bounds
    ratio = sum_unique.mean() / (R2_full.mean() + 1e-10)
    if ratio > 5.0:
        logger.warning(f"   Sum of unique variances is {ratio:.1f}x larger than R2_full - may indicate issues")
    elif ratio < 0.2:
        logger.warning(f"   Sum of unique variances is only {ratio:.1f}x R2_full - features may be highly redundant")
    else:
        logger.info(f"   Variance decomposition appears reasonable")
    
    # 2. Check for negative unique variances (expected in some voxels)
    for space_name, unique_variance in R2_unique.items():
        n_negative = (unique_variance < 0).sum()
        pct_negative = n_negative / len(unique_variance) * 100
        logger.info(f"  {space_name}: {n_negative} voxels ({pct_negative:.1f}%) with negative unique variance")
        
        if pct_negative > 50:
            logger.warning(f"     >50% negative values may indicate this space adds little unique information")


def validate_product_measure(product_measures, R2_full, logger):
    """
    Sanity checks on product measure results.
    
    Parameters:
    -----------
    product_measures : dict
        Keys are space names, values are product measure arrays
    R2_full : array of shape (n_voxels,)
    logger : logging.Logger
    """
    logger.info("Validating product measure results...")
    
    # 1. Check that sum equals R2_full (should be exact by construction)
    sum_product = np.sum([v for v in product_measures.values()], axis=0)
    
    logger.info("  Decomposition property (sum should equal R2_full):")
    logger.info(f"    Mean R2_full: {R2_full.mean():.6f}")
    logger.info(f"    Mean sum(product): {sum_product.mean():.6f}")
    logger.info(f"    Mean absolute difference: {np.abs(sum_product - R2_full).mean():.8f}")
    logger.info(f"    Max absolute difference: {np.abs(sum_product - R2_full).max():.8f}")
    
    # 2. Report negative values (expected with correlated feature spaces)
    logger.info("  Negative contributions (expected with correlated spaces):")
    for space_name, pm_values in product_measures.items():
        n_negative = (pm_values < 0).sum()
        pct_negative = n_negative / len(pm_values) * 100
        logger.info(f"    {space_name}: {n_negative} voxels ({pct_negative:.1f}%)")
        
        if pct_negative > 80:
            logger.warning(f"      >80% negative may indicate this space is primarily used for orthogonalization")
