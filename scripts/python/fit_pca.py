"""
Group PCA on joint motor+scene encoding model weights.

Pools rescaled weight matrices across subjects, applies an R2 voxel filter,
subsamples each subject to a common voxel count to equalize subject
contributions, z-scores features, fits PCA, and validates generalization
via leave-one-subject-out cross-validation. Optionally tests PC
significance against a permutation null distribution.

Voxel filtering and subsampling:
- Voxels with R2_full <= --r2-threshold (default 0.05) are dropped per
  subject. R2 is read from pca_weights.npz produced by prepare_pca_weights.py.
- After filtering, each subject is randomly subsampled (with --voxel-seed)
  to the minimum retained count across the group so all subjects contribute
  equally to the pool.
- The selected voxel indices and the resulting 91k-space mask are saved so
  null pooling can be re-applied consistently if/when null testing is rerun.

Null testing requires pre-pooled null weight matrices produced by running
prepare_pca_weights.py on each phase-randomized VP fit and pooling with
pool_null_weights.py. Pass the resulting files via --null-weight-files.
NOTE: when null fits are re-run, pool_null_weights.py must apply the same
voxel selection used here (currently a TODO; selection is saved to disk).

Usage:
    python fit_pca.py --group GROUP_N3_PRIMARY
    python fit_pca.py --group GROUP_N3_PRIMARY --r2-threshold 0.05
    python fit_pca.py --group GROUP_N3_PRIMARY \\
        --null-weight-files results/pca/null/pooled_*.npz
"""
import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

from mario_encoding.config import PATHS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY

DEFAULT_R2_THRESHOLD = 0.05
DEFAULT_VOXEL_SEED = 42

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, label: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'{label}_fit_pca.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_pca_weights_path(subject_cfg: dict) -> Path:
    subject = subject_cfg['subject']
    train_sessions = subject_cfg['train_sessions']
    test_sessions = subject_cfg['test_sessions']
    experiment_id = subject_cfg['experiment_id']
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    return PATHS['pca'] / dataset_dir / experiment_id / 'pca_weights.npz'


def load_subject_weights(
    subject_cfg: dict, logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Returns:
        weights  : (n_features, n_voxels) float32
        R2       : (n_voxels,) float -- R2_full per voxel (mask-space, same order as weights cols)
        mask     : (91282,) bool -- valid_voxels_mask for this subject
        metadata : dict
    """
    path = get_pca_weights_path(subject_cfg)
    if not path.exists():
        raise FileNotFoundError(f"PCA weights not found: {path}")

    data = np.load(path)
    weights = data['weights_joint'].astype(np.float32)
    mask = data['valid_voxels_mask']
    if 'R2_full' not in data.files:
        raise KeyError(
            f"R2_full missing from {path}. Re-run prepare_pca_weights.py "
            f"after updating it to persist R2_full alongside weights."
        )
    R2 = data['R2_full'].astype(np.float32)
    if R2.shape[0] != weights.shape[1]:
        raise ValueError(
            f"R2_full length {R2.shape[0]} does not match n_voxels {weights.shape[1]}"
        )

    meta_path = path.parent / 'pca_weights_metadata.json'
    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info(f"  sub-{subject_cfg['subject']:02d}: weights {weights.shape}, "
                f"R2 range [{R2.min():.3f}, {R2.max():.3f}], "
                f"R2>0.05: {int((R2 > 0.05).sum())} voxels")
    return weights, R2, mask, metadata


def load_group_weights(group_cfg: dict, logger: logging.Logger) -> tuple[list, list, list, list, list]:
    """
    Load weights for all subjects in the group.

    Returns:
        weights_list       : list of (n_features_i, n_voxels_i) arrays
        R2_list            : list of (n_voxels_i,) arrays (R2_full in mask-space)
        masks_list         : list of (91282,) bool arrays
        metadata_list      : list of dicts
        feature_names_list : list of feature name lists, one per subject
    """
    logger.info("Loading per-subject PCA weights...")
    weights_list, R2_list, masks_list, metadata_list, feature_names_list = [], [], [], [], []
    for sub_cfg in group_cfg['subjects']:
        w, R2, m, meta = load_subject_weights(sub_cfg, logger)
        weights_list.append(w)
        R2_list.append(R2)
        masks_list.append(m)
        metadata_list.append(meta)
        names = []
        for space_features in meta.get('feature_spaces', {}).values():
            names.extend(space_features)
        feature_names_list.append(names)
    n_total = sum(w.shape[1] for w in weights_list)
    logger.info(f"Loaded {len(weights_list)} subjects, {n_total} total voxels")
    return weights_list, R2_list, masks_list, metadata_list, feature_names_list


def align_to_common_features(
    weights_list: list[np.ndarray],
    feature_names_list: list[list[str]],
    subject_ids: list[int],
    logger: logging.Logger,
) -> tuple[list[np.ndarray], list[str], dict]:
    """
    Subset each subject's weight matrix to features present in all subjects.

    Some features may be dropped during VP fitting (e.g. zero-variance for a
    subject's sessions). This finds the intersection and re-indexes each matrix
    so all subjects share the same feature set in the same order.

    Returns:
        aligned_weights  : list of (n_common_features, n_voxels_i) arrays
        common_names     : ordered list of common feature names
        dropped_info     : dict with keys 'all_dropped' (sorted list of names
                           dropped from any subject) and 'per_subject_dropped'
                           (mapping subject_id -> features dropped from that subject)
    """
    common = set(feature_names_list[0])
    for names in feature_names_list[1:]:
        common &= set(names)

    # Preserve order from first subject
    common_names = [f for f in feature_names_list[0] if f in common]

    n_max = max(len(names) for names in feature_names_list)
    n_common = len(common_names)

    all_names = set()
    for names in feature_names_list:
        all_names |= set(names)
    all_dropped = sorted(all_names - common)

    per_subject_dropped = {}
    for sid, names in zip(subject_ids, feature_names_list):
        missing = sorted(set(common_names) - set(names))
        missing_from_subject_only = sorted(set(names) - common)
        per_subject_dropped[sid] = {
            'features_missing_in_subject': missing,
            'features_in_subject_only': missing_from_subject_only,
        }

    if n_common < n_max:
        logger.warning(
            f"Feature count mismatch across subjects: retaining {n_common}/{n_max} "
            f"common features. Dropped: {all_dropped}"
        )
        for sid, info in per_subject_dropped.items():
            if info['features_in_subject_only']:
                logger.warning(
                    f"  sub-{sid:02d}: features present only here, dropped at "
                    f"intersection: {info['features_in_subject_only']}"
                )
            if info['features_missing_in_subject']:
                logger.warning(
                    f"  sub-{sid:02d}: features missing here (forced drop from "
                    f"common set): {info['features_missing_in_subject']}"
                )
    else:
        logger.info(f"All subjects share {n_common} features -- no alignment needed")

    aligned = []
    for w, names in zip(weights_list, feature_names_list):
        idx = [names.index(f) for f in common_names]
        aligned.append(w[idx, :])

    return aligned, common_names, {
        'all_dropped': all_dropped,
        'per_subject_dropped': per_subject_dropped,
    }


# ---------------------------------------------------------------------------
# Voxel filtering and subsampling
# ---------------------------------------------------------------------------

def filter_and_subsample_voxels(
    weights_list: list[np.ndarray],
    R2_list: list[np.ndarray],
    masks_list: list[np.ndarray],
    subject_ids: list[int],
    r2_threshold: float,
    seed: int,
    logger: logging.Logger,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict]:
    """
    Drop voxels with R2 <= threshold per subject, then subsample each subject
    to the minimum retained count across the group.

    Operates on the mask-space arrays returned by load_subject_weights, where
    column index in `weights` corresponds to position in `mask.nonzero()`.

    Returns:
        weights_out     : list of (n_features, n_kept_i) arrays
        masks_out       : list of (91282,) bool arrays restricted to kept voxels
        selection_list  : list of int arrays giving the indices (into the
                          ORIGINAL mask-space weight columns) that were kept
                          for each subject -- saved to disk so null pooling
                          can apply the same selection.
        info            : dict with per-subject and group counts
    """
    rng = np.random.default_rng(seed)

    logger.info(f"Filtering voxels with R2 > {r2_threshold:.3f}...")
    pass_filter = []
    counts_passed = []
    for sid, R2 in zip(subject_ids, R2_list):
        keep = np.where(R2 > r2_threshold)[0]
        pass_filter.append(keep)
        counts_passed.append(len(keep))
        logger.info(
            f"  sub-{sid:02d}: {len(keep):>6d} / {len(R2):>6d} voxels retained "
            f"(R2 mean of kept: {R2[keep].mean():.3f}, max: {R2[keep].max():.3f})"
        )

    if any(c == 0 for c in counts_passed):
        zero_subs = [sid for sid, c in zip(subject_ids, counts_passed) if c == 0]
        raise ValueError(
            f"No voxels survived R2 > {r2_threshold} filter for sub-{zero_subs}. "
            f"Lower --r2-threshold."
        )

    target = min(counts_passed)
    logger.info(f"Subsampling each subject to min count = {target} voxels (seed={seed})")

    weights_out, masks_out, selection_list = [], [], []
    for sid, w, mask, keep in zip(subject_ids, weights_list, masks_list, pass_filter):
        if len(keep) > target:
            chosen = rng.choice(keep, size=target, replace=False)
            chosen.sort()
        else:
            chosen = keep
        selection_list.append(chosen)

        weights_out.append(w[:, chosen])

        # Build new 91282-space mask: True only at positions of selected voxels
        original_indices_91k = np.where(mask)[0]
        selected_indices_91k = original_indices_91k[chosen]
        new_mask = np.zeros_like(mask)
        new_mask[selected_indices_91k] = True
        masks_out.append(new_mask)

        logger.info(
            f"  sub-{sid:02d}: kept {len(chosen):>6d} voxels "
            f"(91k-mask sum: {int(new_mask.sum())})"
        )

    info = {
        'r2_threshold': float(r2_threshold),
        'seed': int(seed),
        'counts_passed_filter': {str(sid): int(c) for sid, c in zip(subject_ids, counts_passed)},
        'subsample_target': int(target),
    }
    return weights_out, masks_out, selection_list, info


# ---------------------------------------------------------------------------
# Per-subject demeaning
# ---------------------------------------------------------------------------

def per_subject_demean(
    weights_list: list[np.ndarray],
    subject_ids: list[int],
    logger: logging.Logger,
) -> list[np.ndarray]:
    """
    Subtract each subject's per-feature mean across its own voxels.

    Mitigates the subject-mean confound on PC scores: without this, subjects
    with more training data systematically shift weight magnitudes, and the
    leading PCs partially capture training-data quantity differences across
    subjects rather than within-subject neural organization. After this step
    each subject's block has zero column-mean by construction, so the leading
    PCs reflect only within-subject voxel-to-voxel structure.

    Input shapes: list of (n_features, n_voxels_i) arrays.
    Output: same shapes, each column-mean (across voxels) subtracted per
    subject and per feature.
    """
    logger.info("Per-subject demeaning (subtracting per-feature subject means)...")
    demeaned = []
    for sid, w in zip(subject_ids, weights_list):
        mu = w.mean(axis=1, keepdims=True)  # (n_features, 1)
        demeaned.append(w - mu)
        logger.info(
            f"  sub-{sid:02d}: per-feature mean range "
            f"[{mu.min():.4f}, {mu.max():.4f}], "
            f"abs-mean of means {np.abs(mu).mean():.4f}"
        )
    return demeaned


# ---------------------------------------------------------------------------
# Pooling and z-scoring
# ---------------------------------------------------------------------------

def pool_weights(weights_list: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """
    Concatenate per-subject weight matrices into a single group matrix.

    Returns:
        pooled     : (n_voxels_total, n_features) -- transposed for PCA convention
        boundaries : cumulative voxel counts per subject, length n_subjects+1
    """
    # Each weight array is (n_features, n_voxels); transpose to (n_voxels, n_features)
    transposed = [w.T for w in weights_list]
    pooled = np.concatenate(transposed, axis=0)
    boundaries = np.cumsum([0] + [w.shape[1] for w in weights_list])
    return pooled, boundaries


def zscore_features(matrix: np.ndarray,
                    params: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    Z-score each feature (column) across voxels (rows).

    If params is provided, apply existing mean/std rather than computing new ones.
    Returns the z-scored matrix and the params dict (mean, std per feature).
    """
    if params is None:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero for zero-variance features
        params = {'mean': mean, 'std': std}
    return (matrix - params['mean']) / params['std'], params


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def fix_pc_signs(components: np.ndarray, feature_names: list[str],
                 logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """
    Flip each PC so the feature with the highest absolute loading is positive.
    This makes 'high PC score' mean 'driven by the dominant feature of that PC'.

    Returns the sign-fixed components and a (n_components,) array of +/-1 sign
    multipliers so scores can be re-aligned consistently.
    """
    components = components.copy()
    signs = np.ones(components.shape[0], dtype=np.float32)
    for k in range(components.shape[0]):
        dominant_idx = np.argmax(np.abs(components[k]))
        if components[k, dominant_idx] < 0:
            components[k] *= -1
            signs[k] = -1
            logger.info(f"  PC{k+1}: flipped (dominant feature: {feature_names[dominant_idx]})")
        else:
            logger.info(f"  PC{k+1}: kept (dominant feature: {feature_names[dominant_idx]})")
    return components, signs


# ---------------------------------------------------------------------------
# LOSO validation
# ---------------------------------------------------------------------------

def align_fold_to_group_pcs(
    fold_components: np.ndarray,
    group_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match fold PCs to group PCs by absolute cosine similarity.

    Uses the Hungarian algorithm (linear_sum_assignment) to find the globally
    optimal one-to-one matching that maximizes total absolute similarity,
    following the approach of Lescroart & Gallant (2019) who used Stable
    Marriage for the same purpose.

    Both inputs are (n_pcs, n_features) with unit-norm rows (sklearn PCA
    convention), so the row-wise dot product is the cosine similarity (not
    Pearson r, because PCA rows are not zero-mean across features).

    Returns:
        aligned        : (n_pcs, n_features) reordered and sign-corrected fold
                         components aligned to group PC order
        abs_cosine     : (n_pcs,) absolute cosine similarity per matched pair
        pearson_r      : (n_pcs,) absolute Pearson r per matched pair (computed
                         after centering each PC across features)
    """
    cos_sim = group_components @ fold_components.T  # (n_pcs_group, n_pcs_fold)
    abs_cos = np.abs(cos_sim)
    row_ind, col_ind = linear_sum_assignment(-abs_cos)

    aligned = fold_components[col_ind].copy()
    abs_cosine = abs_cos[row_ind, col_ind]
    for i in range(len(row_ind)):
        if cos_sim[row_ind[i], col_ind[i]] < 0:
            aligned[i] *= -1

    # Pearson r between centered PC vectors, after alignment
    g_centered = group_components - group_components.mean(axis=1, keepdims=True)
    a_centered = aligned - aligned.mean(axis=1, keepdims=True)
    num = (g_centered * a_centered).sum(axis=1)
    den = np.linalg.norm(g_centered, axis=1) * np.linalg.norm(a_centered, axis=1)
    pearson_r = np.where(den > 0, num / den, 0.0)
    return aligned, abs_cosine, np.abs(pearson_r)


def run_loso(weights_list: list[np.ndarray], group_components: np.ndarray,
             n_pcs: int, logger: logging.Logger,
             subject_ids: list[int] | None = None) -> dict:
    """
    Leave-one-subject-out validation of group PCs.

    At each fold, PCA is fit on N-1 subjects and the fold PCs are matched to
    the canonical group PCs by absolute cosine similarity (Hungarian
    algorithm). The held-out subject's weights are then projected onto the
    aligned fold PCs, giving held-out variance explained.

    NOTE: matching fold PCs to the all-N group PCs is upward-biased because
    the held-out subject contributed to the group fit. Use the held-out
    variance explained as the validation signal; for unbiased PC stability,
    see compute_pairwise_fold_stability().

    Returns dict with keys:
        'heldout_var_explained': (n_subjects, n_pcs)
        'group_align_abs_cosine': (n_subjects, n_pcs) -- biased
        'group_align_pearson_r': (n_subjects, n_pcs)  -- biased
        'fold_components': list of (n_pcs, n_features) per fold,
            sign- and order-aligned to group_components (for stability calc)
    """
    n_subjects = len(weights_list)
    logger.info(f"Running LOSO validation ({n_subjects} folds, {n_pcs} PCs)...")

    heldout_var = np.zeros((n_subjects, n_pcs))
    abs_cosine_mat = np.zeros((n_subjects, n_pcs))
    pearson_mat = np.zeros((n_subjects, n_pcs))
    fold_components_aligned_list: list[np.ndarray] = []

    for fold_idx in range(n_subjects):
        train_weights = [w for i, w in enumerate(weights_list) if i != fold_idx]
        held_weights = weights_list[fold_idx]

        train_pooled, _ = pool_weights(train_weights)
        train_zscored, zscore_params = zscore_features(train_pooled)

        pca_fold = PCA(n_components=n_pcs, svd_solver='full')
        pca_fold.fit(train_zscored)

        fold_components_aligned, abs_cos, pearson_r = align_fold_to_group_pcs(
            pca_fold.components_, group_components
        )
        abs_cosine_mat[fold_idx] = abs_cos
        pearson_mat[fold_idx] = pearson_r
        fold_components_aligned_list.append(fold_components_aligned)

        held_transposed = held_weights.T  # (n_voxels_held, n_features)
        held_zscored, _ = zscore_features(held_transposed, params=zscore_params)

        held_scores = held_zscored @ fold_components_aligned.T

        total_var = np.sum(held_zscored ** 2)
        for k in range(n_pcs):
            reconstruction_k = held_scores[:, k:k+1] @ fold_components_aligned[k:k+1, :]
            heldout_var[fold_idx, k] = np.sum(reconstruction_k ** 2) / total_var

        sub_label = f"{subject_ids[fold_idx]:02d}" if subject_ids else str(fold_idx + 1)
        logger.info(
            f"  Fold {fold_idx+1}/{n_subjects} (held out sub-{sub_label}): "
            + ", ".join(
                f"PC{k+1}={heldout_var[fold_idx,k]*100:.1f}% "
                f"[|cos|={abs_cos[k]:.2f} (biased)]"
                for k in range(min(n_pcs, 4))
            )
        )

    return {
        'heldout_var_explained': heldout_var,
        'group_align_abs_cosine': abs_cosine_mat,
        'group_align_pearson_r': pearson_mat,
        'fold_components': fold_components_aligned_list,
    }


# ---------------------------------------------------------------------------
# Pairwise fold-to-fold stability (uncontaminated)
# ---------------------------------------------------------------------------

def compute_pairwise_fold_stability(
    fold_components_list: list[np.ndarray],
    n_pcs: int,
    logger: logging.Logger,
) -> dict:
    """
    Compute pairwise PC stability across LOSO folds.

    Each fold's PCA is fit on a disjoint N-1 subset (the held-out subject was
    not used). For every fold pair (i, j) with i < j, we re-align fold j's
    components to fold i's by Hungarian on absolute cosine similarity, then
    record per-PC abs cosine and abs Pearson r.

    Even within these pairs there is overlap (folds share N-2 subjects out of
    N-1), but neither fold's PCA contains a held-out subject that is then
    used as a "ground-truth" target, so the stability estimate is not
    upward-biased the way comparing to the all-N group fit is.

    Returns dict with keys:
        'pair_abs_cosine'    : (n_pairs, n_pcs)
        'pair_pearson_r'     : (n_pairs, n_pcs)
        'mean_abs_cosine'    : (n_pcs,)
        'mean_pearson_r'     : (n_pcs,)
        'pairs'              : list of (i, j) tuples
    """
    n_folds = len(fold_components_list)
    pairs = list(combinations(range(n_folds), 2))
    n_pairs = len(pairs)
    logger.info(
        f"Computing pairwise fold-to-fold stability ({n_pairs} pairs, {n_pcs} PCs)..."
    )

    pair_abs_cos = np.zeros((n_pairs, n_pcs))
    pair_pearson = np.zeros((n_pairs, n_pcs))

    for p, (i, j) in enumerate(pairs):
        _, abs_cos, pearson_r = align_fold_to_group_pcs(
            fold_components_list[j], fold_components_list[i]
        )
        pair_abs_cos[p] = abs_cos
        pair_pearson[p] = pearson_r

    mean_abs_cos = pair_abs_cos.mean(axis=0)
    mean_pearson = pair_pearson.mean(axis=0)

    for k in range(n_pcs):
        logger.info(
            f"  PC{k+1}: pairwise |cos| = {mean_abs_cos[k]:.2f} "
            f"[{pair_abs_cos[:, k].min():.2f}-{pair_abs_cos[:, k].max():.2f}], "
            f"|r| = {mean_pearson[k]:.2f} "
            f"[{pair_pearson[:, k].min():.2f}-{pair_pearson[:, k].max():.2f}]"
        )

    return {
        'pair_abs_cosine': pair_abs_cos,
        'pair_pearson_r': pair_pearson,
        'mean_abs_cosine': mean_abs_cos,
        'mean_pearson_r': mean_pearson,
        'pairs': pairs,
    }


# ---------------------------------------------------------------------------
# Null significance testing
# ---------------------------------------------------------------------------

def load_null_weights(null_weight_files: list[Path],
                      logger: logging.Logger) -> list[np.ndarray]:
    """
    Load pre-pooled null weight matrices.

    Each file should contain 'weights_joint' with shape (n_features, n_voxels_total)
    produced by pooling prepare_pca_weights.py outputs across subjects for one
    null iteration.
    """
    logger.info(f"Loading {len(null_weight_files)} null weight matrices...")
    null_matrices = []
    for p in null_weight_files:
        data = np.load(p)
        null_matrices.append(data['weights_joint'].astype(np.float32).T)
    logger.info(f"  Null matrix shape: {null_matrices[0].shape}")
    return null_matrices


def demean_pooled_by_subject(
    pooled: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """
    Subtract each subject's per-feature mean from its block of a pooled
    (n_voxels_total, n_features) matrix. Boundaries are cumulative subject
    voxel counts (length n_subjects + 1).
    """
    out = pooled.copy()
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        out[s:e] -= out[s:e].mean(axis=0, keepdims=True)
    return out


def test_null_significance(real_eigenvalues: np.ndarray,
                           null_matrices: list[np.ndarray],
                           zscore_params: dict,
                           alpha: float,
                           logger: logging.Logger,
                           boundaries: np.ndarray | None = None,
                           apply_per_subject_demean: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Compare real PCA eigenvalues against null distribution.

    Null matrices are optionally per-subject demeaned (must match the real-
    data treatment) then z-scored using the real data's z-score parameters,
    then PCA is fit. Null eigenvalues are collected per PC rank.

    Returns:
        null_eigenvalues    : (n_null_iterations, n_features) array
        significance_mask   : (n_features,) bool -- True where real > threshold
    """
    n_null = len(null_matrices)
    n_features = null_matrices[0].shape[1]
    if apply_per_subject_demean and boundaries is None:
        raise ValueError("boundaries required when apply_per_subject_demean=True")
    logger.info(
        f"Running null significance test ({n_null} iterations, alpha={alpha}, "
        f"per_subject_demean={apply_per_subject_demean})..."
    )

    null_eigenvalues = np.zeros((n_null, n_features))
    for i, null_mat in enumerate(null_matrices):
        mat = null_mat
        if apply_per_subject_demean:
            mat = demean_pooled_by_subject(mat, boundaries)
        null_zscored, _ = zscore_features(mat, params=zscore_params)
        pca_null = PCA(n_components=None, svd_solver='full')
        pca_null.fit(null_zscored)
        null_eigenvalues[i] = pca_null.explained_variance_

    threshold = np.percentile(null_eigenvalues, 100 * (1 - alpha), axis=0)
    significance_mask = real_eigenvalues > threshold

    n_sig = significance_mask.sum()
    logger.info(f"  Significant PCs (p < {alpha}): {n_sig}")
    for k in range(min(n_sig + 2, n_features)):
        status = "SIGNIFICANT" if significance_mask[k] else "not significant"
        logger.info(f"    PC{k+1}: real={real_eigenvalues[k]:.4f}, "
                    f"null 95th={threshold[k]:.4f} [{status}]")

    return null_eigenvalues, significance_mask


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(output_dir: Path, label: str, group_cfg: dict,
                 pca: PCA, components: np.ndarray, scores: np.ndarray,
                 boundaries: np.ndarray, masks_list: list[np.ndarray],
                 selection_list: list[np.ndarray], voxel_filter_info: dict,
                 zscore_params: dict, loso_results: dict,
                 stability_results: dict | None,
                 feature_names: list[str], dropped_info: dict,
                 n_pcs_retained: int,
                 significance_mask: np.ndarray | None,
                 null_eigenvalues: np.ndarray | None,
                 logger: logging.Logger) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    # Components and variance
    np.save(output_dir / 'pca_components.npy', components)
    np.save(output_dir / 'pca_explained_variance.npy', pca.explained_variance_)
    np.save(output_dir / 'pca_explained_variance_ratio.npy', pca.explained_variance_ratio_)
    np.savez_compressed(output_dir / 'pca_zscore_params.npz',
                        mean=zscore_params['mean'], std=zscore_params['std'])

    # Per-subject scores and full-brain score maps (sign-consistent with components)
    for sub_idx, sub_cfg in enumerate(group_cfg['subjects']):
        sub_start = boundaries[sub_idx]
        sub_end = boundaries[sub_idx + 1]
        sub_scores = scores[sub_start:sub_end, :n_pcs_retained]

        subject = sub_cfg['subject']
        np.save(output_dir / f'pca_scores_sub-{subject:02d}.npy', sub_scores)

        mask = masks_list[sub_idx]
        score_maps = np.full((n_pcs_retained, mask.shape[0]), np.nan, dtype=np.float32)
        score_maps[:, mask] = sub_scores.T
        np.save(output_dir / f'pca_score_maps_sub-{subject:02d}.npy', score_maps)

        # Save the voxel selection (indices into the original mask-space weights)
        # so null pooling can apply the same filter+subsample if rerun later.
        np.save(
            output_dir / f'voxel_selection_sub-{subject:02d}.npy',
            selection_list[sub_idx],
        )

    if null_eigenvalues is not None:
        np.save(output_dir / 'null_eigenvalues.npy', null_eigenvalues)

    loso_var = loso_results['heldout_var_explained']
    loso_abscos = loso_results['group_align_abs_cosine']
    loso_pearson = loso_results['group_align_pearson_r']

    metadata = {
        'group': label,
        'n_subjects': len(group_cfg['subjects']),
        'subjects': [s['subject'] for s in group_cfg['subjects']],
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'dropped_features': dropped_info,
        'voxel_filter': voxel_filter_info,
        'n_pcs_total': int(pca.n_components_),
        'n_pcs_retained': n_pcs_retained,
        'explained_variance_ratio': pca.explained_variance_ratio_[:n_pcs_retained].tolist(),
        'cumulative_variance_ratio': float(
            pca.explained_variance_ratio_[:n_pcs_retained].sum()),
        'loso': {
            'heldout_var_explained_mean': loso_var.mean(axis=0).tolist(),
            'heldout_var_explained_std': loso_var.std(axis=0).tolist(),
            'heldout_var_explained_per_fold': loso_var.tolist(),
            'group_align_abs_cosine_mean_BIASED': loso_abscos.mean(axis=0).tolist(),
            'group_align_abs_cosine_per_fold_BIASED': loso_abscos.tolist(),
            'group_align_pearson_r_mean_BIASED': loso_pearson.mean(axis=0).tolist(),
            'group_align_pearson_r_per_fold_BIASED': loso_pearson.tolist(),
            'note_on_biased_metrics': (
                "group_align_* metrics compare each fold's N-1 PCA to the all-N "
                "group PCA, which includes the held-out subject -- upward biased. "
                "Use stability.pairwise_* for an uncontaminated estimate."
            ),
        },
        'significance': {
            'tested': significance_mask is not None,
            'significant_pcs': int(significance_mask.sum()) if significance_mask is not None else None,
            'mask': significance_mask.tolist() if significance_mask is not None else None,
        },
    }

    if stability_results is not None:
        metadata['stability'] = {
            'pairwise_abs_cosine_mean': stability_results['mean_abs_cosine'].tolist(),
            'pairwise_pearson_r_mean': stability_results['mean_pearson_r'].tolist(),
            'pairs': [list(p) for p in stability_results['pairs']],
            'pairwise_abs_cosine_per_pair': stability_results['pair_abs_cosine'].tolist(),
            'pairwise_pearson_r_per_pair': stability_results['pair_pearson_r'].tolist(),
        }

    with open(output_dir / 'pca_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved results to: {output_dir}")
    logger.info(f"  pca_components.npy          {components.shape}")
    logger.info(f"  pca_scores_sub-XX.npy       per subject (sign-aligned to components)")
    logger.info(f"  pca_score_maps_sub-XX.npy   per subject, full brain (91282)")
    logger.info(f"  voxel_selection_sub-XX.npy  per subject (for null pooling)")
    logger.info(f"  pca_metadata.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Fit group PCA on VP model weights')
    parser.add_argument('--group', type=str, required=True,
                        choices=list(GROUP_REGISTRY.keys()),
                        help='Group configuration name')
    parser.add_argument('--n-pcs', type=int, default=None,
                        help='Number of PCs to retain (default: determined by scree/null)')
    parser.add_argument('--r2-threshold', type=float, default=DEFAULT_R2_THRESHOLD,
                        help=f'Drop voxels with R2_full <= threshold per subject '
                             f'before pooling (default: {DEFAULT_R2_THRESHOLD})')
    parser.add_argument('--voxel-seed', type=int, default=DEFAULT_VOXEL_SEED,
                        help=f'Seed for subsampling each subject to the min voxel '
                             f'count (default: {DEFAULT_VOXEL_SEED})')
    parser.add_argument('--per-subject-demean', action='store_true',
                        help='Subtract each subject\'s per-feature mean from its '
                             'weight block before pooling. Removes the subject-mean '
                             'confound that otherwise correlates with training-data '
                             'quantity. Applied to null matrices too when provided.')
    parser.add_argument('--null-weight-files', type=str, nargs='+', default=None,
                        help='Pre-pooled null weight .npz files (one per null iteration). '
                             'Each must contain weights_joint (n_features, n_voxels_total). '
                             'NOTE: nulls must use the same voxel selection as the real fit '
                             '(see voxel_selection_sub-XX.npy outputs).')
    parser.add_argument('--null-alpha', type=float, default=0.05,
                        help='Significance threshold for null testing (default: 0.05)')
    args = parser.parse_args()

    group_cfg = GROUP_REGISTRY[args.group]
    label = group_cfg['output_label']
    output_dir = PATHS['pca'] / label

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, label)

    logger.info("=" * 80)
    logger.info("FIT GROUP PCA")
    logger.info("=" * 80)
    logger.info(f"Group: {args.group}")
    logger.info(f"Subjects: {[s['subject'] for s in group_cfg['subjects']]}")
    logger.info(f"R2 threshold: {args.r2_threshold}")
    logger.info(f"Voxel-subsample seed: {args.voxel_seed}")
    logger.info(f"Per-subject demean: {args.per_subject_demean}")
    logger.info(f"Output: {output_dir}")

    subject_ids = [s['subject'] for s in group_cfg['subjects']]

    # ------------------------------------------------------------------
    # Load weights and R2
    # ------------------------------------------------------------------
    logger.info("")
    weights_list, R2_list, masks_list, metadata_list, feature_names_list = load_group_weights(
        group_cfg, logger
    )

    # ------------------------------------------------------------------
    # Align to common feature set
    # ------------------------------------------------------------------
    logger.info("")
    weights_list, feature_names, dropped_info = align_to_common_features(
        weights_list, feature_names_list, subject_ids, logger
    )
    logger.info(f"Features ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # R2 filter + equalize voxel counts per subject
    # ------------------------------------------------------------------
    logger.info("")
    weights_list, masks_list, selection_list, voxel_filter_info = filter_and_subsample_voxels(
        weights_list, R2_list, masks_list, subject_ids,
        r2_threshold=args.r2_threshold, seed=args.voxel_seed, logger=logger,
    )

    # ------------------------------------------------------------------
    # Optional per-subject demeaning
    # ------------------------------------------------------------------
    voxel_filter_info['per_subject_demean'] = bool(args.per_subject_demean)
    if args.per_subject_demean:
        logger.info("")
        weights_list = per_subject_demean(weights_list, subject_ids, logger)

    # ------------------------------------------------------------------
    # Pool and z-score
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Pooling and z-scoring...")
    pooled, boundaries = pool_weights(weights_list)
    pooled_zscored, zscore_params = zscore_features(pooled)
    logger.info(f"  Pooled matrix: {pooled_zscored.shape}")

    # ------------------------------------------------------------------
    # Group PCA (no fit_transform yet -- we sign-fix components first
    # and then compute scores from sign-fixed components so the two
    # outputs stay consistent (fix for C1))
    # ------------------------------------------------------------------
    logger.info("")
    logger.info(f"Fitting PCA on matrix {pooled_zscored.shape}...")
    pca = PCA(n_components=None, svd_solver='full')
    pca.fit(pooled_zscored)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for k in [1, 2, 3, 4, 5]:
        if k <= len(cumvar):
            logger.info(f"  Cumulative variance PC1-{k}: {cumvar[k-1]*100:.1f}%")

    # ------------------------------------------------------------------
    # Null significance testing (optional)
    # ------------------------------------------------------------------
    significance_mask = None
    null_eigenvalues = None
    if args.null_weight_files:
        logger.info("")
        null_paths = [Path(p) for p in args.null_weight_files]
        null_matrices = load_null_weights(null_paths, logger)
        null_eigenvalues, significance_mask = test_null_significance(
            pca.explained_variance_, null_matrices, zscore_params,
            args.null_alpha, logger,
            boundaries=np.asarray(boundaries),
            apply_per_subject_demean=args.per_subject_demean,
        )

    # ------------------------------------------------------------------
    # Determine n_pcs to retain
    # ------------------------------------------------------------------
    if args.n_pcs is not None:
        n_pcs_retained = args.n_pcs
    elif significance_mask is not None:
        n_pcs_retained = max(1, int(significance_mask.sum()))
    else:
        # Heuristic fallback when no null available: retain PCs up to the
        # point where the successive eigenvalue ratio drop is < 1pp. This is
        # NOT a principled scree analysis -- treat retained PCs as
        # exploratory until null testing is added (see C5 in the audit).
        ratios = pca.explained_variance_ratio_
        marginal = np.diff(ratios)
        elbow_candidates = np.where(marginal > -0.01)[0]
        n_pcs_retained = int(elbow_candidates[0]) + 1 if len(elbow_candidates) > 0 else 5
        logger.warning(
            f"No null distribution provided -- retaining {n_pcs_retained} PCs by "
            f"a heuristic elbow (successive drop < 1pp). This is exploratory; "
            f"re-run with --null-weight-files for a defensible n_pcs."
        )
        logger.warning("  Inspect pca_explained_variance_ratio.npy scree plot to verify")

    logger.info(f"Retaining {n_pcs_retained} PCs")

    # ------------------------------------------------------------------
    # Sign-fix components THEN compute scores from the sign-fixed
    # components, so saved pca_components.npy and pca_score_maps_*.npy
    # remain sign-consistent (fix for C1).
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Fixing PC signs...")
    components, sign_vec = fix_pc_signs(pca.components_, feature_names, logger)
    scores = pooled_zscored @ components.T  # (n_voxels_total, n_pcs)

    # ------------------------------------------------------------------
    # LOSO validation (held-out variance + biased group-alignment metrics)
    # ------------------------------------------------------------------
    logger.info("")
    loso_results = run_loso(
        weights_list, components[:n_pcs_retained], n_pcs_retained,
        logger, subject_ids=subject_ids,
    )

    loso_var = loso_results['heldout_var_explained']
    loso_abscos = loso_results['group_align_abs_cosine']
    logger.info("LOSO summary:")
    for k in range(n_pcs_retained):
        vals = loso_var[:, k]
        ac = loso_abscos[:, k]
        logger.info(
            f"  PC{k+1}: heldout var={vals.mean()*100:.1f}% "
            f"[{vals.min()*100:.1f}-{vals.max()*100:.1f}%], "
            f"|cos|(biased)={ac.mean():.2f} [{ac.min():.2f}-{ac.max():.2f}]"
        )

    # ------------------------------------------------------------------
    # Uncontaminated pairwise PC stability
    # ------------------------------------------------------------------
    logger.info("")
    stability_results = compute_pairwise_fold_stability(
        loso_results['fold_components'], n_pcs_retained, logger,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    logger.info("")
    save_results(
        output_dir, label, group_cfg, pca, components, scores,
        boundaries, masks_list, selection_list, voxel_filter_info,
        zscore_params, loso_results, stability_results,
        feature_names, dropped_info, n_pcs_retained,
        significance_mask, null_eigenvalues, logger,
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
