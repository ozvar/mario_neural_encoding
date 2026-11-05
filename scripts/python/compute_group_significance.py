"""
Group-level significance from per-subject permutation tests.

Pools per-subject significance.npz files (produced by
compute_vp_significance_permutation.py) into a group-level inference using:

1. Weighted Stouffer combination of one-tailed z-scores
       w_i = sqrt(n_test_active_i)   (more test data -> more precise z)
       Z_group[v] = sum_i (w_i * z_i[v]) / sqrt(sum_i w_i**2)
       p_group[v] = 1 - Phi(Z_group[v])
   BH-FDR across voxels (within the >= min-subjects-valid mask).

2. Conjunction map (supplementary, more conservative):
       voxel is "conjunction-significant" iff all subjects with valid data at v
       are individually significant at the per-subject FDR threshold.

A voxel is included in the group test only if it is valid in at least
min_subjects_valid subjects (default 4/5 from PARAMETERS['permutation']).
Voxels missing in too many subjects are NaN in Z/p outputs and False in masks.

Outputs are saved in full 91k grayordinate space so they can be loaded
directly by create_vp_model_cifti.py.

Usage:
    python scripts/python/compute_group_significance.py --group GROUP_N5_PRIMARY
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import norm

from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'group_significance.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def get_vp_dir(subject: int, train_sessions: list, test_sessions: list,
               experiment_id: str) -> Path:
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    return PATHS['variance_partitioning'] / dataset_dir / experiment_id


def load_subject_significance(vp_dir: Path, logger: logging.Logger,
                              fdr_alpha: float) -> dict:
    """
    Load p-values, observed stats, valid-voxel mask, and n_test_active.
    Expand all per-voxel arrays into full 91k space (NaN / False outside mask).
    """
    sig_file = vp_dir / 'significance.npz'
    mask_file = vp_dir / 'valid_voxels_mask.npy'
    if not sig_file.exists():
        raise FileNotFoundError(f"Missing significance file: {sig_file}")
    if not mask_file.exists():
        raise FileNotFoundError(f"Missing valid_voxels_mask: {mask_file}")

    sig = np.load(sig_file, allow_pickle=False)
    valid_mask = np.load(mask_file)
    n_total = len(valid_mask)
    valid_idx = np.where(valid_mask)[0]

    n_test_active = int(sig['n_test_active'])
    logger.info(f"  {vp_dir.parent.name}: n_test_active={n_test_active}, "
                f"valid voxels={int(valid_mask.sum())}/{n_total}")

    pm_spaces = sorted(
        name.split('pm_')[1] for name in sig.files
        if name.startswith('pm_') and not name.startswith('p_pm_')
        and not name.startswith('p_fdr_pm_') and not name.startswith('sig_pm_')
    )

    expanded = {
        'valid_mask': valid_mask,
        'n_test_active': n_test_active,
        'pm_spaces': pm_spaces,
        'observed': {},
        'p_raw': {},
        'sig_per_subject': {},
    }

    def to_full(values_filtered, fill=np.nan, dtype=None):
        arr = np.full(n_total, fill, dtype=dtype or values_filtered.dtype)
        arr[valid_idx] = values_filtered
        return arr

    expanded['observed']['r_full'] = to_full(sig['r_full'].astype(np.float32))
    expanded['p_raw']['r_full'] = to_full(sig['p_r_full'].astype(np.float64),
                                          fill=np.nan)
    sig_key_r = f'sig_r_full_fdr_q{int(fdr_alpha * 100):02d}'
    if sig_key_r in sig.files:
        expanded['sig_per_subject']['r_full'] = to_full(
            sig[sig_key_r].astype(bool), fill=False, dtype=bool)

    for space in pm_spaces:
        expanded['observed'][f'pm_{space}'] = to_full(sig[f'pm_{space}'].astype(np.float32))
        expanded['p_raw'][f'pm_{space}'] = to_full(sig[f'p_pm_{space}'].astype(np.float64))
        sig_key_pm = f'sig_pm_{space}_fdr_q{int(fdr_alpha * 100):02d}'
        if sig_key_pm in sig.files:
            expanded['sig_per_subject'][f'pm_{space}'] = to_full(
                sig[sig_key_pm].astype(bool), fill=False, dtype=bool)

    return expanded


def weighted_stouffer(p_per_subject_full: list, weights: np.ndarray,
                      valid_per_subject_full: list, min_valid: int
                      ) -> tuple:
    """
    Per-voxel weighted Stouffer.
    Returns (Z_group, p_group, n_valid_per_voxel). Voxels with fewer than
    min_valid valid subjects -> Z=NaN, p=NaN.

    Edge cases:
    - Clip raw p-values away from 0 and 1 before norm.ppf to avoid +/-inf z.
    """
    n_subs = len(p_per_subject_full)
    n_voxels = len(p_per_subject_full[0])

    p_arr = np.stack(p_per_subject_full, axis=0)            # (S, V)
    valid_arr = np.stack(valid_per_subject_full, axis=0)    # (S, V) bool
    w = weights.astype(np.float64)[:, None]                  # (S, 1)

    # Clip raw p to (eps, 1-eps).
    eps = 1e-300
    p_clip = np.clip(p_arr, eps, 1.0 - eps)
    z = -norm.ppf(p_clip)  # large positive z = small p = strong evidence

    # Mask z and weights by validity.
    valid_f = valid_arr.astype(np.float64)
    wv = w * valid_f                                         # (S, V)
    wz = wv * z

    n_valid = valid_arr.sum(axis=0).astype(np.int32)         # (V,)
    sumsq_w = (wv ** 2).sum(axis=0)                          # (V,)
    sum_wz = wz.sum(axis=0)                                  # (V,)

    Z_group = np.full(n_voxels, np.nan)
    mask_enough = n_valid >= min_valid
    denom = np.sqrt(sumsq_w[mask_enough])
    Z_group[mask_enough] = np.where(denom > 0,
                                    sum_wz[mask_enough] / denom,
                                    np.nan)
    p_group = np.full(n_voxels, np.nan)
    p_group[mask_enough] = 1.0 - norm.cdf(Z_group[mask_enough])

    return Z_group, p_group, n_valid


def fdr_correction(p_values: np.ndarray, alpha: float) -> tuple:
    """BH-FDR restricted to non-NaN p-values. Returns (q_full, sig_full)."""
    n_total = len(p_values)
    finite_mask = np.isfinite(p_values)
    finite_p = p_values[finite_mask]
    n = len(finite_p)

    q_full = np.full(n_total, np.nan)
    sig_full = np.zeros(n_total, dtype=bool)
    if n == 0:
        return q_full, sig_full

    order = np.argsort(finite_p)
    sorted_p = finite_p[order]
    threshold_line = np.arange(1, n + 1) / n * alpha
    below = sorted_p <= threshold_line
    threshold = sorted_p[int(np.where(below)[0][-1])] if below.any() else 0.0

    q_sorted = np.zeros(n)
    q_sorted[-1] = min(sorted_p[-1], 1.0)
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(sorted_p[i] * n / (i + 1), q_sorted[i + 1], 1.0)
    q_finite = np.zeros(n)
    q_finite[order] = q_sorted

    q_full[finite_mask] = q_finite
    sig_full[finite_mask] = finite_p <= threshold
    return q_full, sig_full


def conjunction_mask(sig_per_subject_full: list, valid_per_subject_full: list,
                     min_valid: int, min_sig: int | None = None) -> np.ndarray:
    """
    Conjunction / partial-conjunction reliability mask.

    A voxel must be valid in >= min_valid subjects, then:
    - min_sig is None (strict conjunction): significant in every subject where it
      is valid (Heller et al. 2007 with u = n_valid).
    - min_sig = k (partial conjunction): significant in >= k of the subjects with
      valid data at that voxel (Heller et al. 2007 with u = k). Count-based; used
      as a descriptive reliability overlay, not a formal FDR-controlled test.
    """
    sig_arr = np.stack(sig_per_subject_full, axis=0)         # (S, V)
    valid_arr = np.stack(valid_per_subject_full, axis=0)
    n_valid = valid_arr.sum(axis=0)
    n_sig_among_valid = (sig_arr & valid_arr).sum(axis=0)
    if min_sig is None:
        return (n_valid >= min_valid) & (n_sig_among_valid == n_valid)
    return (n_valid >= min_valid) & (n_sig_among_valid >= min_sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=str, required=True,
                        choices=list(GROUP_REGISTRY.keys()))
    parser.add_argument('--min-subjects', type=int, default=None,
                        help='Voxel must be valid in >= this many subjects')
    parser.add_argument('--conjunction-min-sig', type=int, default=4,
                        help='Partial-conjunction threshold k: voxel significant '
                             'in >= k valid subjects (in addition to strict '
                             'all-valid conjunction). Default 4.')
    args = parser.parse_args()

    perm_cfg = PARAMETERS['permutation']
    min_valid = args.min_subjects if args.min_subjects is not None \
        else perm_cfg['min_subjects_valid']
    conj_min_sig = args.conjunction_min_sig
    fdr_alpha = perm_cfg['fdr_alpha']

    group_cfg = GROUP_REGISTRY[args.group]
    out_dir = PATHS['variance_partitioning'] / 'group_average' / \
        group_cfg['output_label']
    logger = setup_logging(out_dir)
    logger.info("=" * 80)
    logger.info(f"GROUP SIGNIFICANCE | {args.group} | min_valid={min_valid} | "
                f"fdr_alpha={fdr_alpha}")
    logger.info("=" * 80)

    per_subject = []
    for sub_cfg in group_cfg['subjects']:
        vp_dir = get_vp_dir(sub_cfg['subject'], sub_cfg['train_sessions'],
                            sub_cfg['test_sessions'], sub_cfg['experiment_id'])
        try:
            expanded = load_subject_significance(vp_dir, logger, fdr_alpha)
            per_subject.append((sub_cfg['subject'], expanded))
        except FileNotFoundError as e:
            logger.warning(f"sub-{sub_cfg['subject']:02d} skipped: {e}")

    if len(per_subject) < min_valid:
        raise RuntimeError(
            f"Only {len(per_subject)} subjects loaded but min_valid={min_valid}; "
            f"cannot compute group significance."
        )

    pm_spaces = sorted(set().union(*(p['pm_spaces'] for _, p in per_subject)))
    stat_names = ['r_full'] + [f'pm_{s}' for s in pm_spaces]
    weights = np.array([np.sqrt(p['n_test_active']) for _, p in per_subject])
    logger.info(f"Stouffer weights (sqrt n_test_active): "
                f"{[f'{w:.1f}' for w in weights]}")

    valid_full = [p['valid_mask'] for _, p in per_subject]

    save_dict = {
        'n_valid_per_voxel': np.stack(valid_full, axis=0).sum(axis=0).astype(np.int32),
        'min_subjects_required': np.int32(min_valid),
        'fdr_alpha': np.float32(fdr_alpha),
        'subjects': np.array([s for s, _ in per_subject], dtype=np.int32),
        'stouffer_weights': weights.astype(np.float32),
    }

    for stat in stat_names:
        logger.info("")
        logger.info(f"=== {stat} ===")
        try:
            p_list = [p['p_raw'][stat] for _, p in per_subject]
        except KeyError:
            logger.warning(f"Statistic {stat} missing in at least one subject; "
                           f"skipping.")
            continue

        # Mean observed across subjects (NaN where invalid for that subject).
        obs_stack = np.stack([p['observed'][stat] for _, p in per_subject], axis=0)
        valid_stack = np.stack(valid_full, axis=0)
        with np.errstate(invalid='ignore'):
            mean_obs = np.where(
                valid_stack.sum(axis=0) >= min_valid,
                np.nanmean(np.where(valid_stack, obs_stack, np.nan), axis=0),
                np.nan,
            )

        Z_group, p_group, n_valid = weighted_stouffer(
            p_list, weights, valid_full, min_valid,
        )
        q_group, sig_group = fdr_correction(p_group, alpha=fdr_alpha)

        # Conjunction (per-subject FDR-significant in all valid subjects) and
        # partial conjunction (significant in >= conj_min_sig valid subjects).
        if stat in per_subject[0][1]['sig_per_subject']:
            sig_per_sub_full = [p['sig_per_subject'][stat] for _, p in per_subject]
            sig_conj = conjunction_mask(sig_per_sub_full, valid_full, min_valid)
            sig_conj_partial = conjunction_mask(
                sig_per_sub_full, valid_full, min_valid, min_sig=conj_min_sig)
        else:
            sig_conj = np.zeros_like(sig_group)
            sig_conj_partial = np.zeros_like(sig_group)
            logger.warning(f"No per-subject sig mask for {stat}; conjunction set to 0.")

        n_total = len(sig_group)
        n_in = int((n_valid >= min_valid).sum())
        logger.info(f"  voxels in mask (>= {min_valid} valid): {n_in}/{n_total}")
        logger.info(f"  Stouffer FDR-significant: {int(sig_group.sum())} "
                    f"({sig_group.sum() / max(n_in, 1) * 100:.1f}% of mask)")
        logger.info(f"  Conjunction-significant: {int(sig_conj.sum())} "
                    f"({sig_conj.sum() / max(n_in, 1) * 100:.1f}% of mask)")
        logger.info(f"  Partial-conjunction (>= {conj_min_sig}/valid): "
                    f"{int(sig_conj_partial.sum())} "
                    f"({sig_conj_partial.sum() / max(n_in, 1) * 100:.1f}% of mask)")

        prefix = stat
        save_dict[f'{prefix}_mean_observed'] = mean_obs.astype(np.float32)
        save_dict[f'{prefix}_Z_group'] = Z_group.astype(np.float32)
        save_dict[f'{prefix}_p_group'] = p_group.astype(np.float32)
        save_dict[f'{prefix}_p_fdr_group'] = q_group.astype(np.float32)
        save_dict[f'{prefix}_sig_group_fdr_q{int(fdr_alpha * 100):02d}'] = sig_group
        save_dict[f'{prefix}_sig_conj'] = sig_conj
        save_dict[f'{prefix}_sig_conj_min{conj_min_sig}'] = sig_conj_partial

    # Four-way classification: motor x scene PM significance (Stouffer-based).
    motor_key = f'pm_motor_sig_group_fdr_q{int(fdr_alpha * 100):02d}'
    scene_key = f'pm_scene_sig_group_fdr_q{int(fdr_alpha * 100):02d}'
    if motor_key in save_dict and scene_key in save_dict:
        m = save_dict[motor_key]
        s = save_dict[scene_key]
        classification = (m.astype(np.int8) * 1) + (s.astype(np.int8) * 2)
        # 0=neither, 1=motor only, 2=scene only, 3=both
        save_dict['vp_classification_stouffer'] = classification
        logger.info("")
        logger.info("Four-way classification (Stouffer FDR):")
        for label, code in [('neither', 0), ('motor only', 1), ('scene only', 2),
                            ('both (conjunction candidate)', 3)]:
            n = int((classification == code).sum())
            logger.info(f"  {label}: {n}")

    out_file = out_dir / 'group_significance.npz'
    np.savez_compressed(out_file, **save_dict)
    logger.info("")
    logger.info(f"Saved group significance to {out_file}")

    # Also dump a small JSON for human inspection of parameters.
    meta = {
        'group': args.group,
        'min_subjects_valid': min_valid,
        'fdr_alpha': fdr_alpha,
        'n_perm': perm_cfg['n_perm'],
        'block_size': perm_cfg['block_size'],
        'subjects_included': [int(s) for s, _ in per_subject],
        'stouffer_weights': [float(w) for w in weights],
    }
    with open(out_dir / 'group_significance_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
