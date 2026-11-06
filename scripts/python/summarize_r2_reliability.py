"""
Cross-subject reliability of the R2 map, for GROUP_N5_PRIMARY, restricted to
cortex-only (59412) vertices significant (FDR q<0.05, permutation test) in
>= 4 of 5 subjects (group_r_full_sig_conj_min4.dscalar.nii).

1. Magnitude: each subject's mean R2 among those vertices.
2. Topology: leave-one-out Pearson correlation -- subject i's R2 vs. the mean
   R2 of the other 4 subjects. Mean/range across the 5 LOO r's, with a plain
   vertex-shuffle permutation test on the mean-r statistic.

Usage:
    python scripts/python/summarize_r2_reliability.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from neuro_utils import cortex_only, load_cifti
from scipy.stats import pearsonr

from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.group_configs import GROUP_N5_PRIMARY

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def subject_cifti_path(subject_spec: dict, cifti_dir: Path) -> Path:
    train = subject_spec['train_sessions']
    test = subject_spec['test_sessions']
    train_str = f"{min(train):03d}-{max(train):03d}"
    test_str = f"{min(test):03d}-{max(test):03d}"
    dataset_label = f"sub-{subject_spec['subject']:02d}_train-ses-{train_str}_test-ses-{test_str}"
    return cifti_dir / f"{dataset_label}_{subject_spec['experiment_id']}_R2_full.dscalar.nii"


def load_group_significant_cortex_r2(group_cfg: dict) -> tuple[dict[int, np.ndarray], np.ndarray]:
    group_label = group_cfg['output_label']
    cifti_dir = PATHS['figures'] / 'ciftis' / group_label

    sig_path = cifti_dir / 'group_r_full_sig_conj_min4.dscalar.nii'
    sig_mask = cortex_only(load_cifti(sig_path)[0].squeeze()) > 0

    r2_by_subject = {}
    for subject_spec in group_cfg['subjects']:
        r2_path = subject_cifti_path(subject_spec, cifti_dir)
        r2_by_subject[subject_spec['subject']] = cortex_only(load_cifti(r2_path)[0].squeeze())

    return r2_by_subject, sig_mask


def per_subject_magnitude(r2_by_subject: dict[int, np.ndarray], sig_mask: np.ndarray) -> pd.DataFrame:
    rows = [
        {'subject': subj, 'mean_r2_conj_min4': r2[sig_mask].mean()}
        for subj, r2 in r2_by_subject.items()
    ]
    return pd.DataFrame(rows).sort_values('subject', ignore_index=True)


def _loo_r_values(r2_by_subject: dict[int, np.ndarray], sig_mask: np.ndarray) -> np.ndarray:
    subjects = list(r2_by_subject.keys())
    stacked = np.stack([r2_by_subject[s][sig_mask] for s in subjects])  # (n_subjects, n_sig_vertices)
    r_values = []
    for i in range(len(subjects)):
        others_mean = np.delete(stacked, i, axis=0).mean(axis=0)
        r, _ = pearsonr(stacked[i], others_mean)
        r_values.append(r)
    return np.array(r_values)


def loo_spatial_correlations(r2_by_subject: dict[int, np.ndarray], sig_mask: np.ndarray) -> np.ndarray:
    """Leave-one-out Pearson r per subject, restricted to sig_mask vertices."""
    return _loo_r_values(r2_by_subject, sig_mask)


def permutation_test_mean_loo_r(
    r2_by_subject: dict[int, np.ndarray],
    sig_mask: np.ndarray,
    n_perm: int = PARAMETERS['permutation']['n_perm'],
    seed: int = PARAMETERS['permutation']['seed'],
) -> tuple[float, float]:
    """Observed mean LOO r and a one-tailed permutation p (plain vertex shuffle, no spin test)."""
    observed_mean_r = _loo_r_values(r2_by_subject, sig_mask).mean()

    subjects = list(r2_by_subject.keys())
    stacked = np.stack([r2_by_subject[s][sig_mask] for s in subjects])
    n_vertices = stacked.shape[1]

    rng = np.random.default_rng(seed)
    null_means = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n_vertices)
        permuted = stacked[:, perm]
        r_values = []
        for i in range(len(subjects)):
            others_mean = np.delete(permuted, i, axis=0).mean(axis=0)
            r, _ = pearsonr(stacked[i], others_mean)
            r_values.append(r)
        null_means[p] = np.mean(r_values)

    p_value = (1 + np.sum(null_means >= observed_mean_r)) / (n_perm + 1)
    return observed_mean_r, p_value


def run(group_cfg: dict = GROUP_N5_PRIMARY) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    log.info(f"Group: {group_cfg['output_label']}")
    r2_by_subject, sig_mask = load_group_significant_cortex_r2(group_cfg)
    log.info(f"n group-significant cortical vertices: {sig_mask.sum()}")

    magnitude_df = per_subject_magnitude(r2_by_subject, sig_mask)
    loo_r = loo_spatial_correlations(r2_by_subject, sig_mask)
    observed_mean_r, p_value = permutation_test_mean_loo_r(r2_by_subject, sig_mask)

    group_label = group_cfg['output_label']
    out_path = PATHS['figures'] / 'ciftis' / group_label / f'{group_label}_r2_reliability.csv'
    magnitude_df.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}")

    return magnitude_df, loo_r, observed_mean_r, p_value


if __name__ == '__main__':
    magnitude_df, loo_r, observed_mean_r, p_value = run()

    print(magnitude_df.to_string(index=False))
    print()
    print(f"per-subject mean R2 (group-sig vertices): range {magnitude_df['mean_r2_conj_min4'].min():.4f}-{magnitude_df['mean_r2_conj_min4'].max():.4f}, mean {magnitude_df['mean_r2_conj_min4'].mean():.4f}")
    print(f"LOO spatial correlation (Pearson r): per-subject {np.round(loo_r, 4).tolist()}")
    print(f"LOO spatial correlation: mean r = {observed_mean_r:.4f}, range {loo_r.min():.4f}-{loo_r.max():.4f}, permutation p = {p_value:.5f}")
