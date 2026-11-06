"""
Summarize group-level full-model R2 by Cole-Anticevic CAB-NP network, cortex-only (59412).

For each of the 12 Cole-Anticevic networks, reports the number of vertices,
mean R2 (restricted to vertices significant in >= 4 of 5 subjects, and
unrestricted), and the fraction of the network meeting that criterion.

Usage:
    python scripts/python/summarize_r2_by_network.py
"""
import logging

import numpy as np
import pandas as pd
from neuro_utils import cortex_only, load_atlas, load_cifti

from mario_encoding.config import PATHS
from mario_encoding.group_configs import GROUP_N5_PRIMARY

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def summarize_networks(r2: np.ndarray, sig_mask: np.ndarray,
                        ca_labels: np.ndarray, ca_names: dict[int, str]) -> pd.DataFrame:
    """Mean R2 per Cole-Anticevic network, restricted and unrestricted by significance."""
    rows = []
    for network_id, name in ca_names.items():
        if network_id == 0:
            continue
        mask = ca_labels == network_id
        n_vertices = int(mask.sum())
        if n_vertices == 0:
            continue
        sig_in_network = mask & sig_mask
        n_sig = int(sig_in_network.sum())
        rows.append({
            'network': name,
            'n_vertices': n_vertices,
            'n_significant': n_sig,
            'pct_significant': 100 * n_sig / n_vertices,
            'mean_r2_significant': r2[sig_in_network].mean() if sig_in_network.any() else np.nan,
            'total_r2_significant': r2[sig_in_network].sum() if sig_in_network.any() else 0.0,
            'mean_r2_all': r2[mask].mean(),
        })
    return pd.DataFrame(rows)


def run(group_cfg: dict = GROUP_N5_PRIMARY) -> pd.DataFrame:
    group_label = group_cfg['output_label']
    cifti_dir = PATHS['figures'] / 'ciftis' / group_label
    r2_path = cifti_dir / 'group_R2_full_mean.dscalar.nii'
    sig_path = cifti_dir / 'group_r_full_sig_conj_min4.dscalar.nii'
    log.info(f"Group: {group_label}")
    log.info(f"R2: {r2_path}")
    log.info(f"Significance mask: {sig_path}")

    r2 = cortex_only(load_cifti(r2_path)[0].squeeze())
    sig_mask = cortex_only(load_cifti(sig_path)[0].squeeze()) > 0
    ca_labels, ca_names = load_atlas('cole_anticevic')
    ca_labels = cortex_only(ca_labels)

    summary = summarize_networks(r2, sig_mask, ca_labels, ca_names)
    out_path = cifti_dir / f'{group_label}_r2_by_network.csv'
    summary.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}")
    return summary


if __name__ == '__main__':
    pd.set_option('display.float_format', '{:.4f}'.format)
    summary = run()

    print("--- ranked by extent (n_significant) ---")
    print(summary.sort_values('n_significant', ascending=False, ignore_index=True).to_string(index=False))
    print()
    print("--- ranked by total variance explained (total_r2_significant) ---")
    print(summary.sort_values('total_r2_significant', ascending=False, ignore_index=True).to_string(index=False))
