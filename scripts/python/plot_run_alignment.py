"""
Plot feature timeseries and mean fMRI signal side by side for a set of runs.

Used to visually inspect temporal alignment between behavioral features and
BOLD response. For each run, plots:
  - Top panel: summed feature activity per TR
  - Bottom panel: mean fMRI signal across reference voxels

If features and fMRI are aligned, BOLD rises 2-4 TRs after feature activity
onset. A systematic shift indicates misalignment.

Must be run on the cluster where HCP CIFTI files are accessible.

Outputs:
  - run_alignment_sub{sub}_ses{ses}.png
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.utils.data_loading import get_fmri_filepath, load_practice_metadata

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TR           = PARAMETERS['TR']
FEATURE_COLS = [f for space in sorted(FEATURE_SPACES) for f in FEATURE_SPACES[space]]


def downsampled_tsv_path(subject, session, run, downsampled_path):
    """Return path to downsampled feature TSV for one run."""
    return (
        downsampled_path
        / f'sub-{subject:02d}'
        / f'ses-{session:03d}'
        / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv'
    )


def load_feature_activity(tsv_path):
    """Return summed absolute feature activity per TR as 1D array."""
    df        = pd.read_csv(tsv_path, sep='\t')
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[feat_cols].abs().sum(axis=1).values.astype(float)


def load_fmri_mean(cifti_path, r2_cifti_path, top_n):
    """Return mean BOLD signal -- top-N R2 voxels if R2 CIFTI given, else global mean."""
    data = nib.load(str(cifti_path)).get_fdata().astype(np.float32)
    if r2_cifti_path is not None and r2_cifti_path.exists():
        r2        = nib.load(str(r2_cifti_path)).get_fdata().squeeze()
        threshold = np.sort(r2)[-top_n]
        mask      = r2 >= threshold
        return data[:, mask].mean(axis=1)
    return data.mean(axis=1)


def plot_run_pair(subject, session, runs, pipeline, fmri_path,
                  downsampled_path, r2_cifti_path, top_n, output_path):
    """Plot feature activity and fMRI signal for each run in a single figure."""
    n_runs  = len(runs)
    fig, axes = plt.subplots(n_runs, 2, figsize=(16, 3 * n_runs),
                              sharex=False, sharey='col')

    if n_runs == 1:
        axes = [axes]

    ref_label = f'top-{top_n} R2 voxels' if (r2_cifti_path and r2_cifti_path.exists()) else 'global mean'

    for row, run in enumerate(runs):
        tsv_path   = downsampled_tsv_path(subject, session, run, downsampled_path)
        cifti_path = get_fmri_filepath(subject, session, run, fmri_path, pipeline)

        ax_feat, ax_fmri = axes[row]

        if not tsv_path.exists():
            ax_feat.set_title(f'run-{run:02d} features -- MISSING')
            continue
        if not cifti_path.exists():
            ax_fmri.set_title(f'run-{run:02d} fMRI -- MISSING')
            continue

        logger.info(f"  Loading run-{run:02d}...")
        feat_act = load_feature_activity(tsv_path)
        fmri_sig = load_fmri_mean(cifti_path, r2_cifti_path, top_n)
        n_trs    = min(len(feat_act), len(fmri_sig))
        tr_axis  = np.arange(n_trs) * TR

        ax_feat.fill_between(tr_axis, feat_act[:n_trs], alpha=0.7, color='steelblue')
        ax_feat.set_ylabel('Feature activity')
        ax_feat.set_title(f'run-{run:02d} features')
        ax_feat.set_xlabel('Time (s)')

        ax_fmri.plot(tr_axis, fmri_sig[:n_trs], color='darkorange', linewidth=0.8)
        ax_fmri.set_ylabel('Mean BOLD')
        ax_fmri.set_title(f'run-{run:02d} fMRI ({ref_label})')
        ax_fmri.set_xlabel('Time (s)')

    fig.suptitle(
        f'sub-{subject:02d} ses-{session:03d} -- feature activity vs BOLD signal',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved : {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot feature timeseries vs fMRI signal per run')
    parser.add_argument('--subject',    type=int, required=True)
    parser.add_argument('--session',    type=int, required=True)
    parser.add_argument('--runs',       type=int, nargs='+', default=None,
                        help='Runs to plot (default: all practice runs)')
    parser.add_argument('--r2-cifti',   type=Path, default=None,
                        help='R2 CIFTI for top-voxel fMRI reference (optional)')
    parser.add_argument('--top-voxels', type=int, default=500)
    parser.add_argument('--output-dir', type=Path,
                        default=PATHS['figures'] / 'alignment_check')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline  = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']

    meta = load_practice_metadata(args.subject, args.session, PATHS['practice_phase_metadata'])
    runs = args.runs if args.runs else meta['practice_runs']

    logger.info(f"sub-{args.subject:02d} ses-{args.session:03d} -- plotting {runs}")

    output_path = args.output_dir / f'sub-{args.subject:02d}_ses-{args.session:03d}_run_alignment.png'
    plot_run_pair(
        args.subject, args.session, runs, pipeline, fmri_path,
        PATHS['per_run_downsampled_to_TR'],
        args.r2_cifti, args.top_voxels, output_path,
    )


if __name__ == '__main__':
    main()
