"""
Inspect the per-band alpha hyperparameters selected by GroupRidgeCV.

Loads `hyperparameters_full.npz` (saved by fit_variance_partitioning.py post-A2)
for each subject in a group config and reports the distribution of selected
per-band alphas. Use this to decide whether the alpha grid in
PARAMETERS['variance_partitioning']['solver_params']['alphas'] is appropriately
ranged or wasting random-search budget at infeasible extremes.

What to look for:
- Per-band alpha medians inside the grid (good).
- Mass piling up at the top of the grid (10^alpha_max): regularization saturated;
  the model wants to predict zero. Grid is too narrow at the high end OR features
  have no predictive value for those voxels; extend up if the latter looks wrong.
- Mass piling up at the bottom (10^alpha_min): under-regularized. Grid is too narrow
  at the low end; extend down.
- Both bands distributing similarly: per-band regularization is doing its job.
- One band always tiny, the other always huge: that band is effectively dropped;
  inspect feature scales (B2 discussion).

Usage:
    python scripts/python/inspect_alpha_grid.py --group GROUP_N5_PRIMARY
    python scripts/python/inspect_alpha_grid.py --group GROUP_N5_PRIMARY --plot
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
    )
    return logging.getLogger(__name__)


def get_vp_dir(subject: int, train_sessions: list, test_sessions: list,
               experiment_id: str) -> Path:
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    return PATHS['variance_partitioning'] / dataset_dir / experiment_id


def load_hyperparams(vp_dir: Path, logger: logging.Logger) -> dict:
    hyper_file = vp_dir / 'hyperparameters_full.npz'
    if not hyper_file.exists():
        raise FileNotFoundError(
            f"Hyperparameter file not found: {hyper_file}. "
            f"This VP fit predates the A3 saving change; refit to populate."
        )
    data = np.load(hyper_file, allow_pickle=False)
    return {
        'deltas': data['deltas'],
        'best_alphas': data['best_alphas'],
        'per_band_alpha': data['per_band_alpha'],
        'space_names': [s.decode() if isinstance(s, bytes) else str(s)
                        for s in data['space_names']],
    }


def summarize_band(name: str, alphas: np.ndarray, grid_min: float, grid_max: float,
                   logger: logging.Logger, edge_tol: float = 0.5) -> dict:
    """
    Report percentiles and grid-edge clipping for a per-voxel alpha vector.

    edge_tol is in log10 units: voxels within `edge_tol` decades of the grid
    edge are flagged as potentially clipped.
    """
    log_alpha = np.log10(np.maximum(alphas, np.finfo(np.float32).tiny))
    log_min = np.log10(grid_min)
    log_max = np.log10(grid_max)

    pct = np.percentile(log_alpha, [1, 5, 25, 50, 75, 95, 99])
    near_low = float(np.mean(log_alpha < log_min + edge_tol))
    near_high = float(np.mean(log_alpha > log_max - edge_tol))

    logger.info(f"  {name}:")
    logger.info(f"    log10(alpha) percentiles "
                f"(1/5/25/50/75/95/99): {pct.round(2).tolist()}")
    logger.info(f"    grid log10 range: [{log_min:.1f}, {log_max:.1f}]")
    logger.info(f"    voxels within {edge_tol} decades of low edge: {near_low * 100:.1f}%")
    logger.info(f"    voxels within {edge_tol} decades of high edge: {near_high * 100:.1f}%")

    return {
        'log10_percentiles': pct,
        'frac_near_low': near_low,
        'frac_near_high': near_high,
        'log_min': log_min,
        'log_max': log_max,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=str, required=True,
                        choices=list(GROUP_REGISTRY.keys()))
    parser.add_argument('--plot', action='store_true',
                        help='Save log10(alpha) histogram per band per subject')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Where to write plots (default: results/figures/alpha_grid/)')
    parser.add_argument('--edge-tol', type=float, default=0.5,
                        help='Decades from grid edge considered clipping (default 0.5)')
    args = parser.parse_args()

    logger = setup_logging()

    group_cfg = GROUP_REGISTRY[args.group]
    alphas_grid = np.asarray(PARAMETERS['variance_partitioning']['solver_params']['alphas'])
    grid_min, grid_max = float(alphas_grid.min()), float(alphas_grid.max())

    logger.info(f"Group: {args.group} (label={group_cfg['output_label']})")
    logger.info(f"Configured alpha grid: {alphas_grid.size} values from "
                f"{grid_min:.3g} to {grid_max:.3g} (log10: "
                f"{np.log10(grid_min):.1f} to {np.log10(grid_max):.1f})")

    per_subject_alphas = {}
    pooled = {}

    for sub_cfg in group_cfg['subjects']:
        sub = sub_cfg['subject']
        vp_dir = get_vp_dir(sub, sub_cfg['train_sessions'],
                            sub_cfg['test_sessions'], sub_cfg['experiment_id'])
        logger.info("")
        logger.info(f"sub-{sub:02d} | {vp_dir}")
        try:
            data = load_hyperparams(vp_dir, logger)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        space_names = data['space_names']
        per_band_alpha = data['per_band_alpha']  # (n_bands, n_voxels)

        per_subject_alphas[sub] = {'space_names': space_names, 'alphas': per_band_alpha}

        for idx, name in enumerate(space_names):
            summarize_band(name, per_band_alpha[idx], grid_min, grid_max,
                           logger, edge_tol=args.edge_tol)
            pooled.setdefault(name, []).append(per_band_alpha[idx])

    if not per_subject_alphas:
        logger.error("No hyperparameter files found for any subject. Aborting.")
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info("POOLED ACROSS SUBJECTS")
    logger.info("=" * 80)
    for name, alpha_list in pooled.items():
        pooled_arr = np.concatenate(alpha_list)
        summarize_band(name, pooled_arr, grid_min, grid_max, logger,
                       edge_tol=args.edge_tol)

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        out_dir = args.output_dir or (PATHS['figures'] / 'alpha_grid')
        out_dir.mkdir(parents=True, exist_ok=True)

        n_subs = len(per_subject_alphas)
        space_names = next(iter(per_subject_alphas.values()))['space_names']
        n_bands = len(space_names)

        fig, axes = plt.subplots(n_subs, n_bands, figsize=(4 * n_bands, 2.5 * n_subs),
                                 squeeze=False, sharex=True)
        log_min, log_max = np.log10(grid_min), np.log10(grid_max)
        bins = np.linspace(log_min - 1, log_max + 1, 60)

        for row, (sub, payload) in enumerate(per_subject_alphas.items()):
            for col, name in enumerate(payload['space_names']):
                ax = axes[row, col]
                ax.hist(np.log10(payload['alphas'][col]), bins=bins,
                        color='steelblue', alpha=0.8)
                ax.axvline(log_min, color='red', linestyle=':', linewidth=1)
                ax.axvline(log_max, color='red', linestyle=':', linewidth=1)
                ax.set_title(f"sub-{sub:02d} | {name}")
                if row == n_subs - 1:
                    ax.set_xlabel('log10(alpha)')
                if col == 0:
                    ax.set_ylabel('n voxels')

        fig.suptitle(f'{args.group}: per-band alpha distributions '
                     f'(grid dotted in red)', y=1.02)
        fig.tight_layout()
        out_file = out_dir / f'{group_cfg["output_label"]}_alpha_grid.png'
        fig.savefig(out_file, dpi=120, bbox_inches='tight')
        logger.info(f"Plot saved to: {out_file}")


if __name__ == '__main__':
    main()
