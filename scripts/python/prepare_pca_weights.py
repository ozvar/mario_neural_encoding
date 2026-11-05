"""
Rescale and prepare encoding model weights for PCA.

Loads weights saved by fit_variance_partitioning.py, applies the per-voxel
rescaling from Dupre la Tour et al. (2025) to correct for the scale bias
introduced by per-voxel regularization, averages across FIR delays, and
saves the rescaled joint (motor + scene) weight matrix for group PCA.

Rescaling procedure (Dupre la Tour et al. 2025):
    factor = sqrt(max(0, R2_full)) / norm(weights_full, per voxel)
    weights_per_delay *= factor[None, None, :]   # broadcast across delays
    weights_avg = weights_per_delay.mean(axis=0)

Notes on what is and isn't preserved:
- The scaling factor uses the norm of the full delay-expanded weight
  matrix so each voxel's factor reflects total weight magnitude across all
  features and delays. This matches Dupre la Tour 2025.
- Per-voxel unit-norm is NOT preserved after delay averaging: averaging
  across non-identical FIR delays generally shrinks the resulting vector
  norm. The output is "sum-across-delays of rescaled weights / n_delays",
  i.e. an R2-weighted tuning vector per voxel, not a unit-norm tuning vector.
- R2_full is saved alongside the weights so downstream code (fit_pca.py)
  can apply a voxel-selection threshold.

Usage:
    # Single subject
    python prepare_pca_weights.py --subject 1 --train-sessions 7 8 9 10 12 \\
        --test-sessions 14 --experiment-id 20260317_124704

    # Full group (reads configs from group_configs.py)
    python prepare_pca_weights.py --group GROUP_N3_PRIMARY
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from mario_encoding.config import PATHS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}


def setup_logging(subject: int, output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'sub-{subject:02d}_prepare_pca_weights.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )
    return logging.getLogger(__name__)


def get_dirs(subject: int, train_sessions: list[int], test_sessions: list[int],
             experiment_id: str) -> tuple[Path, Path]:
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
    vp_dir = PATHS['variance_partitioning'] / dataset_dir / experiment_id
    output_dir = PATHS['pca'] / dataset_dir / experiment_id
    return vp_dir, output_dir


def load_vp_outputs(vp_dir: Path, logger: logging.Logger) -> tuple:
    logger.info(f"Loading VP outputs from: {vp_dir}")

    weights_file = vp_dir / 'weights_full.npz'
    r2_file = vp_dir / 'R2_scores.npz'
    mask_file = vp_dir / 'valid_voxels_mask.npy'
    metadata_file = vp_dir / 'metadata.json'

    for f in [weights_file, r2_file, mask_file, metadata_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    weights_data = np.load(weights_file)
    r2_data = np.load(r2_file)
    valid_voxels_mask = np.load(mask_file)

    with open(metadata_file) as f:
        metadata = json.load(f)

    weights_full = weights_data['weights_full']
    motor_per_delay = weights_data['motor_weights_per_delay']
    scene_per_delay = weights_data['scene_weights_per_delay']
    R2_full = r2_data['R2_full']

    logger.info(f"  weights_full shape: {weights_full.shape}")
    logger.info(f"  motor_weights_per_delay shape: {motor_per_delay.shape}")
    logger.info(f"  scene_weights_per_delay shape: {scene_per_delay.shape}")
    logger.info(f"  R2_full shape: {R2_full.shape}, mean: {R2_full.mean():.4f}")
    logger.info(f"  valid_voxels_mask: {valid_voxels_mask.sum()} voxels")

    return weights_full, motor_per_delay, scene_per_delay, R2_full, valid_voxels_mask, metadata


def compute_rescale_factors(weights_full: np.ndarray, R2_full: np.ndarray,
                            logger: logging.Logger) -> np.ndarray:
    """
    Compute per-voxel rescale factors from the full delay-expanded weight matrix.

    Factor = sqrt(max(0, R2)) / norm(weights_full, per voxel)

    Using weights_full for the norm ensures the scale reflects total weight
    magnitude across all features and delays before the rescaling is applied.
    """
    logger.info("Computing per-voxel rescale factors...")
    norms = np.linalg.norm(weights_full, axis=0)
    logger.info(f"  Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
    logger.info(f"  Zero-norm voxels: {(norms == 0).sum()}")

    scale = np.sqrt(np.maximum(0, R2_full))
    rescale_factors = scale / (norms + 1e-10)

    logger.info(f"  Rescale factor range: [{rescale_factors.min():.4f}, {rescale_factors.max():.4f}]")
    return rescale_factors


def rescale_and_average(per_delay: np.ndarray, rescale_factors: np.ndarray,
                        space_name: str, logger: logging.Logger) -> np.ndarray:
    """
    Apply per-voxel rescale factors to a (n_delays, n_features, n_voxels) array
    and average across delays to yield (n_features, n_voxels).
    """
    logger.info(f"  Rescaling {space_name}: {per_delay.shape}")
    rescaled = per_delay * rescale_factors[None, None, :]
    averaged = rescaled.mean(axis=0)
    logger.info(f"    -> averaged shape: {averaged.shape}")
    return averaged


def build_joint_matrix(motor_avg: np.ndarray, scene_avg: np.ndarray,
                       logger: logging.Logger) -> np.ndarray:
    weights_joint = np.concatenate([motor_avg, scene_avg], axis=0)
    logger.info(f"Joint weight matrix: {weights_joint.shape} "
                f"(motor {motor_avg.shape[0]} + scene {scene_avg.shape[0]} features)")
    return weights_joint


def save_pca_weights(weights_joint: np.ndarray, motor_avg: np.ndarray,
                     scene_avg: np.ndarray, valid_voxels_mask: np.ndarray,
                     R2_full: np.ndarray, metadata: dict,
                     output_dir: Path, logger: logging.Logger) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_file = output_dir / 'pca_weights.npz'
    np.savez_compressed(
        weights_file,
        weights_joint=weights_joint,
        motor_weights=motor_avg,
        scene_weights=scene_avg,
        valid_voxels_mask=valid_voxels_mask,
        R2_full=R2_full,
    )
    logger.info(f"Saved weights to: {weights_file}")

    meta_out = {
        'n_features_total': int(weights_joint.shape[0]),
        'n_features_motor': int(motor_avg.shape[0]),
        'n_features_scene': int(scene_avg.shape[0]),
        'n_voxels': int(weights_joint.shape[1]),
        'feature_spaces': metadata.get('feature_spaces', {}),
        'rescaling': 'dupre_la_tour_2025'
    }
    meta_file = output_dir / 'pca_weights_metadata.json'
    with open(meta_file, 'w') as f:
        json.dump(meta_out, f, indent=2)
    logger.info(f"Saved metadata to: {meta_file}")


def process_subject(subject_cfg: dict) -> None:
    subject = subject_cfg['subject']
    train_sessions = subject_cfg['train_sessions']
    test_sessions = subject_cfg['test_sessions']
    experiment_id = subject_cfg['experiment_id']

    vp_dir, output_dir = get_dirs(subject, train_sessions, test_sessions, experiment_id)
    logger = setup_logging(subject, output_dir)

    logger.info("=" * 80)
    logger.info("PREPARE PCA WEIGHTS")
    logger.info("=" * 80)
    logger.info(f"Subject: {subject}")
    logger.info(f"Train sessions: {train_sessions}")
    logger.info(f"Test sessions: {test_sessions}")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"VP dir: {vp_dir}")
    logger.info(f"Output dir: {output_dir}")

    if not vp_dir.exists():
        raise FileNotFoundError(f"VP output directory not found: {vp_dir}")

    weights_full, motor_per_delay, scene_per_delay, R2_full, valid_voxels_mask, metadata = \
        load_vp_outputs(vp_dir, logger)

    logger.info("")
    rescale_factors = compute_rescale_factors(weights_full, R2_full, logger)
    del weights_full

    logger.info("")
    logger.info("Rescaling and averaging across delays...")
    motor_avg = rescale_and_average(motor_per_delay, rescale_factors, 'motor', logger)
    scene_avg = rescale_and_average(scene_per_delay, rescale_factors, 'scene', logger)

    logger.info("")
    weights_joint = build_joint_matrix(motor_avg, scene_avg, logger)

    logger.info("")
    save_pca_weights(weights_joint, motor_avg, scene_avg, valid_voxels_mask,
                     R2_full, metadata, output_dir, logger)

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description='Rescale VP model weights for PCA')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--group', type=str, choices=list(GROUP_REGISTRY.keys()),
                      help='Process all subjects in a group config')
    mode.add_argument('--subject', type=int,
                      help='Single subject number')

    parser.add_argument('--train-sessions', type=int, nargs='+')
    parser.add_argument('--test-sessions', type=int, nargs='+')
    parser.add_argument('--experiment-id', type=str)
    args = parser.parse_args()

    if args.group:
        group_cfg = GROUP_REGISTRY[args.group]
        for subject_cfg in group_cfg['subjects']:
            process_subject(subject_cfg)
    else:
        if not all([args.train_sessions, args.test_sessions, args.experiment_id]):
            parser.error('--subject requires --train-sessions, --test-sessions, and --experiment-id')
        process_subject({
            'subject': args.subject,
            'train_sessions': args.train_sessions,
            'test_sessions': args.test_sessions,
            'experiment_id': args.experiment_id,
        })


if __name__ == '__main__':
    main()
