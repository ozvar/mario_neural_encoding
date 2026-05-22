"""
Create CIFTI maps for variance partitioning and group PCA results.

Usage:
    # Full group (VP metrics + PCA scores if available)
    python create_vp_model_cifti.py --group GROUP_N3_PRIMARY

    # Single subject (VP metrics only)
    python create_vp_model_cifti.py --subject 1 --train-sessions 7 8 9 10 12 \\
        --test-sessions 14 --experiment-id 20260317_124704
"""
import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.cifti2 import Cifti2Header, Cifti2Image
from nibabel.cifti2.cifti2_axes import ScalarAxis

from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY
from mario_encoding.utils.data_loading import get_template_cifti_path

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}

VP_METRICS_REQUIRED = ['R2_full', 'R2_unique_motor', 'R2_unique_scene', 'R2_shared']
VP_METRICS_OPTIONAL = ['product_measure_motor', 'product_measure_scene']


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def get_dataset_label(subject: int, train_sessions: list[int],
                      test_sessions: list[int]) -> str:
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    return f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'


def get_vp_dir(subject: int, train_sessions: list[int], test_sessions: list[int],
               experiment_id: str, vp_results_path: Path) -> Path:
    dataset_label = get_dataset_label(subject, train_sessions, test_sessions)
    return vp_results_path / dataset_label / experiment_id


def write_cifti_scalar(data: np.ndarray, template_path: Path, output_path: Path,
                       map_name: str, logger: logging.Logger) -> None:
    """Write a (91282,) float32 array as a CIFTI dscalar file."""
    template = nib.load(str(template_path))
    cifti_data = data.astype(np.float32)[np.newaxis, :]
    brain_model_axis = template.header.get_axis(1)
    scalar_axis = ScalarAxis([map_name])
    header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    img = Cifti2Image(cifti_data, header=header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.to_filename(str(output_path))
    logger.info(f"  Saved: {output_path.name}")


def expand_to_fullbrain(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Expand a masked array (n_valid,) to full brain (91282,) with NaN padding."""
    full = np.full(len(mask), np.nan, dtype=np.float32)
    full[mask] = scores
    return full


def create_vp_ciftis(subject: int, train_sessions: list[int], test_sessions: list[int],
                     experiment_id: str, vp_results_path: Path, cifti_output_dir: Path,
                     fmri_path: Path, pipeline: str, logger: logging.Logger) -> list[Path]:
    """Create VP metric CIFTIs for one subject."""
    vp_dir = get_vp_dir(subject, train_sessions, test_sessions, experiment_id, vp_results_path)
    if not vp_dir.exists():
        raise FileNotFoundError(f"VP results not found: {vp_dir}")

    r2_file = vp_dir / 'R2_scores.npz'
    mask_file = vp_dir / 'valid_voxels_mask.npy'
    if not r2_file.exists():
        raise FileNotFoundError(f"R2_scores.npz not found: {r2_file}")
    if not mask_file.exists():
        raise FileNotFoundError(f"valid_voxels_mask.npy not found: {mask_file}")

    r2_data = np.load(r2_file)
    mask = np.load(mask_file)
    logger.info(f"  R2 keys: {list(r2_data.keys())}, valid voxels: {mask.sum()}")

    template_path = get_template_cifti_path(subject, fmri_path, pipeline)
    logger.info(f"  Template: {template_path}")

    dataset_label = get_dataset_label(subject, train_sessions, test_sessions)
    prefix = f'{dataset_label}_{experiment_id}'
    created = []

    for key in VP_METRICS_REQUIRED:
        if key not in r2_data:
            logger.warning(f"  Required key '{key}' not in R2_scores.npz -- skipping")
            continue
        out_path = cifti_output_dir / f'{prefix}_{key}.dscalar.nii'
        write_cifti_scalar(expand_to_fullbrain(r2_data[key], mask), template_path,
                           out_path, key, logger)
        created.append(out_path)

    for key in VP_METRICS_OPTIONAL:
        if key not in r2_data:
            logger.info(f"  Optional key '{key}' not present -- skipping")
            continue
        out_path = cifti_output_dir / f'{prefix}_{key}.dscalar.nii'
        write_cifti_scalar(expand_to_fullbrain(r2_data[key], mask), template_path,
                           out_path, key, logger)
        created.append(out_path)

    return created


def create_pca_ciftis(subject: int, group_label: str, pca_output_dir: Path,
                      cifti_output_dir: Path, fmri_path: Path, pipeline: str,
                      logger: logging.Logger) -> list[Path]:
    """Create PCA score CIFTIs for one subject from group PCA output.

    Score maps are (n_pcs, 91282) with NaN for invalid voxels -- no mask expansion needed.
    """
    score_map_file = pca_output_dir / f'pca_score_maps_sub-{subject:02d}.npy'
    if not score_map_file.exists():
        logger.warning(f"  PCA score maps not found: {score_map_file} -- skipping")
        return []

    score_maps = np.load(score_map_file)  # (n_pcs, 91282)
    n_pcs = score_maps.shape[0]
    logger.info(f"  PCA score maps: {score_maps.shape}")

    template_path = get_template_cifti_path(subject, fmri_path, pipeline)
    created = []
    for k in range(n_pcs):
        map_name = f'PC{k + 1}'
        out_path = cifti_output_dir / f'sub-{subject:02d}_{group_label}_{map_name}.dscalar.nii'
        write_cifti_scalar(score_maps[k], template_path, out_path, map_name, logger)
        created.append(out_path)

    return created


def process_subject(subject_cfg: dict, group_label: str | None,
                    pca_output_dir: Path | None, cifti_output_dir: Path,
                    vp_results_path: Path, fmri_path: Path, pipeline: str,
                    logger: logging.Logger) -> None:
    subject = subject_cfg['subject']
    train_sessions = subject_cfg['train_sessions']
    test_sessions = subject_cfg['test_sessions']
    experiment_id = subject_cfg['experiment_id']

    logger.info(f"sub-{subject:02d}: creating VP CIFTIs")
    vp_created = create_vp_ciftis(
        subject, train_sessions, test_sessions, experiment_id,
        vp_results_path, cifti_output_dir, fmri_path, pipeline, logger,
    )
    logger.info(f"sub-{subject:02d}: created {len(vp_created)} VP CIFTIs")

    if pca_output_dir is not None:
        logger.info(f"sub-{subject:02d}: creating PCA score CIFTIs")
        pca_created = create_pca_ciftis(
            subject, group_label, pca_output_dir, cifti_output_dir,
            fmri_path, pipeline, logger,
        )
        logger.info(f"sub-{subject:02d}: created {len(pca_created)} PCA CIFTIs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create CIFTI maps for VP and group PCA results'
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--group', type=str, choices=list(GROUP_REGISTRY.keys()),
                      help='Process all subjects in a group config')
    mode.add_argument('--subject', type=int, help='Single subject number')
    parser.add_argument('--train-sessions', type=int, nargs='+')
    parser.add_argument('--test-sessions', type=int, nargs='+')
    parser.add_argument('--experiment-id', type=str)
    args = parser.parse_args()

    pipeline = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']
    vp_results_path = PATHS['variance_partitioning']

    if args.group:
        group_cfg = GROUP_REGISTRY[args.group]
        group_label = group_cfg['output_label']
        cifti_output_dir = PATHS['figures'] / 'ciftis' / group_label
        pca_output_dir = PATHS['pca'] / group_label
        if not pca_output_dir.exists():
            pca_output_dir = None

        log_path = PATHS['logs'] / f'{group_label}_create_ciftis.log'
        logger = setup_logging(log_path)

        logger.info("=" * 80)
        logger.info("CREATE VP AND PCA CIFTIS")
        logger.info("=" * 80)
        logger.info(f"Group: {args.group}")
        logger.info(f"Pipeline: {pipeline}")
        logger.info(f"Output: {cifti_output_dir}")
        logger.info(f"PCA source: {pca_output_dir if pca_output_dir else 'not found -- PCA CIFTIs skipped'}")

        for subject_cfg in group_cfg['subjects']:
            logger.info("")
            process_subject(
                subject_cfg, group_label, pca_output_dir, cifti_output_dir,
                vp_results_path, fmri_path, pipeline, logger,
            )

    else:
        if not all([args.train_sessions, args.test_sessions, args.experiment_id]):
            parser.error('--subject requires --train-sessions, --test-sessions, and --experiment-id')

        subject_cfg = {
            'subject': args.subject,
            'train_sessions': args.train_sessions,
            'test_sessions': args.test_sessions,
            'experiment_id': args.experiment_id,
        }
        dataset_label = get_dataset_label(args.subject, args.train_sessions, args.test_sessions)
        cifti_output_dir = PATHS['figures'] / 'ciftis' / dataset_label / args.experiment_id
        log_path = PATHS['logs'] / f'{dataset_label}_{args.experiment_id}_create_ciftis.log'
        logger = setup_logging(log_path)

        logger.info("=" * 80)
        logger.info("CREATE VP CIFTIS")
        logger.info("=" * 80)
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Train sessions: {args.train_sessions}")
        logger.info(f"Test sessions: {args.test_sessions}")
        logger.info(f"Experiment ID: {args.experiment_id}")
        logger.info(f"Pipeline: {pipeline}")
        logger.info(f"Output: {cifti_output_dir}")

        process_subject(
            subject_cfg, None, None, cifti_output_dir,
            vp_results_path, fmri_path, pipeline, logger,
        )

    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
