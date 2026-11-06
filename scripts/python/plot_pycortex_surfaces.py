"""
Create pycortex visualizations from CIFTI variance partitioning results.

This script generates flatmap and inflated surface visualizations for individual
subjects or group averages using pycortex.

Output structure:
    {PATHS['pycortex']}/
        {experiment_id}/
            sub-{XX}_test_r2_flatmap.png
            sub-{XX}_test_r2_inflated.png
            group_R2_full_flatmap.png
            group_R2_full_inflated.png
            visualization.log
"""
import argparse
import logging
import cortex
from pathlib import Path
from datetime import datetime

import numpy as np
import hashlib
import nibabel as nib
import matplotlib.pyplot as plt

from mario_encoding.config import PATHS


# Monkey-patch the broken hash function
#def _hash_fixed(array):
#    """Fixed numpy hash function."""
#    return hashlib.sha1(array.tobytes()).hexdigest()
#
#import cortex.dataset.braindata
#cortex.dataset.braindata._hash = _hash_fixed

def setup_logging(output_dir):
    """Configure logging to file and console."""
    log_file = output_dir / 'visualization.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)


def construct_cifti_path(subject, train_sessions, test_sessions, experiment_id,
                        metric='test_r2', is_group=False):
    """
    Construct path to CIFTI file.

    Parameters:
    -----------
    subject : int or None
    train_sessions : list of int
    test_sessions : list of int
    experiment_id : str
    metric : str
    is_group : bool

    Returns:
    --------
    cifti_path : Path
    """
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"

    if is_group:
        base_dir = PATHS['variance_partitioning'] / 'group_average' / experiment_id
        filename = f'group_{metric}_mean.dscalar.nii'
    else:
        dataset_dir = f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}'
        base_dir = PATHS['figures'] / 'model_ciftis' / dataset_dir / experiment_id
        filename = f'{dataset_dir}_{experiment_id}_{metric}.dscalar.nii'

    return base_dir / filename


def construct_output_path(output_dir, subject, metric, viz_type, is_group=False):
    """
    Construct output path for visualization.

    Parameters:
    -----------
    output_dir : Path
    subject : int or None
    metric : str
    viz_type : str
    is_group : bool

    Returns:
    --------
    output_path : Path
    """
    if is_group:
        filename = f'group_{metric}_{viz_type}.png'
    else:
        filename = f'sub-{subject:02d}_{metric}_{viz_type}.png'
    return output_dir / filename


def load_cifti_to_vertices(cifti_path, pycortex_subject='hcp_32k'):
    """
    Load CIFTI and convert to pycortex vertex format.

    Parameters:
    -----------
    cifti_path : Path
    pycortex_subject : str

    Returns:
    --------
    vertex_data : array of shape (n_vertices,)
    """
    cifti = nib.load(str(cifti_path))
    data = cifti.get_fdata()[0]
    brain_axis = cifti.header.get_axis(1)

    left_indices = []
    left_data = []
    right_indices = []
    right_data = []

    for name, data_indices, model in brain_axis.iter_structures():
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            left_indices = list(model.vertex)
            left_data = data[data_indices]
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            right_indices = list(model.vertex)
            right_data = data[data_indices]

    left_full = np.full(32492, np.nan)
    right_full = np.full(32492, np.nan)

    left_full[left_indices] = left_data
    right_full[right_indices] = right_data

    vertex_data = np.concatenate([left_full, right_full])
    return vertex_data


def create_flatmap(vertex_data, output_path, vmin=0, vmax=0.3, cmap='hot',
                  with_rois=False, with_sulci=True, with_labels=True,
                  height=2048, dpi=300):
    """
    Create and save flatmap visualization.

    Parameters:
    -----------
    vertex_data : array
    output_path : Path
    vmin : float
    vmax : float
    cmap : str
    with_rois : bool
    with_sulci : bool
    with_labels : bool
    height : int
    dpi : int
    """
    vertex_data_clean = np.nan_to_num(vertex_data, nan=0.0)
    vertex_obj = cortex.Vertex(vertex_data_clean, 'hcp_32k', vmin=vmin, vmax=vmax, cmap=cmap)
    fig = cortex.quickshow(vertex_obj, with_rois=with_rois, with_sulci=with_sulci,
                          with_labels=with_labels, height=height)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def create_inflated_views(vertex_data, output_path, vmin=0, vmax=0.3, cmap='hot'):
    """
    Create and save inflated surface views using pycortex rendering.

    Parameters:
    -----------
    vertex_data : array
    output_path : Path
    vmin : float
    vmax : float
    cmap : str
    """
    from cortex.export import save_3d_views

    # Convert NaN to 0 for webgl compatibility
    vertex_data_clean = np.nan_to_num(vertex_data, nan=0.0)
    vertex_obj = cortex.Vertex(vertex_data_clean, 'hcp_32k', vmin=vmin, vmax=vmax, cmap=cmap)

    list_angles = ['lateral_pivot', 'medial_pivot']
    list_surfaces = ['inflated', 'inflated']

    viewer_params = {
        'labels_visible': [],
        'overlays_visible': [],
        'bg_color': (1, 1, 1, 1)
    }
    handle = cortex.webshow(vertex_obj, open_browser=True)
    # save_3d_views expects base name without extension
    base_name = str(output_path.parent / output_path.stem)

    save_3d_views(
        vertex_obj,
        base_name,
        list_angles=list_angles,
        list_surfaces=list_surfaces,
        viewer_params=viewer_params,
        size=(8192, 6144),
        sleep=5
    )


def visualize_metric(subject, train_sessions, test_sessions, experiment_id,
                    metric, viz_type, output_dir, is_group, vmin, vmax, cmap, logger):
    """
    Create visualization for a single metric.

    Parameters:
    -----------
    subject : int or None
    train_sessions : list of int
    test_sessions : list of int
    experiment_id : str
    metric : str
    viz_type : str
    output_dir : Path
    is_group : bool
    vmin : float
    vmax : float
    cmap : str
    logger : logging.Logger
    """
    cifti_path = construct_cifti_path(subject, train_sessions, test_sessions,
                                      experiment_id, metric, is_group)

    if not cifti_path.exists():
        logger.warning(f"CIFTI not found: {cifti_path}")
        return

    vertex_data = load_cifti_to_vertices(cifti_path)
    output_path = construct_output_path(output_dir, subject, metric, viz_type, is_group)

    if viz_type == 'flatmap':
        create_flatmap(vertex_data, output_path, vmin=vmin, vmax=vmax, cmap=cmap)
    elif viz_type == 'inflated':
        create_inflated_views(vertex_data, output_path, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        raise ValueError(f"Unknown visualization type: {viz_type}")

    subject_str = 'group' if is_group else f'sub-{subject:02d}'
    logger.info(f"Created {viz_type}: {subject_str} {metric}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create pycortex visualizations from variance partitioning results'
    )
    parser.add_argument('--subjects', type=int, nargs='+', required=True,
                       help='Subject numbers (e.g., 1 2 3)')
    parser.add_argument('--train-sessions', type=int, nargs='+', required=True,
                       help='Training session numbers')
    parser.add_argument('--test-sessions', type=int, nargs='+', required=True,
                       help='Test session numbers')
    parser.add_argument('--experiment-id', type=str, required=True,
                       help='Experiment ID (timestamp format: YYYYmmdd_HHMMSS)')
    parser.add_argument('--metrics', type=str, nargs='+', default=['test_r2'],
                       help='Metrics to visualize (e.g., test_r2 unique_variance_motor)')
    parser.add_argument('--viz-types', type=str, nargs='+', default=['flatmap'],
                       choices=['flatmap', 'inflated'],
                       help='Visualization types to generate')
    parser.add_argument('--include-group', action='store_true',
                       help='Also visualize group average (if available)')
    parser.add_argument('--vmin', type=float, default=0,
                       help='Minimum value for colormap')
    parser.add_argument('--vmax', type=float, default=0.3,
                       help='Maximum value for colormap')
    parser.add_argument('--cmap', type=str, default='hot',
                       help='Matplotlib colormap name')
    args = parser.parse_args()

    output_dir = PATHS['pycortex'] / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("="*80)
    logger.info("PYCORTEX VISUALIZATION")
    logger.info("="*80)
    logger.info(f"Subjects: {args.subjects}")
    logger.info(f"Metrics: {args.metrics}")
    logger.info(f"Visualization types: {args.viz_types}")
    logger.info(f"Experiment ID: {args.experiment_id}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    for subject in args.subjects:
        for metric in args.metrics:
            for viz_type in args.viz_types:
                visualize_metric(subject, args.train_sessions, args.test_sessions,
                               args.experiment_id, metric, viz_type, output_dir,
                               is_group=False, vmin=args.vmin, vmax=args.vmax,
                               cmap=args.cmap, logger=logger)

    if args.include_group:
        for metric in args.metrics:
            for viz_type in args.viz_types:
                visualize_metric(None, args.train_sessions, args.test_sessions,
                               args.experiment_id, metric, viz_type, output_dir,
                               is_group=True, vmin=args.vmin, vmax=args.vmax,
                               cmap=args.cmap, logger=logger)

    logger.info("")
    logger.info("="*80)
    logger.info("COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")


if __name__ == '__main__':
    main()
