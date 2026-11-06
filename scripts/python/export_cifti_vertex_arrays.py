"""
Export CIFTI dscalar maps to per-hemisphere vertex arrays for use in R.

Reads motor PM, scene PM, motor sig, and scene sig CIFTIs and writes one
.npy file per hemisphere per map to a staging directory. Values are placed
at the correct 32k fs_LR vertex indices; medial-wall vertices are NaN.

Usage:
    python export_cifti_vertex_arrays.py --group GROUP_N5_PRIMARY [--out-dir DIR]
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import nibabel as nib

from mario_encoding.config import PATHS
from mario_encoding import group_configs


N_VERTS = 32492


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    return logging.getLogger(__name__)


def cifti_to_vertex_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_full, right_full) each of shape (32492,), NaN at medial wall."""
    cifti = nib.load(str(path))
    data = cifti.get_fdata()[0]
    brain_axis = cifti.header.get_axis(1)
    left_full = np.full(N_VERTS, np.nan)
    right_full = np.full(N_VERTS, np.nan)
    for name, data_indices, model in brain_axis.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            left_full[list(model.vertex)] = data[data_indices]
        elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            right_full[list(model.vertex)] = data[data_indices]
    return left_full, right_full


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CIFTI vertex arrays for R")
    parser.add_argument("--group", default="GROUP_N5_PRIMARY")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: results/r_cifti_arrays/<group_label>)")
    args = parser.parse_args()
    logger = setup_logging()

    group_cfg = getattr(group_configs, args.group)
    label = group_cfg["output_label"]

    ga_cands = sorted((PATHS["variance_partitioning"] / "group_average").glob(
        "*/group_product_measure_motor.dscalar.nii"))
    if not ga_cands:
        raise FileNotFoundError("No group_product_measure_motor.dscalar.nii found")
    ga_dir = ga_cands[-1].parent

    cifti_dir = PATHS["figures"] / "ciftis" / label

    out_dir = args.out_dir or (PATHS["figures"].parent / "r_cifti_arrays" / label)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = {
        "motor_pm":    ga_dir / "group_product_measure_motor.dscalar.nii",
        "scene_pm":    ga_dir / "group_product_measure_scene.dscalar.nii",
        "motor_sig":   cifti_dir / "group_pm_motor_sig_conj_min4.dscalar.nii",
        "scene_sig":   cifti_dir / "group_pm_scene_sig_conj_min4.dscalar.nii",
    }

    for name, path in maps.items():
        logger.info(f"Reading {name}: {path}")
        left, right = cifti_to_vertex_arrays(path)
        np.save(out_dir / f"{name}_L.npy", left)
        np.save(out_dir / f"{name}_R.npy", right)
        logger.info(f"  finite L={np.isfinite(left).sum()}  finite R={np.isfinite(right).sum()}")

    logger.info(f"\nWrote 8 arrays to {out_dir}")


if __name__ == "__main__":
    main()
