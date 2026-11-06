"""
Make border files outlining the salient (high-R2) parcels for the group map.

wb_command is unusable in this environment (libOSMesa load failure), so instead
of generating borders we filter the shipped Glasser .border files (one closed
<Border Name="?_<parcel>_ROI"> per parcel) down to the salient parcels.

Salient = Glasser parcels with a substantial bright core: at least
--min-bright-frac of the parcel's vertices are FDR-significant AND exceed
--bright-value on the group r_full map. (Selecting by parcel *mean* missed
parcels like 6r/6v/PEF that have a bright core but a dim tail.)

Outputs to the group cifti dir:
    salient.L.border, salient.R.border   load in wb_view as borders
"""
import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import nibabel as nib

from mario_encoding.config import PATHS
from mario_encoding import group_configs

ATLAS = Path("/home/ozvar/Atlases/HCP_S1200_Atlas")
GLASSER360 = ATLAS / ("Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_"
                      "Areas_Group_Colors.32k_fs_LR.dlabel.nii")
BORDER = {
    "L": ATLAS / "Q1-Q6_RelatedValidation210.L.CorticalAreas_dil_Final_Final_Areas_Group.32k_fs_LR.border",
    "R": ATLAS / "Q1-Q6_RelatedValidation210.R.CorticalAreas_dil_Final_Final_Areas_Group.32k_fs_LR.border",
}


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    return logging.getLogger(__name__)


def salient_parcels(cifti_dir: Path, bright_value: float, min_bright_frac: float,
                    logger: logging.Logger) -> set[str]:
    r = nib.load(str(cifti_dir / "group_r_full_mean_observed.dscalar.nii")).get_fdata(dtype=np.float32).squeeze()
    sig = nib.load(str(cifti_dir / "group_r_full_sig_conj_min4.dscalar.nii")).get_fdata(dtype=np.float32).squeeze() > 0.5
    g = nib.load(str(GLASSER360))
    lab = g.get_fdata().squeeze().astype(int)
    names = g.header.get_axis(0).label[0]
    n = lab.shape[0]
    r, sig = r[:n], sig[:n]
    bright = sig & (r > bright_value)
    keep = set()
    logger.info(f"Salient parcels (>= {min_bright_frac:.0%} of vertices significant & r > {bright_value}):")
    rows = []
    for lid in np.unique(lab):
        if lid == 0:
            continue
        p = lab == lid
        if p.sum() < 20:
            continue
        nb = int((bright & p).sum())
        frac = nb / int(p.sum())
        if frac >= min_bright_frac:
            nm = names[lid][0]
            keep.add(nm)
            rows.append((nm, frac, nb, float(r[p].max())))
    for nm, frac, nb, mx in sorted(rows, key=lambda x: x[1], reverse=True):
        logger.info(f"  {nm:<14} frac={frac:.0%}  n_bright={nb:<4} peak_r={mx:.3f}")
    logger.info(f"  -> {len(keep)} parcels")
    return keep


def filter_border(in_path: Path, out_path: Path, keep: set[str], logger: logging.Logger) -> None:
    tree = ET.parse(str(in_path))
    root = tree.getroot()
    cls = root.find("Class")
    removed = 0
    for border in list(cls.findall("Border")):
        if border.get("Name") not in keep:
            cls.remove(border)
            removed += 1
    kept = len(cls.findall("Border"))
    tree.write(str(out_path), encoding="UTF-8", xml_declaration=True)
    logger.info(f"  {out_path.name}: kept {kept}, removed {removed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Glasser borders to salient parcels")
    parser.add_argument("--group", default="GROUP_N5_PRIMARY")
    parser.add_argument("--bright-value", type=float, default=0.5,
                        help="r value above which a vertex counts as 'bright'")
    parser.add_argument("--min-bright-frac", type=float, default=0.20,
                        help="min fraction of a parcel's vertices that must be bright")
    args = parser.parse_args()
    logger = setup_logging()

    label = getattr(group_configs, args.group)["output_label"]
    cifti_dir = PATHS["figures"] / "ciftis" / label

    keep = salient_parcels(cifti_dir, args.bright_value, args.min_bright_frac, logger)
    logger.info("")
    for hemi, src in BORDER.items():
        # keep only this hemisphere's parcels (names are prefixed L_/R_)
        hemi_keep = {nm for nm in keep if nm.startswith(f"{hemi}_")}
        filter_border(src, cifti_dir / f"salient.{hemi}.border", hemi_keep, logger)
    logger.info(f"\nLoad salient.L/R.border in wb_view (Data Series / Borders).")


if __name__ == "__main__":
    main()
