"""
Define variance-partitioning ROIs from the group product-measure maps.

Each cortical vertex carries motor (M = PM_motor) and scene (S = PM_scene)
product measures, with M + S ~ R2_full. Rather than the four-way significance
classification (which is ~83% "both" and cannot isolate unimodal regions), ROIs
are defined from three scores under a significance + magnitude gate:

    shared      = min(M, S)   (both bands substantially encoded)
    motor-lean  = M - S        (motor dominates)
    scene-lean  = S - M        (scene dominates)

Significance gates use the >=4/5 partial-conjunction masks
(group_pm_*_sig_conj_min4.dscalar.nii), not the deprecated Stouffer masks.

Gates per category:
    shared : motor-sig AND scene-sig AND M > PM_THRESHOLD AND S > PM_THRESHOLD
    motor  : motor-sig AND M > PM_THRESHOLD AND (M - S) > DELTA
    scene  : scene-sig AND S > PM_THRESHOLD AND (S - M) > DELTA

The ROI set (ROI_SPEC below) is editable; rerun the script to regenerate.

Outputs (into the group cifti dir):
    roi_qualifying_masks.npz   per-ROI boolean qualifying-vertex masks (91k) +
                               metadata; consumed by the tuning-profile script.
    group_vp_dominance_MmS.dscalar.nii   diverging M-S, NaN outside reliable cortex
    group_vp_shared_min.dscalar.nii      min(M,S) where both significant, else NaN
    group_vp_categorical.dscalar.nii     1=motor-dom, 2=scene-dom, 3=shared, 0=other
Also logs a ranked parcel table per category for reference.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.cifti2 import Cifti2Image, Cifti2Header
from nibabel.cifti2.cifti2_axes import ScalarAxis

from mario_encoding.config import PATHS
from mario_encoding import group_configs

# --------------------------------------------------------------------------
# Editable parameters
# --------------------------------------------------------------------------
GLASSER360 = Path(
    "/home/ozvar/Atlases/HCP_S1200_Atlas/"
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
)
DEFAULT_PM_THRESHOLD = 0.08   # substantial-encoding floor on the product measure
DEFAULT_DELTA = 0.02          # min (M-S)/(S-M) gap for a vertex to count as dominant

# ROI set. Each entry: name, category (shared|motor|scene), and the Glasser
# parcels by name (R_<p>_ROI + L_<p>_ROI, bilateral) or by R-hemisphere id list
# ("rids", auto-bilateralised with +180). Edit freely and rerun.
ROI_SPEC = [
    # --- shared (5) ---
    {"name": "VIP",        "category": "shared", "parcels": ["VIP"]},
    {"name": "IPS1",       "category": "shared", "parcels": ["IPS1"]},
    {"name": "7AL",        "category": "shared", "parcels": ["7AL"]},
    {"name": "7PC",        "category": "shared", "parcels": ["7PC"]},
    {"name": "SMA_preSMA", "category": "shared", "rids": [41, 57, 43, 44, 55, 96]},  # AMN-Action
    # --- motor (2) ---
    {"name": "M1_area4",   "category": "motor",  "parcels": ["4"]},
    {"name": "6d",         "category": "motor",  "parcels": ["6d"]},
    {"name": "1",          "category": "motor",  "parcels": ["1"]},
    # --- scene (3) --- chosen for scene-dominance in the tuning metric
    {"name": "MT",         "category": "scene",  "parcels": ["MT"]},
    {"name": "PCV",        "category": "scene",  "parcels": ["PCV"]},
    {"name": "TPOJ3",      "category": "scene",  "parcels": ["TPOJ3"]},
]
# --------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    return logging.getLogger(__name__)


def load_cifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32).squeeze()


def resolve_parcel_ids(spec_entry: dict, name_to_id: dict) -> list[int]:
    """Return bilateral Glasser label ids for an ROI spec entry."""
    if "rids" in spec_entry:
        return [i for i in spec_entry["rids"]] + [i + 180 for i in spec_entry["rids"]]
    ids = []
    for p in spec_entry["parcels"]:
        for hemi in ("R", "L"):
            key = f"{hemi}_{p}_ROI"
            if key not in name_to_id:
                raise ValueError(f"Parcel '{key}' (ROI '{spec_entry['name']}') not in atlas")
            ids.append(name_to_id[key])
    return ids


def write_map(data: np.ndarray, template_img, name: str, out_path: Path,
              logger: logging.Logger) -> None:
    axis1 = template_img.header.get_axis(1)
    header = Cifti2Header.from_axes((ScalarAxis([name]), axis1))
    img = Cifti2Image(data.astype(np.float32)[np.newaxis, :], header=header)
    nib.save(img, str(out_path))
    logger.info(f"  wrote {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Define VP ROIs from product-measure maps")
    parser.add_argument("--group", default="GROUP_N5_PRIMARY",
                        help="Group config name (for output_label / paths)")
    parser.add_argument("--pm-threshold", type=float, default=DEFAULT_PM_THRESHOLD)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--shared-pm-threshold", type=float, default=0.04,
                        help="separate, lower PM floor for the shared VIZ map only "
                             "(ROI gates still use --pm-threshold); widens the shared map")
    parser.add_argument("--ga-dir", type=str, default=None,
                        help="group_average/<id> dir with group_product_measure_*.dscalar.nii "
                             "(default: latest under results/variance_partitioning/group_average)")
    args = parser.parse_args()
    logger = setup_logging()

    group_cfg = getattr(group_configs, args.group)
    label = group_cfg["output_label"]
    cifti_dir = PATHS["figures"] / "ciftis" / label

    # Product-measure maps: latest group_average dir unless overridden.
    if args.ga_dir:
        ga = Path(args.ga_dir)
    else:
        cands = sorted((PATHS["variance_partitioning"] / "group_average").glob(
            "*/group_product_measure_motor.dscalar.nii"))
        if not cands:
            raise FileNotFoundError("No group_product_measure_motor.dscalar.nii under group_average/*")
        ga = cands[-1].parent
    logger.info(f"PM maps:      {ga}")
    logger.info(f"sig + output: {cifti_dir}")
    logger.info(f"PM_THRESHOLD={args.pm_threshold}  DELTA={args.delta}\n")

    M = load_cifti(ga / "group_product_measure_motor.dscalar.nii")
    S = load_cifti(ga / "group_product_measure_scene.dscalar.nii")
    msig = load_cifti(cifti_dir / "group_pm_motor_sig_conj_min4.dscalar.nii") > 0.5
    ssig = load_cifti(cifti_dir / "group_pm_scene_sig_conj_min4.dscalar.nii") > 0.5
    template_img = nib.load(str(cifti_dir / "group_pm_motor_sig_conj_min4.dscalar.nii"))

    gimg = nib.load(str(GLASSER360))
    lab = gimg.get_fdata().squeeze().astype(int)
    names = gimg.header.get_axis(0).label[0]
    name_to_id = {nm: k for k, (nm, _) in names.items() if k != 0}
    ncort = lab.shape[0]
    n = M.shape[0]

    thr, delta = args.pm_threshold, args.delta
    # Cortex-only gates (subcortex has no Glasser label); pad to full length.
    def cort(a):
        return a[:ncort]
    Mc, Sc, msigc, ssigc, labc = cort(M), cort(S), cort(msig), cort(ssig), lab

    qual = {
        "shared": msigc & ssigc & (Mc > thr) & (Sc > thr),
        "motor":  msigc & (Mc > thr) & ((Mc - Sc) > delta),
        "scene":  ssigc & (Sc > thr) & ((Sc - Mc) > delta),
    }
    score = {"shared": np.minimum(Mc, Sc), "motor": Mc - Sc, "scene": Sc - Mc}

    # --- ranked parcel table per category (reference) ---
    for cat in ("shared", "motor", "scene"):
        rows = []
        for lid in np.unique(labc):
            if lid == 0:
                continue
            p = labc == lid
            if p.sum() < 20:
                continue
            q = p & qual[cat]
            if q.sum() < 10:
                continue
            rows.append((names[lid][0], int(p.sum()), q.sum() / p.sum(),
                         score[cat][q].mean(), Mc[q].mean(), Sc[q].mean()))
        rows.sort(key=lambda x: x[3], reverse=True)
        logger.info(f"=== top parcels: {cat} ===")
        logger.info(f"{'parcel':<13}{'nv':>5}{'%qual':>7}{'score':>7}{'M':>7}{'S':>7}")
        for x in rows[:12]:
            logger.info(f"{x[0][:12]:<13}{x[1]:>5}{x[2]*100:>6.0f}%{x[3]:>7.3f}{x[4]:>7.3f}{x[5]:>7.3f}")
        logger.info("")

    # --- per-ROI qualifying masks (full 91k space) ---
    masks, meta_cat, meta_ids, meta_nqual = {}, {}, {}, {}
    for entry in ROI_SPEC:
        ids = resolve_parcel_ids(entry, name_to_id)
        parcel_mask_c = np.isin(labc, ids)
        roi_qual_c = parcel_mask_c & qual[entry["category"]]
        full = np.zeros(n, dtype=bool)
        full[:ncort] = roi_qual_c
        masks[entry["name"]] = full
        meta_cat[entry["name"]] = entry["category"]
        meta_ids[entry["name"]] = np.array(ids)
        meta_nqual[entry["name"]] = int(roi_qual_c.sum())
        logger.info(f"ROI {entry['name']:<12} ({entry['category']:<6}): "
                    f"{int(parcel_mask_c.sum()):>5} parcel vtx, {int(roi_qual_c.sum()):>5} qualifying")

    out_npz = cifti_dir / "roi_qualifying_masks.npz"
    np.savez(out_npz,
             roi_names=np.array(list(masks.keys())),
             categories=np.array([meta_cat[k] for k in masks]),
             pm_threshold=thr, delta=delta,
             **{f"mask_{k}": v for k, v in masks.items()},
             **{f"ids_{k}": meta_ids[k] for k in masks})
    logger.info(f"\nSaved ROI masks: {out_npz}")

    # --- visualization CIFTIs ---
    # Diverging overview (M-S) across all reliably-predicted cortex.
    dom_full = np.full(n, np.nan, dtype=np.float32)
    dom_full[:ncort] = np.where(cort(msig) | cort(ssig), Mc - Sc, np.nan)
    write_map(dom_full, template_img, "vp_dominance_MmS",
              cifti_dir / "group_vp_dominance_MmS.dscalar.nii", logger)

    # Per-band gated value maps: the band's actual value within its gate, NaN
    # elsewhere, so each loads alone with its own palette scaled threshold..max.
    # Shared map uses the lower --shared-pm-threshold (wider extent); motor/scene
    # use the same gate as the ROIs. Shared = min(M,S) is a conjunctive proxy
    # (both bands co-encode), NOT the reduced-model R2_shared.
    shared_thr = args.shared_pm_threshold
    shared_map_gate = msigc & ssigc & (Mc > shared_thr) & (Sc > shared_thr)
    band = {"motor": (Mc, qual["motor"]),
            "scene": (Sc, qual["scene"]),
            "shared": (np.minimum(Mc, Sc), shared_map_gate)}
    for cat, (vals, gate) in band.items():
        m = np.full(n, np.nan, dtype=np.float32)
        m[:ncort] = np.where(gate, vals, np.nan)
        write_map(m, template_img, f"vp_{cat}_gated",
                  cifti_dir / f"group_vp_{cat}_gated.dscalar.nii", logger)
        vv = vals[gate]
        logger.info(f"    {cat}: {int(gate.sum())} vtx, "
                    f"value range [{vv.min():.3f}, {vv.max():.3f}]")
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
