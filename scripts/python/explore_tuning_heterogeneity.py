"""
Exploratory: within-scene tuning heterogeneity across anatomically diverse parcels.

Unlike plot_tuning_profiles.py (which uses the dominance-gated ROI masks), this
includes every FDR-significant vertex within each parcel, so weakly-predicted but
interesting regions (parahippocampal, ventral) are not gated out. Draws scene-only
rose plots two ways so the pattern can be judged independent of scaling:
  scene_raw/   common radius (global scene max) -- shows magnitude + shape
  scene_norm/  per-parcel scene-max -- pure shape comparison
Also prints/saves the top-scene-feature table. Existing outputs are untouched.

Edit EXPLORE_PARCELS and rerun.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib

from mario_encoding.config import PATHS, FEATURE_SPACES
from mario_encoding import group_configs

GLASSER360 = Path("/home/ozvar/Atlases/HCP_S1200_Atlas/"
                  "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii")
# anatomically diverse probe set (bilateral); edit freely
EXPLORE_PARCELS = ["V1", "MT", "FFC", "PHA1", "MIP", "PFt", "4", "6d", "VIP", "TPOJ2", "TPOJ3", "PCV", "7PL", "7AM", "46", "9-46d", "FEF", "LIPv", "STV"]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    return logging.getLogger(__name__)


def vp_dir(sc):
    s, tr, te = sc["subject"], sc["train_sessions"], sc["test_sessions"]
    return PATHS["variance_partitioning"] / (
        f"sub-{s:02d}_train-ses-{min(tr):03d}-{max(tr):03d}"
        f"_test-ses-{min(te):03d}-{max(te):03d}") / sc["experiment_id"]


def draw_scene_rose(scene_vals, scene_colors, ylim, title, out):
    n = len(scene_vals)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.95
    fig = plt.figure(figsize=(3.0, 3.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(angles, np.nan_to_num(scene_vals), width=width, color=scene_colors,
           align="edge", edgecolor="none")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_ylim(0, ylim)
    ax.set_title(title, fontsize=11, pad=8)
    fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="GROUP_N5_PRIMARY")
    args = ap.parse_args()
    logger = setup_logging()

    cfg = getattr(group_configs, args.group)
    label = cfg["output_label"]
    fig = PATHS["figures"] / "ciftis" / label
    sig = nib.load(str(fig / "group_r_full_sig_conj_min4.dscalar.nii")).get_fdata(dtype=np.float32).squeeze() > 0.5
    g = nib.load(str(GLASSER360)); lab = g.get_fdata().squeeze().astype(int)
    names = g.header.get_axis(0).label[0]
    rev = {nm: k for k, (nm, _) in names.items() if k}
    nc = lab.shape[0]; N = sig.shape[0]
    lab91 = np.zeros(N, int); lab91[:nc] = lab; sig = sig[:N]

    subs, scene_sets = [], []
    for sc in cfg["subjects"]:
        d = vp_dir(sc)
        if not (d / "feature_sds.npz").exists():
            logger.warning(f"sub-{sc['subject']:02d}: feature_sds.npz MISSING -- skipping"); continue
        w = np.load(d / "weights_full.npz"); sd = np.load(d / "feature_sds.npz", allow_pickle=True)
        valid = np.load(d / "valid_voxels_mask.npy")
        ws = np.maximum(w["scene_weights_avg"] * sd["scene"][:, None], 0.0)
        subs.append((ws, [str(x) for x in sd["scene_features"]], valid))
        scene_sets.append(set(str(x) for x in sd["scene_features"]))
    common = [f for f in FEATURE_SPACES["scene"] if all(f in s for s in scene_sets)]
    logger.info(f"{len(subs)} subjects, {len(common)} common scene features")

    profiles = {}
    for pname in EXPLORE_PARCELS:
        ids = [rev.get(f"{h}_{pname}_ROI") for h in ("L", "R")]
        ids = [i for i in ids if i is not None]
        if not ids:
            logger.warning(f"{pname}: not in atlas, skip"); continue
        pmask = np.isin(lab91, ids) & sig
        per = []
        for ws, snames, valid in subs:
            cols = pmask[valid]
            if cols.sum() < 5:
                continue
            sidx = [snames.index(f) for f in common]
            per.append(ws[:, cols][sidx].mean(1))
        if len(per) < 3:
            logger.warning(f"{pname}: <3 subjects with vertices, skip"); continue
        profiles[pname] = np.nanmean(np.vstack(per), axis=0)

    scene_colors = list(plt.cm.hsv(np.linspace(0.0, 0.95, len(common))))
    out = PATHS["figures"] / "tuning" / "explore"
    (out / "scene_raw").mkdir(parents=True, exist_ok=True)
    (out / "scene_norm").mkdir(parents=True, exist_ok=True)
    gmax = max(np.nanmax(v) for v in profiles.values())

    logger.info(f"\n{'parcel':<8} top-3 scene features")
    for pname, prof in profiles.items():
        top = np.argsort(prof)[::-1][:3]
        logger.info(f"{pname:<8} " + ", ".join(common[i] for i in top))
        draw_scene_rose(prof, scene_colors, gmax, f"{pname} (raw)", out / "scene_raw" / f"{pname}.png")
        norm = prof / np.nanmax(prof) if np.nanmax(prof) > 0 else prof
        draw_scene_rose(norm, scene_colors, 1.0, f"{pname} (norm)", out / "scene_norm" / f"{pname}.png")

    np.savez(out / "explore_profiles.npz",
             parcels=np.array(list(profiles.keys())),
             scene_features=np.array(common),
             **{f"profile_{p}": v for p, v in profiles.items()})
    logger.info(f"\nSaved scene roses (raw + norm) + explore_profiles.npz to {out}")


if __name__ == "__main__":
    main()
