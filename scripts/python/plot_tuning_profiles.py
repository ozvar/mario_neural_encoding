"""
ROI feature-tuning rose plots from the variance-partitioning encoding weights.

For each ROI in roi_qualifying_masks.npz (define_vp_rois.py), build a standardized
tuning profile and draw it as a polar bar (rose) plot, one figure per ROI, with a
distinct colour per feature (motor: self-similar blues; scene: varied palette).

Per subject:
  W_std = weights_avg * feature_SD   (standardized coefficient, per band, by name)
  W_pos = max(W_std, 0)
  ROI profile = mean of W_pos over the ROI's qualifying vertices that are also
                valid voxels for that subject.
Across subjects: motor features are identical (6); scene features are intersected
by name; profiles are averaged.

Outputs (results/figures/tuning/):
  tuning_profiles.npz        per-ROI group-mean profiles + names + categories
  roses/<ROI>.png            one transparent rose per ROI (compose externally)
  feature_legend.png         colour -> feature-name key (motor + scene)
"""
import argparse
import logging
from pathlib import Path

import colorsys

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from mario_encoding.config import PATHS, FEATURE_SPACES
from mario_encoding import group_configs
from mario_encoding.utils.viz_utils import save_figure

GLASSER360 = Path("/home/ozvar/Atlases/HCP_S1200_Atlas/"
                  "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii")

# Scene features grouped into semantic bands; spokes are coloured by group and
# each group's tallest spoke is labelled (if large enough). Edit freely.
SCENE_GROUPS = {
    "Enemies":    ["Enemy", "2-Horde", "3-Horde", "4-Horde", "Gap enemy", "Enemy valley",
                   "Enemy stair valley", "enemy_drawn15", "enemy_drawn16", "enemy_drawn17",
                   "enemy_drawn18", "enemy_drawn19"],
    "Gaps":       ["Gap", "Multiple gaps", "Variable gaps", "Pillar gap", "Gap stair valley"],
    "Terrain":    ["Valley", "Pipe valley", "Empty valley", "Roof valley", "Empty stair valley",
                   "Roof", "Stair up", "Stair down"],
    "Navigation": ["2-Path", "3-Path", "Moving platform"],
    "Reward":     ["Risk/Reward", "Reward"],
    "Powerup":    ["mario_big", "mario_fire", "mario_star", "powerup_visible"],
    "Landmarks":  ["Flagpole", "Beginning", "Bonus zone", "Waterworld", "Checkpoint"],
}
GROUP_ORDER = ["motor", "Enemies", "Gaps", "Terrain", "Navigation", "Reward", "Powerup", "Landmarks"]
GROUP_COLORS = {
    "motor": "#1f3b73", "Enemies": "#e41a1c", "Gaps": "#ff7f00", "Terrain": "#a65628",
    "Navigation": "#4daf4a", "Reward": "#984ea3", "Powerup": "#f781bf", "Landmarks": "#377eb8",
}

ACTION_HEX = "#258cb0"
SCENE_HEX  = "#ec8e48"


def feature_layout(motor_names, common_scene):
    """Order features motor-first then scene-by-group; return (ordered_names, group_of, perm).

    perm maps positions in the original [motor_names + common_scene] profile vector
    onto the grouped display order.
    """
    feat_to_group = {f: g for g, fs in SCENE_GROUPS.items() for f in fs}
    orig = list(motor_names) + list(common_scene)
    ordered, group_of = [], []
    for f in motor_names:
        ordered.append(f); group_of.append("motor")
    for g in GROUP_ORDER[1:]:
        for f in SCENE_GROUPS[g]:
            if f in common_scene:
                ordered.append(f); group_of.append(g)
    # any scene feature not covered by a group (safety) -> append as 'Other'
    for f in common_scene:
        if f not in feat_to_group:
            ordered.append(f); group_of.append("Landmarks")
    perm = [orig.index(f) for f in ordered]
    return ordered, group_of, perm


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    return logging.getLogger(__name__)


def vp_dir(subj_cfg: dict) -> Path:
    s, tr, te = subj_cfg["subject"], subj_cfg["train_sessions"], subj_cfg["test_sessions"]
    name = (f"sub-{s:02d}_train-ses-{min(tr):03d}-{max(tr):03d}"
            f"_test-ses-{min(te):03d}-{max(te):03d}")
    return PATHS["variance_partitioning"] / name / subj_cfg["experiment_id"]


def subject_wpos(d: Path):
    w = np.load(d / "weights_full.npz")
    sd = np.load(d / "feature_sds.npz", allow_pickle=True)
    valid = np.load(d / "valid_voxels_mask.npy")
    mw, sw = w["motor_weights_avg"], w["scene_weights_avg"]
    msd, ssd = sd["motor"], sd["scene"]
    mnames = [str(x) for x in sd["motor_features"]]
    snames = [str(x) for x in sd["scene_features"]]
    if mw.shape[0] != msd.shape[0] or sw.shape[0] != ssd.shape[0]:
        raise ValueError(f"{d}: weight rows != SD lengths")
    return (np.maximum(mw * msd[:, None], 0.0), mnames,
            np.maximum(sw * ssd[:, None], 0.0), snames, valid)


def roi_profile(wpos: np.ndarray, valid: np.ndarray, roi_mask: np.ndarray):
    cols = roi_mask[valid]
    if cols.sum() == 0:
        return np.full(wpos.shape[0], np.nan)
    return wpos[:, cols].mean(axis=1)


def parcel_masks(parcels, sig_path):
    """Build 91k FDR-significant masks for manually-listed Glasser parcels (bilateral)."""
    sig = nib.load(str(sig_path)).get_fdata(dtype=np.float32).squeeze() > 0.5
    g = nib.load(str(GLASSER360)); lab = g.get_fdata().squeeze().astype(int)
    rev = {nm: k for k, (nm, _) in g.header.get_axis(0).label[0].items() if k}
    nc, N = lab.shape[0], sig.shape[0]
    lab91 = np.zeros(N, int); lab91[:nc] = lab; sig = sig[:N]
    masks, missing = {}, []
    for p in parcels:
        ids = [rev.get(f"{h}_{p}_ROI") for h in ("L", "R")]
        ids = [i for i in ids if i is not None]
        if not ids:
            missing.append(p); continue
        masks[p] = np.isin(lab91, ids) & sig
    return masks, missing


def prettify(name: str) -> str:
    """Display label: underscores -> spaces, uppercase first character only."""
    s = str(name).replace("_", " ")
    return s[:1].upper() + s[1:] if s else s


def _sequential_colors(hex_center: str, n: int) -> list:
    """Sample n unique colors from a sequential cmap centered on hex_center."""
    r, g, b = to_rgb(hex_center)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    light = colorsys.hls_to_rgb(h, min(l + 0.45, 0.92), s * 0.55)
    dark  = colorsys.hls_to_rgb(h, max(l - 0.28, 0.10), s)
    cmap  = LinearSegmentedColormap.from_list("_", [light, (r, g, b), dark])
    return [cmap(x) for x in np.linspace(0.0, 1.0, n)]


def feature_colors(group_of):
    """Per-feature colours: action sequential cmap around ACTION_HEX, scene around SCENE_HEX."""
    n_motor = group_of.count("motor")
    n_scene = len(group_of) - n_motor
    return _sequential_colors(ACTION_HEX, n_motor) + _sequential_colors(SCENE_HEX, n_scene)


def draw_rose(values, group_of, colors, feat_names, normalize, label_frac, title, out_dir: Path,
              name: str, ylim_abs=None):
    """Polar bar with one colour per feature; the tallest spoke within each group is
    labelled with its feature name (in that spoke's colour) iff value >= label_frac*max.
    ylim_abs sets a common radius across ROIs (for raw/no-normalize cross-ROI comparison)."""
    vals = np.array(values, dtype=float)
    if normalize and np.nanmax(vals) > 0:
        vals = vals / np.nanmax(vals)
    n = len(vals)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.95
    fig = plt.figure(figsize=(3.4, 3.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.bar(angles, np.nan_to_num(vals), width=width, color=colors,
           align="edge", edgecolor="none")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    mx = np.nanmax(vals) if np.nanmax(vals) > 0 else 1.0
    ax.set_ylim(0, (ylim_abs if ylim_abs is not None else mx) * 1.3)
    for g in GROUP_ORDER:
        idxs = [i for i, gg in enumerate(group_of) if gg == g]
        if not idxs:
            continue
        ti = idxs[int(np.nanargmax([vals[i] for i in idxs]))]
        if vals[ti] >= label_frac * mx and vals[ti] > 0:
            ax.text(angles[ti] + width / 2, vals[ti] + 0.08 * mx, prettify(feat_names[ti]),
                    color=colors[ti], fontsize=7, ha="center", va="center")
    # ax.set_title(title, fontsize=11, pad=8)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
    save_figure(fig, out_dir, f"{name}.png", dpi=200, pad_inches=0.02)


def draw_legend(feat_names, colors, out_dir: Path, name: str = "feature_legend"):
    fig, ax = plt.subplots(figsize=(5, len(feat_names) * 0.16 + 0.6))
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, [prettify(f) for f in feat_names], loc="center", ncol=2,
              fontsize=7, frameon=False)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    save_figure(fig, out_dir, f"{name}.png", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser(description="ROI feature-tuning rose plots (one per ROI)")
    ap.add_argument("--group", default="GROUP_N5_PRIMARY")
    ap.add_argument("--no-normalize", action="store_true",
                    help="plot raw standardized weights instead of per-ROI max-normalised")
    ap.add_argument("--parcels", nargs="+", default=None,
                    help="manual Glasser parcel names (bilateral); uses FDR-significant "
                         "vertices instead of the gated ROI masks, full motor+scene roses")
    ap.add_argument("--label-frac", type=float, default=0.35,
                    help="label a group's top spoke only if >= this fraction of the max spoke")
    ap.add_argument("--subdir", default=None,
                    help="output subfolder name under results/figures/tuning/ (overrides default)")
    args = ap.parse_args()
    logger = setup_logging()
    normalize = not args.no_normalize

    cfg = getattr(group_configs, args.group)
    label = cfg["output_label"]
    if args.parcels:
        sig_path = PATHS["figures"] / "ciftis" / label / "group_r_full_sig_conj_min4.dscalar.nii"
        roi_masks, missing = parcel_masks(args.parcels, sig_path)
        for m in missing:
            logger.warning(f"parcel '{m}' not in atlas (check casing, e.g. 7Am not 7AM) -- skip")
        roi_names = list(roi_masks.keys())
        categories = {r: "manual" for r in roi_names}
        roses_sub, npz_name = "roses_manual", "tuning_profiles_manual.npz"
    else:
        masks = np.load(PATHS["figures"] / "ciftis" / label / "roi_qualifying_masks.npz",
                        allow_pickle=True)
        roi_names = [str(x) for x in masks["roi_names"]]
        categories = {str(n): str(c) for n, c in zip(masks["roi_names"], masks["categories"])}
        roi_masks = {r: masks[f"mask_{r}"] for r in roi_names}
        roses_sub, npz_name = "roses", "tuning_profiles.npz"
    if args.subdir:
        roses_sub = args.subdir

    subj_data, scene_sets = [], []
    for sc in cfg["subjects"]:
        d = vp_dir(sc)
        if not (d / "feature_sds.npz").exists():
            logger.warning(f"sub-{sc['subject']:02d}: feature_sds.npz MISSING -- skipping")
            continue
        wm, mnames, ws, snames, valid = subject_wpos(d)
        subj_data.append((sc["subject"], wm, mnames, ws, snames, valid))
        scene_sets.append(set(snames))
    if not subj_data:
        raise SystemExit("No subjects with feature_sds.npz found.")
    logger.info(f"Using {len(subj_data)} subjects: {[s[0] for s in subj_data]}")

    common_scene = [f for f in FEATURE_SPACES["scene"] if all(f in s for s in scene_sets)]
    motor_names = subj_data[0][2]
    feat_names = motor_names + common_scene
    logger.info(f"features: motor {len(motor_names)}, scene {len(common_scene)}")

    profiles = {}
    for roi in roi_names:
        rmask = roi_masks[roi]
        per_subj = []
        for subj, wm, mnames, ws, snames, valid in subj_data:
            pm = roi_profile(wm, valid, rmask)
            s_idx = [snames.index(f) for f in common_scene]
            ps = roi_profile(ws, valid, rmask)[s_idx]
            per_subj.append(np.concatenate([pm, ps]))
        profiles[roi] = np.nanmean(np.vstack(per_subj), axis=0)

    # grouped display order (motor first, then scene by semantic group)
    ordered_names, group_of, perm = feature_layout(motor_names, common_scene)

    out_dir = PATHS["figures"] / "tuning"
    (out_dir / roses_sub).mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / npz_name,
             roi_names=np.array(roi_names),
             categories=np.array([categories[r] for r in roi_names]),
             feature_names=np.array(feat_names),
             ordered_feature_names=np.array(ordered_names),
             group_of=np.array(group_of),
             n_motor=len(motor_names),
             **{f"profile_{r}": profiles[r] for r in roi_names})

    colors = feature_colors(group_of)
    draw_legend(ordered_names, colors, out_dir)
    # common radius across ROIs when not normalizing, so raw magnitudes are comparable
    ymax = None if normalize else max(np.nanmax(profiles[r][perm]) for r in roi_names)
    for roi in roi_names:
        prof_ordered = profiles[roi][perm]
        draw_rose(prof_ordered, group_of, colors, ordered_names, normalize, args.label_frac,
                  f"{roi} ({categories[roi]})", out_dir / roses_sub, roi, ylim_abs=ymax)
    logger.info(f"Saved {len(roi_names)} roses + legend to {out_dir / roses_sub}")


if __name__ == "__main__":
    main()
