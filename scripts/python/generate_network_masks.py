"""Generate binary CIFTI network masks for network correspondence analysis.

Networks produced:
  - AMN (whole): Cole-Anticevic Cingulo-Opercular (label 4)
  - AMN-Decision: Glasser proxy (dACC + anterior insula + aPFC)
  - AMN-Action: Glasser proxy (posterior dACC + SMA/pre-SMA)
  - AMN-Feedback: Glasser proxy (middle insula + SMG + IFS/IFJ)
  - AMN-ParsMarginalis: Glasser proxy (area 23c + 31pv)
  - SCAN-M1-proxy: Glasser area 4 bilateral (whole primary motor cortex;
      inter-effector zones not isolatable at group level without individual PFM)
  - DAN: Cole-Anticevic Dorsal Attention (label 5)
  - MDN-proxy: Cole-Anticevic Frontoparietal (label 7; Fedorenko MDN blocked)

All outputs: binary dscalar.nii in fsLR_32k / 91282-grayordinate space.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ATLAS_DIR = Path("/home/ozvar/Atlases")
COLE_ANTICEVIC = (
    ATLAS_DIR
    / "HCP_S1200_Atlas"
    / "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_netassignments_LR.dlabel.nii"
)
GLASSER360 = (
    ATLAS_DIR
    / "HCP_S1200_Atlas"
    / "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
)
OUTPUT_DIR = ATLAS_DIR / "NetworkMasks"

# ---------------------------------------------------------------------------
# Cole-Anticevic network label IDs
# ---------------------------------------------------------------------------
CA_LABELS = {
    "CON": 4,   # Cingulo-Opercular = AMN
    "DAN": 5,   # Dorsal Attention
    "FPN": 7,   # Frontoparietal = MDN proxy
}

# ---------------------------------------------------------------------------
# Glasser 360 parcel label IDs
# R hemisphere: 1-180, L hemisphere: 181-360 (L_id = R_id + 180)
#
# AMN-Decision (Anterior): dACC + anterior insula + aPFC
#   - 24dd: dorsal anterior cingulate, dorsal division
#   - a24pr: anterior perigenual ACC
#   - a24: anterior area 24
#   - d32: dorsal area 32
#   - AVI: anterior ventral insula
#   - AAIC: anterior agranular insular cortex
#   - 10r: frontopolar area 10r
#   - 10pp: frontopolar area 10pp
#   - a10p: anterior frontopolar area 10p
#   - p32: posterior area 32
#
# AMN-Action (Central): posterior dACC + SMA/pre-SMA
#   - 24dv: dorsal anterior cingulate, ventral division
#   - p24pr: posterior perigenual ACC
#   - SCEF: supplementary cingulate eye field (cingulate motor area)
#   - 6ma: medial area 6 (pre-SMA)
#   - 6mp: medial premotor area
#   - 6a: SMA proper
#
# AMN-Feedback (Lateral): middle insula + SMG + IFS/IFJ
#   - MI: middle insula
#   - Ig: granular insula
#   - PoI1: posterior insula 1
#   - PoI2: posterior insula 2
#   - PF, PFm, PFop, PFt: supramarginal gyrus / inferior parietal
#   - IFSp, IFSa: inferior frontal sulcus
#   - IFJa, IFJp: inferior frontal junction
#
# AMN-ParsMarginalis: pars marginalis of cingulate sulcus
#   - 23c: area 23c (pars marginalis)
#   - 31pv: area 31pv (adjacent posterior cingulate)
#
# SCAN-M1-proxy: primary motor cortex (proxy only -- intra-M1 inter-effector
#   zones cannot be separated at group level without individual PFM)
#   - 4: primary motor cortex
# ---------------------------------------------------------------------------

_R_GLASSER = {
    "AMN-Decision": [40, 59, 61, 62, 111, 112, 65, 90, 89, 64],
    "AMN-Action":   [41, 57, 43, 44, 55, 96],
    "AMN-Feedback": [109, 168, 167, 106, 148, 149, 147, 116, 81, 82, 79, 80],
    "AMN-ParsMarginalis": [38, 35],
    "SCAN-M1-proxy": [8],
}

def _bilateral(r_ids: Sequence[int]) -> list[int]:
    """Return bilateral label IDs (R hemisphere ids + L hemisphere ids)."""
    return list(r_ids) + [i + 180 for i in r_ids]

GLASSER_LABELS: dict[str, list[int]] = {
    name: _bilateral(ids) for name, ids in _R_GLASSER.items()
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cifti_data(path: Path) -> tuple[np.ndarray, nib.cifti2.BrainModelAxis]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32).squeeze()
    brain_model_axis = img.header.get_axis(1)
    return data, brain_model_axis


def save_binary_mask(
    mask: np.ndarray,
    brain_model_axis: nib.cifti2.BrainModelAxis,
    name: str,
    output_path: Path,
) -> None:
    scalar_axis = nib.cifti2.ScalarAxis([name])
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    data_2d = mask.astype(np.float32)[np.newaxis, :]
    img = nib.Cifti2Image(data_2d, header=header)
    nib.save(img, str(output_path))
    n = int(mask.sum())
    log.info("  Saved %s (%d grayordinates) -> %s", name, n, output_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading Cole-Anticevic atlas (%s)", COLE_ANTICEVIC.name)
    ca_data, brain_model_axis = load_cifti_data(COLE_ANTICEVIC)
    n_grayordinates = ca_data.shape[0]
    log.info("  Grayordinates: %d", n_grayordinates)

    log.info("Loading Glasser 360 atlas (%s)", GLASSER360.name)
    g_data, _ = load_cifti_data(GLASSER360)

    # ---- Cole-Anticevic whole-network masks --------------------------------
    log.info("\n--- Cole-Anticevic whole-network masks ---")
    ca_specs = {
        "AMN_CON_cole-anticevic": ("CON", "AMN-whole (Cingulo-Opercular)"),
        "DAN_cole-anticevic":     ("DAN", "Dorsal Attention Network"),
        "MDN-proxy_FPN_cole-anticevic": ("FPN", "MDN proxy (Frontoparietal)"),
    }
    for fname, (net_key, label) in ca_specs.items():
        mask = (ca_data == CA_LABELS[net_key])
        outpath = OUTPUT_DIR / f"{fname}.dscalar.nii"
        save_binary_mask(mask, brain_model_axis, label, outpath)

    # ---- Glasser-based AMN subnetwork and SCAN masks -----------------------
    # Glasser covers only cortical grayordinates (indices 0:59412).
    # Embed into the full 91282-length array with zeros for subcortex.
    n_cortex = g_data.shape[0]  # 59412
    assert n_cortex == 59412, f"Unexpected Glasser grayordinate count: {n_cortex}"
    log.info("\n--- Glasser-based proxy masks ---")
    glasser_specs = {
        "AMN-Decision_glasser-proxy":       "AMN-Decision (dACC+ant.ins+aPFC)",
        "AMN-Action_glasser-proxy":         "AMN-Action (posterior dACC+SMA)",
        "AMN-Feedback_glasser-proxy":       "AMN-Feedback (mid.ins+SMG+IFS)",
        "AMN-ParsMarginalis_glasser-proxy": "AMN-ParsMarginalis (area 23c+31pv)",
        "SCAN-M1-proxy_glasser":            "SCAN-M1-proxy (whole M1, area 4)",
    }
    for fname, label in glasser_specs.items():
        network_key = fname.split("_glasser")[0]
        ids = set(GLASSER_LABELS[network_key])
        cortex_mask = np.isin(g_data, list(ids))
        full_mask = np.zeros(n_grayordinates, dtype=bool)
        full_mask[:n_cortex] = cortex_mask
        outpath = OUTPUT_DIR / f"{fname}.dscalar.nii"
        save_binary_mask(full_mask, brain_model_axis, label, outpath)

    # ---- Summary -----------------------------------------------------------
    log.info("\n--- Summary ---")
    masks = sorted(OUTPUT_DIR.glob("*.dscalar.nii"))
    log.info("Masks written to %s:", OUTPUT_DIR)
    for m in masks:
        img = nib.load(str(m))
        d = img.get_fdata(dtype=np.float32).squeeze()
        log.info("  %-55s  %6d grayordinates", m.name, int(d.sum()))


if __name__ == "__main__":
    main()
