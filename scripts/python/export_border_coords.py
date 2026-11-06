"""
Export Cole-Anticevic network border vertex indices and barycentric weights.

For each target network and hemisphere, saves three .npy files:
  <net>_<H>_verts.npy   int32  (N, 3)  vertex index triplets
  <net>_<H>_weights.npy float32 (N, 3) barycentric weights
  <net>_<H>_breaks.npy  int32  (K,)    row indices where each border part starts

Border positions are intentionally NOT pre-computed here.  The R render
script evaluates positions from the already-transformed surface vertex array,
guaranteeing exact alignment with the mesh regardless of the rotation applied.

Usage:
    python export_border_coords.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ATLAS_DIR = Path("/home/ozvar/Atlases/HCP_S1200_Atlas")
BORDER_FILES = {
    "L": ATLAS_DIR / "ColeAnticevic_all-networks.L.border",
    "R": ATLAS_DIR / "ColeAnticevic_all-networks.R.border",
}

REPO_DIR = Path(__file__).resolve().parents[2]
OUT_DIR  = REPO_DIR / "results" / "r_cifti_arrays" / "borders"

NETWORKS = ("Visual2", "Cingulo-Opercular")


def parse_border(border_file: Path, network: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (verts, weights, breaks) for all BorderParts of a network.

    verts:   int32  (N, 3)  vertex index triplets
    weights: float32 (N, 3) barycentric weights
    breaks:  int32  (K,)    row index of the first point in each part
    """
    tree = ET.parse(str(border_file))
    all_verts, all_weights, breaks = [], [], []
    for b in tree.getroot().findall(".//Border"):
        if b.get("Name") != network:
            continue
        for part in b.findall("BorderPart"):
            closed = part.get("Closed", "False").lower() == "true"
            vert_lines   = (part.find("Vertices").text or "").strip().splitlines()
            weight_lines = (part.find("Weights").text  or "").strip().splitlines()

            pverts, pweights = [], []
            for vl, wl in zip(vert_lines, weight_lines):
                vv = list(map(int,   vl.split()))
                ww = list(map(float, wl.split()))
                if len(vv) == 1:
                    pverts.append([vv[0], vv[0], vv[0]])
                    pweights.append([1.0, 0.0, 0.0])
                elif len(vv) == 3:
                    pverts.append(vv)
                    pweights.append(ww)

            if not pverts:
                continue
            if closed:
                pverts.append(pverts[0])
                pweights.append(pweights[0])

            breaks.append(len(all_verts))
            all_verts.extend(pverts)
            all_weights.extend(pweights)

    verts   = np.array(all_verts,   dtype=np.int32)
    weights = np.array(all_weights, dtype=np.float32)
    breaks  = np.array(breaks,      dtype=np.int32)
    return verts, weights, breaks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for network in NETWORKS:
        net_slug = network.lower().replace(" ", "_").replace("-", "_")
        for hemi, bf in BORDER_FILES.items():
            verts, weights, breaks = parse_border(bf, network)
            print(f"{hemi} {network}: {len(breaks)} parts, {len(verts)} points")
            np.save(OUT_DIR / f"{net_slug}_{hemi}_verts.npy",   verts)
            np.save(OUT_DIR / f"{net_slug}_{hemi}_weights.npy", weights)
            np.save(OUT_DIR / f"{net_slug}_{hemi}_breaks.npy",  breaks)
            print(f"  -> {net_slug}_{hemi}_{{verts,weights,breaks}}.npy")

    print(f"\nDone. Wrote to {OUT_DIR}")


if __name__ == "__main__":
    main()
