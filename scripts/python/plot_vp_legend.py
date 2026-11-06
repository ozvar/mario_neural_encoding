"""
Generate the dual-band VP legend PNG.

Produces a square heatmap with motor PM on the x-axis and scene PM on the
y-axis, using the same colormap as plot_dual_band_map.py composited over
the cortex background gray (#808080).

Usage:
    python plot_vp_legend.py --out PATH [--vmax 0.20] [--dpi 300]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "python"))
from plot_dual_band_map import make_cmap_array, DEFAULT_BLEND_SCALE, CMAP_N


BG_GRAY = 0x80 / 255   # #808080 -- matches CORTEX_BG_COLOR in plot_dual_band_map.R


def make_legend(out_path: Path, vmax: float = 0.20, dpi: int = 900) -> None:
    arr = make_cmap_array(
        CMAP_N, blend_scale=DEFAULT_BLEND_SCALE, opaque=True, bg_gray=BG_GRAY
    ).astype("float32") / 255.0
    img = arr[:, :, :3]

    ticks = np.arange(0, vmax + 1e-9, 0.05)

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.imshow(
        img, origin="lower", aspect="equal",
        extent=[0, vmax, 0, vmax], interpolation="bilinear",
    )
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect("equal", adjustable="box")

    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(labelsize=8)

    ax.set_xlabel("Action PM", fontsize=9)
    ax.set_ylabel("Perception PM", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    svg_path = out_path.with_suffix(".svg")
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig.savefig(str(svg_path), bbox_inches="tight")
    plt.close(fig)
    print(f"Legend saved: {out_path}")
    print(f"Legend saved: {svg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-band VP legend")
    parser.add_argument("--out",  type=Path, required=True, help="Output PNG path")
    parser.add_argument("--vmax", type=float, default=0.20)
    parser.add_argument("--dpi",  type=int,   default=300)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_legend(args.out, vmax=args.vmax, dpi=args.dpi)


if __name__ == "__main__":
    main()
