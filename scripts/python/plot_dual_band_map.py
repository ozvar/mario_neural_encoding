"""
Dual-band pycortex surface map: motor x scene product-measure.

Colormap: magnitude-normalized additive blend.
  Color encodes dominance direction; alpha encodes magnitude.
    (motor only)  -> action  #1f8bb3  (full saturation)
    (scene only)  -> percep  #ed8e48  (full saturation)
    (equal-high)  -> additive overflow -> near white
  Alpha = clip(max(motor, scene) * 4, 0, 1).

Gate: vertices outside (msig | ssig) and optional PM magnitude floor are zeroed.

Outputs into results/figures/pycortex/{group_label}/ (with --views all):
    dual_band_flatmap_lh.png         LH half of bilateral flatmap
    dual_band_flatmap_rh.png         RH half of bilateral flatmap
    dual_band_inflated_lh_lateral.png
    dual_band_inflated_lh_medial.png
    dual_band_inflated_rh_lateral.png
    dual_band_inflated_rh_medial.png
    dual_band_legend.png
"""
import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cortex
import cortex.export
from cortex.export.save_views import save_3d_views
from PIL import Image

from mario_encoding.config import PATHS
from mario_encoding import group_configs


ACTION_HEX   = '#1f8bb3'
PERCEP_HEX   = '#ed8e48'
CMAP_NAME         = 'BuOr_2D'
CMAP_NAME_OPAQUE  = 'BuOr_2D'
DEFAULT_VMAX      = 0.20
DEFAULT_BLEND_SCALE = 0.85   # max additive weight per channel; < 1 dims the shared region
DEFAULT_OPAQUE_BG = 0.5      # neutral gray under transparency for 3D (WebGL) views
CMAP_N            = 256

_INFLATED_ANGLES = ('lateral_pivot', 'medial_pivot')


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
    return logging.getLogger(__name__)


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip('#')[:6]
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)])


def make_cmap_array(n: int = CMAP_N, blend_scale: float = DEFAULT_BLEND_SCALE,
                    opaque: bool = False, bg_gray: float = DEFAULT_OPAQUE_BG) -> np.ndarray:
    """
    (n, n, 4) RGBA uint8, natural orientation:
      arr[row, col] = color for (scene_norm=row/n, motor_norm=col/n)

    Color: normalize motor and scene by max(motor, scene) so hue is always
    fully saturated regardless of absolute magnitude.
      rgb = clip(motor_n * action + scene_n * percep, 0, 1)
    At equal-high (motor_n=1, scene_n=1), additive overflow -> near white.

    Alpha = clip(max(motor, scene) * 4, 0, 1).
    If opaque=True, alpha is composited onto bg_gray and alpha set to 1
    (WebGL panels render the mesh itself; a transparent mesh becomes invisible,
    yielding blank screenshots that crash cortex.export.plot_panels).
    Flip vertically before saving (pycortex: PNG row 0 = scene=vmax2).
    """
    c_action = _hex_to_rgb(ACTION_HEX)
    c_percep = _hex_to_rgb(PERCEP_HEX)

    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    motor, scene = np.meshgrid(x, y)

    mag = np.maximum(motor, scene)
    safe = mag > 1e-6
    denom = np.where(safe, mag, 1.0)
    motor_n = np.where(safe, motor / denom, 0.0)
    scene_n = np.where(safe, scene / denom, 0.0)

    rgb = np.clip(blend_scale * (motor_n[..., None] * c_action + scene_n[..., None] * c_percep), 0, 1)
    alpha = np.clip(mag * 4, 0, 1)
    if opaque:
        rgb = rgb * alpha[..., None] + bg_gray * (1 - alpha[..., None])
        alpha = np.ones_like(alpha)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return (np.clip(rgba, 0, 1) * 255).astype(np.uint8)


def register_cmap(name: str = CMAP_NAME, n: int = CMAP_N,
                  blend_scale: float = DEFAULT_BLEND_SCALE,
                  opaque: bool = False, bg_gray: float = DEFAULT_OPAQUE_BG,
                  logger: logging.Logger = None) -> None:
    """Save colormap PNG to pycortex webgl colormaps dir (idempotent)."""
    cmap_dir = Path(cortex.options.config.get('webgl', 'colormaps'))
    cmap_dir.mkdir(parents=True, exist_ok=True)
    out = cmap_dir / f'{name}.png'
    arr = make_cmap_array(n, blend_scale=blend_scale, opaque=opaque, bg_gray=bg_gray)
    Image.fromarray(arr[::-1]).save(str(out))
    if logger:
        logger.info(f'Colormap: {out}')


def load_cifti_to_vertex_array(path: Path) -> np.ndarray:
    """91282-space CIFTI -> 64984 pycortex vertex array, NaN at medial wall."""
    cifti = nib.load(str(path))
    data = cifti.get_fdata()[0]
    brain_axis = cifti.header.get_axis(1)
    left_full = np.full(32492, np.nan)
    right_full = np.full(32492, np.nan)
    for name, data_indices, model in brain_axis.iter_structures():
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            left_full[list(model.vertex)] = data[data_indices]
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            right_full[list(model.vertex)] = data[data_indices]
    return np.concatenate([left_full, right_full])


def apply_gate(arr: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Zero vertices outside gate (-> transparent at colormap origin)."""
    return np.where(gate & np.isfinite(arr), arr, 0.0)


def make_legend(vmax_motor: float, vmax_scene: float, out_path: Path,
                cmap_name: str = None) -> None:
    """2D colorbar sourced from the stock pycortex cmap PNG. Square axes box;
    no in-heatmap text labels (add those externally e.g. in Inkscape)."""
    cmap_dir = Path(cortex.options.config.get('webgl', 'colormaps'))
    arr = np.array(Image.open(cmap_dir / f'{cmap_name}.png')).astype(np.float32) / 255.0
    if arr.shape[-1] == 4:
        fg = arr[..., :3]
        a = arr[..., 3:4]
        composite = fg * a + np.ones_like(fg) * (1 - a)
    else:
        composite = arr[..., :3]

    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(composite, extent=[0, vmax_motor, 0, vmax_scene],
              interpolation='bilinear')
    ax.set_xlabel('Action Product Measure', fontsize=9)
    ax.set_ylabel('Perception Product Measure', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _trim_image_borders(arr: np.ndarray) -> np.ndarray:
    """Trim fully-transparent (RGBA) or fully-white (RGB) borders."""
    if arr.ndim != 3:
        return arr
    if arr.shape[-1] == 4:
        mask = arr[..., 3] > 0
    else:
        mask = (arr < 255).any(axis=-1)
    if not mask.any():
        return arr
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def render_flatmap_split(vtx2d: cortex.Vertex2D, out_dir: Path, height: int,
                         curv_brightness: float = 0.4, curv_contrast: float = 0.3,
                         logger: logging.Logger = None) -> None:
    """Render bilateral flatmap, then save LH/RH halves as separate PNGs."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cortex.quickflat.make_png(
            tmp_path, vtx2d,
            with_curvature=True, with_colorbar=False,
            with_rois=False, with_sulci=True, with_labels=True, with_borders=True,
            linewidth=5, curvature_contrast=curv_contrast, curvature_brightness=curv_brightness,
            height=height,
        )
        img = np.array(Image.open(tmp_path))
    finally:
        os.unlink(tmp_path)

    half = img.shape[1] // 2
    lh, rh = img[:, :half], img[:, half:]
    for side_name, side_arr in (('lh', lh), ('rh', rh)):
        out_path = out_dir / f'dual_band_flatmap_{side_name}.png'
        Image.fromarray(_trim_image_borders(side_arr)).save(out_path)
        if logger:
            logger.info(f'Flatmap {side_name.upper()}:  {out_path}')


def render_inflated_split(vtx2d: cortex.Vertex2D, out_dir: Path, sleep: int = 20,
                          windowsize: tuple = (1920, 1080),
                          logger: logging.Logger = None) -> None:
    """Render inflated LH/RH x lateral/medial as 4 separate PNGs (one viewer session)."""
    temp_dir = tempfile.mkdtemp()
    base = os.path.join(temp_dir, 'fig')
    try:
        file_names = save_3d_views(
            vtx2d, base,
            list_angles=list(_INFLATED_ANGLES),
            list_surfaces=['inflated'] * len(_INFLATED_ANGLES),
            size=windowsize, sleep=sleep, trim=True,
            viewer_params=dict(labels_visible=[], overlays_visible=[]),
        )
        for fname, angle in zip(file_names, _INFLATED_ANGLES):
            img = np.array(Image.open(fname))
            half = img.shape[1] // 2
            left, right = img[:, :half], img[:, half:]
            if angle == 'medial_pivot':
                # medial_pivot view is mirror-flipped: left half is RH
                left, right = right, left
            angle_short = angle.split('_')[0]  # 'lateral' or 'medial'
            for side_name, side_arr in (('lh', left), ('rh', right)):
                out_path = out_dir / f'dual_band_inflated_{side_name}_{angle_short}.png'
                Image.fromarray(_trim_image_borders(side_arr)).save(out_path)
                if logger:
                    logger.info(f'Inflated {side_name.upper()} {angle_short}: {out_path}')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Dual-band pycortex surface map')
    parser.add_argument('--group', default='GROUP_N5_PRIMARY')
    parser.add_argument('--vmax', type=float, default=DEFAULT_VMAX,
                        help='Motor PM colormap ceiling (default 0.20)')
    parser.add_argument('--vmax-scene', type=float, default=DEFAULT_VMAX,
                        help='Scene PM colormap ceiling (default 0.20)')
    parser.add_argument('--blend-scale', type=float, default=DEFAULT_BLEND_SCALE,
                        help='Additive weight per channel [0-1]; lower = dimmer shared region (default 0.75)')
    parser.add_argument('--pm-threshold', type=float, default=0.0,
                        help='Optional PM magnitude floor added to significance gate')
    parser.add_argument('--curv-brightness', type=float, default=0.4,
                        help='Curvature background brightness 0=black 1=white (default 0.4)')
    parser.add_argument('--curv-contrast', type=float, default=0.3,
                        help='Curvature sulcal contrast (default 0.3)')
    parser.add_argument('--sleep', type=int, default=20,
                        help='Seconds to wait per panel before screenshotting (default 20)')
    parser.add_argument('--views', choices=['all', 'flatmap', 'inflated'], default='all',
                        help='all=flatmap (LH+RH) + inflated (LH/RH x lateral/medial), '
                             'flatmap=flatmap halves only, inflated=inflated panels only')
    parser.add_argument('--height', type=int, default=2048,
                        help='Flatmap height in pixels')
    args = parser.parse_args()
    logger = setup_logging()

    group_cfg = getattr(group_configs, args.group)
    label = group_cfg['output_label']
    cifti_dir = PATHS['figures'] / 'ciftis' / label
    out_dir = PATHS['pycortex'] / label
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f'Group:    {args.group}')
    logger.info(f'CIFTI:    {cifti_dir}')
    logger.info(f'Output:   {out_dir}')
    logger.info(f'vmax motor: {args.vmax}  vmax scene: {args.vmax_scene}  blend_scale: {args.blend_scale}  pm_threshold: {args.pm_threshold}')
    logger.info(f'views:    {args.views}\n')

    #register_cmap(blend_scale=args.blend_scale, logger=logger)
    #register_cmap(name=CMAP_NAME_OPAQUE, blend_scale=args.blend_scale,
    #              opaque=True, logger=logger)

    ga_cands = sorted((PATHS['variance_partitioning'] / 'group_average').glob(
        '*/group_product_measure_motor.dscalar.nii'))
    if not ga_cands:
        raise FileNotFoundError('No group_product_measure_motor.dscalar.nii found')
    ga = ga_cands[-1].parent
    logger.info(f'PM maps:  {ga}')

    motor_v = load_cifti_to_vertex_array(ga / 'group_product_measure_motor.dscalar.nii')
    scene_v = load_cifti_to_vertex_array(ga / 'group_product_measure_scene.dscalar.nii')

    msig_v = load_cifti_to_vertex_array(
        cifti_dir / 'group_pm_motor_sig_conj_min4.dscalar.nii') > 0.5
    ssig_v = load_cifti_to_vertex_array(
        cifti_dir / 'group_pm_scene_sig_conj_min4.dscalar.nii') > 0.5

    gate = (msig_v | ssig_v) & (np.fmax(motor_v, scene_v) > args.pm_threshold)
    motor_gated = apply_gate(motor_v, gate)
    scene_gated = apply_gate(scene_v, gate)
    logger.info(f'Gate:     {gate.sum()} / {np.isfinite(motor_v).sum()} cortical vertices\n')

    def _vtx(cmap_name: str) -> cortex.Vertex2D:
        return cortex.Vertex2D(
            scene_gated, motor_gated, 'hcp_32k',
            vmin=0, vmax=args.vmax_scene,
            vmin2=0, vmax2=args.vmax,
            cmap=cmap_name,
        )

    if args.views in ('all', 'flatmap'):
        render_flatmap_split(_vtx(CMAP_NAME), out_dir, args.height,
                             curv_brightness=args.curv_brightness,
                             curv_contrast=args.curv_contrast,
                             logger=logger)

    if args.views in ('all', 'inflated'):
        render_inflated_split(_vtx(CMAP_NAME_OPAQUE), out_dir,
                              sleep=args.sleep, logger=logger)

    legend_path = out_dir / 'dual_band_legend.png'
    make_legend(args.vmax, args.vmax_scene, legend_path, cmap_name=CMAP_NAME)
    logger.info(f'Legend:   {legend_path}')


if __name__ == '__main__':
    main()
