"""
Plot product-measure variance decomposition panels with explicit shared/unique
categorization.

For each tau in {0.6, 0.7, 0.8, 0.9}, voxels passing
    sig_full (>= 4/5 subjects individually FDR q<0.05) AND R2_full >= floor_r2
are classified by the larger band's share of R2_full = clip(pm_motor,0) + clip(pm_scene,0):

    motor-unique : pm_motor / R2_full >= tau
    scene-unique : pm_scene / R2_full >= tau
    shared       : otherwise (both bands carry non-trivial fraction)

Per-voxel category values used for the figures:
    shared       -> R2_full
    motor-unique -> pm_motor
    scene-unique -> pm_scene

Five SVG files per tau under results/figures/vp_decomposition/{output_label}/:
    pie_variance_tau-{tau}.svg
    pie_count_tau-{tau}.svg
    histogram_full_vs_shared_tau-{tau}.svg
    ridges_shared_tau-{tau}.svg
    ridges_unique_overlapping_tau-{tau}.svg

Inputs:
    results/variance_partitioning/group_average/{output_label}/group_significance.npz
    PATHS['ColeAnticevic_networks']

Usage:
    python scripts/python/plot_vp_decomposition_panels.py --group GROUP_N5_PRIMARY
"""
import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.patches as mpatches

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from nibabel.cifti2 import Cifti2Header, Cifti2Image
from nibabel.cifti2.cifti2_axes import ScalarAxis
from scipy import stats

from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY
from mario_encoding.utils import sns_styleset, add_clean_end_ticks


def save_figure(fig, fig_dir: Path, file_name: str, dpi: int = 300) -> None:
    """Save figure as both SVG and PNG (filename's extension is replaced for each)."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file_name).stem
    for ext in ('svg', 'png'):
        fig.savefig(fig_dir / f'{stem}.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}

MOTOR_COLOR = '#188bb7'   # blue
SCENE_COLOR = '#ec8e48'   # orange
SHARED_COLOR = '#aa52c7'  # purple
FULL_COLOR = '#8c8c8c'    # gray

TAUS = [0.6, 0.7, 0.8, 0.9]
FLOOR_R2 = 0.05
MIN_N_FOR_RIDGE = 30   # below this, suppress ridge AND label for that network


# ============================================================================
# Network parcellation
# ============================================================================

def get_cole_anticevic_config() -> Dict:
    return {
        'labels': [
            'Primary Visual', 'Visual2', 'Somatomotor', 'Auditory',
            'Cingulo-Opercular', 'Dorsal Attention', 'Language',
            'Frontoparietal', 'Ventral Multimodal', 'Posterior Multimodal',
            'Orbito-Affective', 'Default',
        ],
        'colors': [
            '#0000FF', '#6400FF', '#00FFFF', '#FA3EFB',
            '#990099', '#00FF00', '#009B9B',
            '#FFFF00', '#FF9D00', '#B15928',
            '#417D00', '#FF0000',
        ],
    }


def load_network_parcellation(parcellation_path: Path) -> np.ndarray:
    cifti = nib.load(str(parcellation_path))
    return cifti.get_fdata().squeeze().astype(int)


# ============================================================================
# Data loading and categorization
# ============================================================================

def load_group_inputs(group_label: str, fdr_alpha: float) -> Dict:
    group_cfg = GROUP_REGISTRY[group_label]
    sig_dir = PATHS['variance_partitioning'] / 'group_average' / group_cfg['output_label']
    sig_path = sig_dir / 'group_significance.npz'
    if not sig_path.exists():
        raise FileNotFoundError(
            f"group_significance.npz not found at {sig_path}. "
            f"Run compute_group_significance.py --group {group_label} first."
        )

    data = np.load(sig_path)
    pm_motor = data['pm_motor_mean_observed'].astype(np.float64)
    pm_scene = data['pm_scene_mean_observed'].astype(np.float64)

    pm_motor_clip = np.clip(pm_motor, 0, None)
    pm_scene_clip = np.clip(pm_scene, 0, None)
    r2_full = pm_motor_clip + pm_scene_clip  # clipped R2_full proxy

    return {
        'group_label': group_label,
        'output_label': group_cfg['output_label'],
        'pm_motor': pm_motor_clip,
        'pm_scene': pm_scene_clip,
        'r2_full': r2_full,
        'sig_full': data['r_full_sig_conj_min4'].astype(bool),
        'fdr_alpha': fdr_alpha,
    }


def _find_template_cifti() -> Path:
    """Locate any local 91k .dscalar.nii to reuse its brain model axis as a header template."""
    search_roots = [
        PATHS['figures'],
        PATHS['variance_partitioning'] / 'group_average',
        PATHS['variance_partitioning'],
    ]
    for root in search_roots:
        if not Path(root).exists():
            continue
        hits = sorted(Path(root).rglob('*.dscalar.nii'))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "No local .dscalar.nii found to use as CIFTI template. "
        "Run create_vp_model_cifti.py once, or pass an explicit template."
    )


def _write_cifti_scalar(data: np.ndarray, template_path: Path,
                        output_path: Path, map_name: str) -> None:
    """Write a (91282,) float32 array as a CIFTI dscalar using template's brain model axis."""
    template = nib.load(str(template_path))
    cifti_data = data.astype(np.float32)[np.newaxis, :]
    brain_model_axis = template.header.get_axis(1)
    scalar_axis = ScalarAxis([map_name])
    header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    img = Cifti2Image(cifti_data, header=header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.to_filename(str(output_path))


def write_category_ciftis(inputs: Dict, cats: Dict[str, np.ndarray], tau: float,
                          output_label: str,
                          template_path: Path | None = None) -> None:
    """Write three NaN-padded dscalars (shared, motor-unique, scene-unique) for one tau."""
    if template_path is None:
        template_path = _find_template_cifti()
    out_dir = PATHS['figures'] / 'ciftis' / output_label
    n_grayordinates = len(inputs['r2_full'])
    tau_tag = f'tau-{tau:.1f}'

    spec = [
        ('shared',       'r2_full',  cats['shared']),
        ('motor_unique', 'pm_motor', cats['motor_unique']),
        ('scene_unique', 'pm_scene', cats['scene_unique']),
    ]
    for cat_name, source_key, mask in spec:
        full = np.zeros(n_grayordinates, dtype=np.float32)
        full[mask] = inputs[source_key][mask]
        out_path = out_dir / f'vp_category_{cat_name}_{tau_tag}.dscalar.nii'
        _write_cifti_scalar(full, template_path, out_path,
                            map_name=f'{cat_name}_{tau_tag}')
        print(f"    cifti: wrote {out_path.name} "
              f"(n_voxels={int(mask.sum())}, template={template_path.name})")


def categorize_voxels(pm_motor: np.ndarray, pm_scene: np.ndarray,
                      r2_full: np.ndarray, sig_full: np.ndarray,
                      floor_r2: float, tau: float) -> Dict[str, np.ndarray]:
    """
    Returns boolean masks for the three categories plus the admitted pool.
    Inputs assumed already clipped at 0.
    """
    admitted = sig_full & np.isfinite(r2_full) & (r2_full >= floor_r2)

    with np.errstate(invalid='ignore', divide='ignore'):
        p_motor = np.where(r2_full > 0, pm_motor / r2_full, 0.0)
        p_scene = np.where(r2_full > 0, pm_scene / r2_full, 0.0)

    motor_unique = admitted & (p_motor >= tau)
    scene_unique = admitted & (p_scene >= tau)
    shared = admitted & ~motor_unique & ~scene_unique

    return {
        'admitted': admitted,
        'shared': shared,
        'motor_unique': motor_unique,
        'scene_unique': scene_unique,
    }


# ============================================================================
# Panel: three-category pie
# ============================================================================

def plot_three_category_pie(pm_motor: np.ndarray, pm_scene: np.ndarray,
                            r2_full: np.ndarray, cats: Dict[str, np.ndarray],
                            aggregation: str, output_path: Path) -> None:
    """aggregation: 'variance' (sum of category value) or 'count' (n voxels)."""
    if aggregation == 'variance':
        v_shared = float(r2_full[cats['shared']].sum())
        v_motor = float(pm_motor[cats['motor_unique']].sum())
        v_scene = float(pm_scene[cats['scene_unique']].sum())
    elif aggregation == 'count':
        v_shared = float(cats['shared'].sum())
        v_motor = float(cats['motor_unique'].sum())
        v_scene = float(cats['scene_unique'].sum())
    else:
        raise ValueError(aggregation)

    sizes = [v_shared, v_motor, v_scene]
    colors = [SHARED_COLOR, MOTOR_COLOR, SCENE_COLOR]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=None, colors=colors, startangle=90,
           wedgeprops={'edgecolor': 'black', 'linewidth': 4})
    ax.axis('equal')
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


# ============================================================================
# Panel: R2_full distribution vs shared subset
# ============================================================================

def plot_full_vs_shared_histogram(r2_full: np.ndarray, cats: Dict[str, np.ndarray],
                                  output_path: Path,
                                  bw_method: str = 'scott') -> None:
    """
    Overlaid KDEs of R2_full over admitted voxels vs R2_full over shared
    voxels (subset of admitted). Mirrors plot_full_vs_shared_kde from the
    earlier unique/shared pipeline.
    """
    full_vals = r2_full[cats['admitted']]
    shared_vals = r2_full[cats['shared']]
    full_vals = full_vals[np.isfinite(full_vals)]
    shared_vals = shared_vals[np.isfinite(shared_vals)]

    fig, ax = plt.subplots(figsize=(8, 8))

    if len(full_vals) < 2:
        save_figure(fig, output_path.parent, output_path.name)
        return

    x_min = float(full_vals.min())
    x_max = float(full_vals.max())
    x_eval = np.linspace(x_min, x_max, 500)

    density_full = stats.gaussian_kde(full_vals, bw_method=bw_method)(x_eval)
    ax.fill_between(x_eval, density_full, color=FULL_COLOR, alpha=0.5)
    ax.plot(x_eval, density_full, color='black', linewidth=2, linestyle='dashed')

    if len(shared_vals) >= 2:
        density_shared = stats.gaussian_kde(shared_vals, bw_method=bw_method)(x_eval)
        ax.fill_between(x_eval, density_shared, color=SHARED_COLOR, alpha=0.7)
        ax.plot(x_eval, density_shared, color='black', linewidth=2, linestyle='solid')

    handles = [
        mpatches.Patch(facecolor=FULL_COLOR, alpha=0.5,
                       edgecolor='black', linewidth=2, linestyle='dashed',
                       label='Full model'),
        mpatches.Patch(facecolor=SHARED_COLOR, alpha=0.7,
                       edgecolor='black', linewidth=2, linestyle='solid',
                       label='Shared'),
    ]
    ax.set_xlabel('R$^2$')
    ax.set_ylim(top=15)
    ax.legend(handles=handles, frameon=False,
              bbox_to_anchor=(0.98, 0.98), loc='upper right')
    add_clean_end_ticks(ax, y_margin=0.05)
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


# ============================================================================
# Panel: per-network mean pm (tau-free)
# ============================================================================

def plot_network_mean_pm(pm_motor: np.ndarray, pm_scene: np.ndarray,
                         sig_full: np.ndarray, network_labels: np.ndarray,
                         network_config: Dict, output_path: Path,
                         min_n: int) -> None:
    """
    Horizontal grouped bar chart of mean pm_motor and pm_scene per network,
    over all FDR-significant vertices. No R2 floor, no tau.
    Networks ordered by mean R2_full (= mean pm_motor + mean pm_scene) descending.
    """
    rows = []
    for net_idx, net_name in enumerate(network_config['labels']):
        net_id = net_idx + 1
        sel = (network_labels == net_id) & sig_full & np.isfinite(pm_motor) & np.isfinite(pm_scene)
        n = int(sel.sum())
        if n < min_n:
            continue
        rows.append({
            'Network': net_name,
            'motor': float(pm_motor[sel].mean()),
            'scene': float(pm_scene[sel].mean()),
        })

    fig, ax = plt.subplots(figsize=(8, max(6, len(rows) * 0.9)))
    if not rows:
        save_figure(fig, output_path.parent, output_path.name)
        return

    df = pd.DataFrame(rows)
    df['r2_full'] = df['motor'] + df['scene']
    df = df.sort_values('r2_full', ascending=True)  # ascending -> highest at top in barh

    y = np.arange(len(df))
    height = 0.35

    ax.barh(y + height / 2, df['motor'].values, height=height,
            color=MOTOR_COLOR, alpha=0.85, label='Motor')
    ax.barh(y - height / 2, df['scene'].values, height=height,
            color=SCENE_COLOR, alpha=0.85, label='Scene')

    ax.set_yticks(y)
    ax.set_yticklabels(df['Network'].values, fontsize=28)
    ax.set_xlabel('Mean R$^2$', fontsize=28)
    ax.tick_params(axis='x', labelsize=24)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(frameon=False, fontsize=24)

    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


# ============================================================================
# Panel: per-network pm ridges (tau-free)
# ============================================================================

def plot_network_pm_ridges(pm_motor: np.ndarray, pm_scene: np.ndarray,
                           sig_full: np.ndarray, network_labels: np.ndarray,
                           network_config: Dict, output_path: Path,
                           min_n: int) -> None:
    """
    Overlapping KDEs of pm_motor (blue) and pm_scene (orange) per network,
    over all FDR-significant vertices. No R2 floor, no tau.
    Networks ordered by mean R2_full descending (same as plot_network_mean_pm).
    """
    df_motor = _network_long_df(pm_motor, sig_full, network_labels, network_config)
    df_scene = _network_long_df(pm_scene, sig_full, network_labels, network_config)

    fig, ax = plt.subplots(figsize=(7, 10))
    if df_motor.empty and df_scene.empty:
        save_figure(fig, output_path.parent, output_path.name)
        return

    motor_counts = df_motor.groupby('Network').size() if not df_motor.empty else pd.Series(dtype=int)
    scene_counts = df_scene.groupby('Network').size() if not df_scene.empty else pd.Series(dtype=int)

    # Order by mean R2_full descending
    all_nets = set(motor_counts.index) | set(scene_counts.index)
    net_r2 = {}
    for net in all_nets:
        m = float(df_motor.loc[df_motor['Network'] == net, 'value'].mean()) if net in motor_counts.index else 0.0
        s = float(df_scene.loc[df_scene['Network'] == net, 'value'].mean()) if net in scene_counts.index else 0.0
        net_r2[net] = m + s
    network_order = [n for n, _ in sorted(net_r2.items(), key=lambda x: x[1], reverse=True)]

    kept_order = []
    for net in network_order:
        m_n = int(motor_counts.get(net, 0))
        s_n = int(scene_counts.get(net, 0))
        if m_n >= min_n or s_n >= min_n:
            kept_order.append(net)
    network_order = kept_order

    if not network_order:
        save_figure(fig, output_path.parent, output_path.name)
        return

    motor_map = {net: MOTOR_COLOR for net in network_order}
    scene_map = {net: SCENE_COLOR for net in network_order}

    vals_all = np.concatenate([df_motor['value'].values, df_scene['value'].values])
    x_min = 0.0
    x_max = float(np.percentile(vals_all, 99)) * 1.1

    y_positions = _draw_ridges(ax, df_motor, network_order, motor_map,
                               x_min, x_max, alpha=0.5, min_n=min_n)
    _draw_ridges(ax, df_scene, network_order, scene_map,
                 x_min, x_max, alpha=0.5, min_n=min_n)

    _format_ridge_axes(ax, x_min, x_max, y_positions,
                       network_order_top_down=network_order[::-1],
                       xlabel='R$^2$',
                       show_yticklabels=True)
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


# ============================================================================
# Panels: per-network ridge plots (categorical)
# ============================================================================

def _network_long_df(values: np.ndarray, voxel_mask: np.ndarray,
                     network_labels: np.ndarray, network_config: Dict
                     ) -> pd.DataFrame:
    rows = []
    for net_idx, net_name in enumerate(network_config['labels']):
        net_id = net_idx + 1
        sel = (network_labels == net_id) & voxel_mask & np.isfinite(values)
        if not sel.any():
            continue
        for v in values[sel]:
            rows.append({'Network': net_name, 'value': float(v)})
    return pd.DataFrame(rows)


def _draw_ridges(ax, df: pd.DataFrame, network_order: List[str],
                 color_for_network: Dict[str, str], x_min: float, x_max: float,
                 alpha: float = 0.6, bw_method: float = 0.5,
                 min_n: int = 2):
    n_networks = len(network_order)
    y_spacing = 1.0 / (n_networks + 1)
    y_positions = np.arange(n_networks) * y_spacing + y_spacing
    network_y_map = {net: y for net, y in zip(network_order[::-1], y_positions)}

    x_range = np.linspace(x_min, x_max, 500)
    for net in network_order[::-1]:
        y_base = network_y_map[net]
        net_data = df.loc[df['Network'] == net, 'value'].values
        if len(net_data) < min_n:
            continue
        kde = stats.gaussian_kde(net_data, bw_method=bw_method)
        density = kde(x_range)
        density_norm = density / density.max() * (y_spacing * 0.8)

        ax.plot([x_min, x_max], [y_base, y_base], color='black',
                linewidth=0.5, alpha=0.3, zorder=1)
        ax.fill_between(x_range, y_base, y_base + density_norm,
                        color=color_for_network[net], alpha=alpha, zorder=2)
        ax.plot(x_range, y_base + density_norm, color='black',
                linewidth=2, zorder=3)
    return y_positions


def _format_ridge_axes(ax, x_min: float, x_max: float, y_positions: np.ndarray,
                       network_order_top_down: List[str], xlabel: str,
                       show_yticklabels: bool) -> None:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='major', width=2, length=8,
                   labelsize=25, pad=10)
    ax.set_xlabel(xlabel, fontsize=28)
    ax.tick_params(axis='x', labelsize=28)
    ax.set_yticks(y_positions)
    if show_yticklabels:
        ax.set_yticklabels(network_order_top_down, fontsize=28, va='baseline')
    else:
        ax.set_yticklabels([])
    ax.tick_params(left=False)


def plot_shared_ridge(r2_full: np.ndarray, cats: Dict[str, np.ndarray],
                      network_labels: np.ndarray, network_config: Dict,
                      output_path: Path, min_n: int) -> None:
    """Per-network KDE of R2_full restricted to shared-category voxels.

    Networks with fewer than min_n shared voxels are dropped entirely
    (no ridge, no label).
    """
    df = _network_long_df(r2_full, cats['shared'], network_labels, network_config)
    fig, ax = plt.subplots(figsize=(7, 10))
    if df.empty:
        save_figure(fig, output_path.parent, output_path.name)
        return

    counts = df.groupby('Network').size()
    suppressed = sorted(counts[counts < min_n].index.tolist())
    df = df[df['Network'].isin(counts[counts >= min_n].index)]
    if df.empty:
        save_figure(fig, output_path.parent, output_path.name)
        return
    if suppressed:
        print(f"    shared: suppressed (n<{min_n}): {suppressed}")

    network_medians = df.groupby('Network')['value'].median().sort_values(ascending=False)
    network_order = network_medians.index.tolist()

    color_for_network = {}
    for net in network_order:
        idx = network_config['labels'].index(net)
        color_for_network[net] = network_config['colors'][idx]

    x_min = float(df['value'].min()) - 0.005
    x_max = float(df['value'].max()) * 1.1

    y_positions = _draw_ridges(ax, df, network_order, color_for_network,
                               x_min, x_max, alpha=0.6, min_n=min_n)
    _format_ridge_axes(ax, x_min, x_max, y_positions,
                       network_order_top_down=network_order[::-1],
                       xlabel='R$^2$ shared',
                       show_yticklabels=True)
    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


def plot_unique_ridges_overlapping(pm_motor: np.ndarray, pm_scene: np.ndarray,
                                   cats: Dict[str, np.ndarray],
                                   network_labels: np.ndarray, network_config: Dict,
                                   output_path: Path, min_n: int) -> None:
    """
    Per-network overlapping KDEs of pm_motor (motor-unique voxels) and pm_scene
    (scene-unique voxels). Networks with both motor-unique and scene-unique
    counts below min_n are dropped entirely (no row, no label). Per-band ridges
    are individually suppressed when their own count is below min_n.
    Ordering by median shared R2_full so the figure aligns with plot_shared_ridge.
    """
    df_motor = _network_long_df(pm_motor, cats['motor_unique'],
                                network_labels, network_config)
    df_scene = _network_long_df(pm_scene, cats['scene_unique'],
                                network_labels, network_config)

    fig, ax = plt.subplots(figsize=(10, 10))
    if df_motor.empty and df_scene.empty:
        save_figure(fig, output_path.parent, output_path.name)
        return

    df_shared_for_order = _network_long_df(pm_motor + pm_scene, cats['shared'],
                                           network_labels, network_config)
    if not df_shared_for_order.empty:
        medians = (df_shared_for_order.groupby('Network')['value']
                                       .median().sort_values(ascending=False))
        network_order = medians.index.tolist()
    else:
        combined = pd.concat([df_motor, df_scene], ignore_index=True)
        medians = combined.groupby('Network')['value'].median().sort_values(ascending=False)
        network_order = medians.index.tolist()

    motor_counts = df_motor.groupby('Network').size() if not df_motor.empty else pd.Series(dtype=int)
    scene_counts = df_scene.groupby('Network').size() if not df_scene.empty else pd.Series(dtype=int)

    kept_order = []
    suppressed_row = []
    suppressed_band = []
    for net in network_order:
        m_n = int(motor_counts.get(net, 0))
        s_n = int(scene_counts.get(net, 0))
        if m_n < min_n and s_n < min_n:
            suppressed_row.append(f"{net} (m={m_n}, s={s_n})")
            continue
        kept_order.append(net)
        if m_n < min_n:
            suppressed_band.append(f"{net} motor (n={m_n})")
        if s_n < min_n:
            suppressed_band.append(f"{net} scene (n={s_n})")
    network_order = kept_order
    if not network_order:
        save_figure(fig, output_path.parent, output_path.name)
        return
    if suppressed_row:
        print(f"    unique: rows dropped (m<{min_n} and s<{min_n}): {suppressed_row}")
    if suppressed_band:
        print(f"    unique: bands suppressed within kept rows: {suppressed_band}")

    motor_map = {net: MOTOR_COLOR for net in network_order}
    scene_map = {net: SCENE_COLOR for net in network_order}

    vals_all = np.concatenate([df_motor['value'].values, df_scene['value'].values])
    x_min = float(vals_all.min()) - 0.005
    x_max = float(vals_all.max()) * 1.1

    y_positions = _draw_ridges(ax, df_motor, network_order, motor_map,
                               x_min, x_max, alpha=0.5, min_n=min_n)
    _draw_ridges(ax, df_scene, network_order, scene_map, x_min, x_max,
                 alpha=0.5, min_n=min_n)

    _format_ridge_axes(ax, x_min, x_max, y_positions,
                       network_order_top_down=network_order[::-1],
                       xlabel='R$^2$ unique',
                       show_yticklabels=True)

    plt.tight_layout()
    save_figure(fig, output_path.parent, output_path.name)


# ============================================================================
# Main
# ============================================================================

def _summarize_categories(cats: Dict[str, np.ndarray], tau: float) -> str:
    n_adm = int(cats['admitted'].sum())
    if n_adm == 0:
        return f"  tau={tau}: no admitted voxels"
    return (
        f"  tau={tau}: admitted={n_adm} | "
        f"shared={int(cats['shared'].sum())} "
        f"({100 * cats['shared'].sum() / n_adm:.1f}%) | "
        f"motor-unique={int(cats['motor_unique'].sum())} "
        f"({100 * cats['motor_unique'].sum() / n_adm:.1f}%) | "
        f"scene-unique={int(cats['scene_unique'].sum())} "
        f"({100 * cats['scene_unique'].sum() / n_adm:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--group', type=str, required=True,
                        choices=list(GROUP_REGISTRY.keys()))
    parser.add_argument('--fdr-alpha', type=float, default=None)
    parser.add_argument('--floor-r2', type=float, default=FLOOR_R2)
    parser.add_argument('--taus', type=float, nargs='+', default=TAUS)
    parser.add_argument('--min-n-for-ridge', type=int, default=MIN_N_FOR_RIDGE,
                        help='Minimum per-network voxel count to draw a ridge; '
                             'below this the ridge and its label are suppressed.')
    parser.add_argument('--cifti-taus', type=float, nargs='*', default=[0.6],
                        help='Tau values for which to also emit three category '
                             'dscalar.nii (shared, motor_unique, scene_unique). '
                             'Empty list disables CIFTI output.')
    args = parser.parse_args()

    fdr_alpha = args.fdr_alpha if args.fdr_alpha is not None \
        else PARAMETERS['permutation']['fdr_alpha']

    sns_styleset()
    plt.rcParams['svg.fonttype'] = 'none'
    inputs = load_group_inputs(args.group, fdr_alpha=fdr_alpha)
    network_labels = load_network_parcellation(Path(PATHS['ColeAnticevic_networks']))
    network_config = get_cole_anticevic_config()

    out_dir = PATHS['figures'] / 'vp_decomposition' / inputs['output_label']
    print(f"Group: {args.group}")
    print(f"FDR alpha: {fdr_alpha}  floor_r2: {args.floor_r2}")
    print(f"FDR-significant (r_full): {int(inputs['sig_full'].sum())} voxels")
    print(f"Output directory: {out_dir}")

    plot_network_mean_pm(
        inputs['pm_motor'], inputs['pm_scene'], inputs['sig_full'],
        network_labels, network_config,
        output_path=out_dir / 'network_mean_pm.svg',
        min_n=args.min_n_for_ridge,
    )
    plot_network_pm_ridges(
        inputs['pm_motor'], inputs['pm_scene'], inputs['sig_full'],
        network_labels, network_config,
        output_path=out_dir / 'network_pm_ridges.svg',
        min_n=args.min_n_for_ridge,
    )

    for tau in args.taus:
        cats = categorize_voxels(
            inputs['pm_motor'], inputs['pm_scene'], inputs['r2_full'],
            inputs['sig_full'], args.floor_r2, tau,
        )
        print(_summarize_categories(cats, tau))

        tau_tag = f'tau-{tau:.1f}'

        plot_three_category_pie(
            inputs['pm_motor'], inputs['pm_scene'], inputs['r2_full'], cats,
            aggregation='variance',
            output_path=out_dir / f'pie_variance_{tau_tag}.svg',
        )
        plot_three_category_pie(
            inputs['pm_motor'], inputs['pm_scene'], inputs['r2_full'], cats,
            aggregation='count',
            output_path=out_dir / f'pie_count_{tau_tag}.svg',
        )
        plot_full_vs_shared_histogram(
            inputs['r2_full'], cats,
            output_path=out_dir / f'histogram_full_vs_shared_{tau_tag}.svg',
        )
        plot_shared_ridge(
            inputs['r2_full'], cats, network_labels, network_config,
            output_path=out_dir / f'ridges_shared_{tau_tag}.svg',
            min_n=args.min_n_for_ridge,
        )
        plot_unique_ridges_overlapping(
            inputs['pm_motor'], inputs['pm_scene'], cats,
            network_labels, network_config,
            output_path=out_dir / f'ridges_unique_overlapping_{tau_tag}.svg',
            min_n=args.min_n_for_ridge,
        )

        if any(abs(tau - t) < 1e-9 for t in args.cifti_taus):
            cats_cifti = categorize_voxels(
                inputs['pm_motor'], inputs['pm_scene'], inputs['r2_full'],
                inputs['sig_full'], floor_r2=0.0, tau=tau,
            )
            write_category_ciftis(inputs, cats_cifti, tau, inputs['output_label'])

    print("Done.")


if __name__ == '__main__':
    main()
