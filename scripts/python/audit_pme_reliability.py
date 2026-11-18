"""Stage B audit of held-out BOLD reliability within level attempts.

Reads the Stage A attempt table, loads held-out BOLD for the primary group, and
measures split-half reliability, same-level versus different-level pattern
similarity and active-TR coverage. No encoding model is fitted. Figures are
produced by plot_pme_reliability.py.
"""
import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from mario_encoding.config import PARAMETERS, PATHS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY
from mario_encoding.utils.data_loading import (
    load_practice_fmri,
    load_practice_metadata,
    zscore_runs_active,
)
from mario_encoding.utils.screen_annotations import (
    aggregate_screen_fractions,
    compute_screen_gameplay_mask,
    create_screen_columns,
)

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}
DEFAULT_GROUP = 'GROUP_N5_PRIMARY'

TR = float(PARAMETERS['TR'])
FRAME_RATE = float(PARAMETERS['frame_rate'])
SCREEN_RULE = PARAMETERS['screen_gameplay_mask']

STAGE_A_TABLE = (PATHS['variance_partitioning'].parent / 'pme_audit' /
                 'attempt_table.csv')
DEFAULT_MASK_NAME = 'group_r_full_sig_conj_min4.dscalar.nii'
MIN_HALF_TRS = 3
MIN_ATTEMPT_TRS = 2 * MIN_HALF_TRS
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# Provisional reliability rule, fixed before inspecting held-out BOLD.
MIN_SUBJECTS_PASS = 4
MIN_RELIABLE_LEVELS = 10
MIN_MEDIAN_SPLIT_HALF = 0.50
MIN_SAME_LEVEL_R = 0.10


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(output_dir / 'audit.log'),
                  logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def safe_median(series):
    """Median that returns NaN for an all-missing series without warnings."""
    values = series.dropna()
    return float(values.median()) if len(values) else np.nan


def frame_to_markdown(df):
    """Render a DataFrame as a simple Markdown table without tabulate."""
    columns = [str(col) for col in df.columns]
    lines = ['| ' + ' | '.join(columns) + ' |',
             '|' + '|'.join(['---'] * len(columns)) + '|']
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float) and np.isfinite(value):
                values.append(f'{value:.3f}')
            elif isinstance(value, float):
                values.append('')
            else:
                values.append(str(value).replace('|', '/'))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def relative_path(filename):
    """Return the annotation-style path starting at the subject directory."""
    text = str(filename)
    index = text.find('sub-')
    if index < 0:
        raise ValueError(f'Cannot find sub- in filename: {filename}')
    return text[index:]


def find_run_path(directory, subject, session, run, suffix):
    return (Path(directory) / f'sub-{subject:02d}' / f'ses-{session:03d}' /
            f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_{suffix}.tsv')


def find_annotation_path(root, subject, session, run):
    return (Path(root) / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func' /
            f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run:02d}'
            '_desc-annotated_events.tsv')


def find_vp_dir(subject_cfg):
    """Return the production variance-partitioning directory for one subject."""
    train = subject_cfg['train_sessions']
    test = subject_cfg['test_sessions']
    train_label = f'{min(train):03d}-{max(train):03d}'
    test_label = f'{min(test):03d}-{max(test):03d}'
    dataset = (f"sub-{int(subject_cfg['subject']):02d}_train-ses-{train_label}"
               f'_test-ses-{test_label}')
    return (PATHS['variance_partitioning'] / dataset /
            subject_cfg['experiment_id'])


def load_saved_active_count(subject_cfg, logger):
    """Read the saved fit's retained test-TR count for a mask sanity check."""
    path = find_vp_dir(subject_cfg) / 'significance.npz'
    if not path.exists():
        logger.warning('No significance.npz for sub-%02d; cannot compare masks',
                       int(subject_cfg['subject']))
        return np.nan
    with np.load(path) as data:
        if 'n_test_active' not in data:
            logger.warning('No n_test_active in %s', path)
            return np.nan
        return int(data['n_test_active'])


def load_mask(mask_path, logger):
    """Load a group dscalar mask and restrict it to cortex."""
    image = nib.load(str(mask_path))
    data = image.get_fdata(dtype=np.float32).squeeze()
    if data.ndim != 1:
        raise ValueError(f'Mask {mask_path} is not one-dimensional: {data.shape}')
    axis = image.header.get_axis(1)
    structures = np.asarray([
        name.decode() if isinstance(name, bytes) else str(name)
        for name in axis.name])
    if structures.shape != data.shape:
        raise ValueError(
            f'Mask structures {structures.shape} do not match data {data.shape}')
    cortex = np.char.startswith(structures, 'CIFTI_STRUCTURE_CORTEX_')
    mask = (data > 0.5) & cortex
    if not mask.any():
        raise ValueError(f'Mask {mask_path} is empty after cortex restriction')
    logger.info('Mask: %s (%d cortical vertices of %d total)',
                mask_path, int(mask.sum()), int(cortex.sum()))
    return mask


def load_subject_mask(subject_cfg, group_mask, logger):
    """Intersect the group mask with the subject's fitted valid-voxel mask."""
    vp_dir = find_vp_dir(subject_cfg)
    valid_path = vp_dir / 'valid_voxels_mask.npy'
    if valid_path.exists():
        valid = np.load(valid_path).astype(bool)
        if valid.shape != group_mask.shape:
            raise ValueError(
                f'Subject mask {valid.shape} does not match group mask '
                f'{group_mask.shape}')
    else:
        logger.warning('Missing valid_voxels_mask for sub-%02d; using group mask',
                       int(subject_cfg['subject']))
        valid = np.ones(group_mask.shape, dtype=bool)
    subject_mask = group_mask & valid
    if not subject_mask.any():
        raise ValueError(
            f'No vertices remain for sub-{int(subject_cfg["subject"]):02d}')
    logger.info('sub-%02d mask: %d of %d group vertices',
                int(subject_cfg['subject']), int(subject_mask.sum()),
                int(group_mask.sum()))
    return subject_mask


def load_attempt_table(path, group, logger):
    """Load the Stage A attempt table and restrict it to the requested group."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'Stage A attempt table not found: {path}. '
            'Run scripts/python/audit_pme_epochs.py first.')
    attempts = pd.read_csv(path)
    required = {'subject', 'session', 'run', 'rep_id', 'attempt_id', 'level',
                'outcome', 'session_tr_start', 'session_tr_end', 'n_trs'}
    missing = sorted(required - set(attempts.columns))
    if missing:
        raise ValueError(f'Attempt table missing columns: {missing}')

    attempts['subject'] = attempts['subject'].astype(int)
    attempts['session'] = attempts['session'].astype(int)
    expected = {(int(subject_cfg['subject']), int(session))
                for subject_cfg in group['subjects']
                for session in subject_cfg['test_sessions']}
    present = set(zip(attempts['subject'], attempts['session']))
    missing_pairs = expected - present
    if missing_pairs:
        raise ValueError(f'Attempt table has no rows for: {sorted(missing_pairs)}')
    extra_pairs = present - expected
    if extra_pairs:
        logger.warning('Ignoring attempts outside the group: %s',
                       sorted(extra_pairs))
    keep = attempts.apply(
        lambda row: (int(row['subject']), int(row['session'])) in expected,
        axis=1)
    attempts = attempts.loc[keep].sort_values(
        ['subject', 'session', 'session_tr_start']).reset_index(drop=True)
    logger.info('Attempts: %d across %d subject-session pairs',
                len(attempts), len(expected))
    return attempts


def read_run_screen_tables(subject, session, run, combined_dir,
                           annotation_root, logger):
    """Read the combined repvars and validated annotated-event tables."""
    combined_path = find_run_path(
        combined_dir, subject, session, run, 'desc-combined')
    annotation_path = find_annotation_path(
        annotation_root, subject, session, run)
    if not combined_path.exists():
        raise FileNotFoundError(f'Combined file not found: {combined_path}')
    if not annotation_path.exists():
        raise FileNotFoundError(f'Annotation file not found: {annotation_path}')

    combined = pd.read_csv(combined_path, sep='\t')
    required = {'filename', 'frame_index', 'frame_time_in_run'}
    missing = sorted(required - set(combined.columns))
    if missing:
        raise ValueError(f'{combined_path.name} missing columns: {missing}')
    combined['relative_path'] = combined['filename'].map(relative_path)

    annotations = pd.read_csv(
        annotation_path, sep='\t',
        dtype={'trial_type': str, 'stim_file': str})
    return combined, annotations, combined_path.name


def build_tr_screen_fractions(combined, annotations, n_trs, label):
    """Aggregate framewise Screen labels into per-TR gameplay fractions."""
    with_screen, coverage = create_screen_columns(
        combined, annotations, frame_rate=FRAME_RATE)
    with_screen['TR_bin'] = np.floor(
        with_screen['frame_time_in_run'] / TR).astype(int)
    per_tr = (with_screen.groupby('TR_bin', sort=True)
              .apply(aggregate_screen_fractions, include_groups=False))

    expected_index = np.arange(n_trs)
    if len(per_tr) and (per_tr.index.min() < 0 or
                        per_tr.index.max() >= n_trs):
        raise ValueError(
            f'{label}: screen TR bins outside 0..{n_trs - 1}')
    per_tr = per_tr.reindex(expected_index).fillna(0.0)
    p = per_tr['screen_gameplay'].to_numpy(dtype=float)
    n_empty = int((per_tr['screen_n_frames'].to_numpy(dtype=float) == 0).sum())
    return p, coverage, n_empty


def load_screen_for_run(subject, session, run, n_trs, combined_dir,
                        annotation_root, logger):
    """Build the Screen/Gameplay mask for one run from validated annotations."""
    combined, annotations, label = read_run_screen_tables(
        subject, session, run, combined_dir, annotation_root, logger)
    p, coverage, n_empty = build_tr_screen_fractions(
        combined, annotations, n_trs, label)
    active = compute_screen_gameplay_mask(
        p, rule=SCREEN_RULE['rule'], threshold=SCREEN_RULE['threshold'],
        logger=None)
    logger.debug(
        'sub-%02d ses-%03d run-%02d: %d reps, %d TRs, %d gameplay TRs, '
        '%d TRs without frames',
        subject, session, run, coverage['n_repetitions'], n_trs,
        int(active.sum()), n_empty)
    return p, active


def load_session_bold(subject, session, metadata, subject_mask, annotation_root,
                      logger):
    """Load and z-score held-out BOLD, returning the true gameplay mask."""
    pipeline = PARAMETERS['preprocessing_pipeline']
    fmri_path = (PATHS['hcp_data'] if pipeline == 'hcp'
                 else PATHS['fmriprep_data'])
    Y = load_practice_fmri(
        subject, session, metadata['practice_runs'], fmri_path, pipeline)
    if Y.shape[1] != subject_mask.size:
        raise ValueError(
            f'sub-{subject:02d} ses-{session:03d}: BOLD has {Y.shape[1]} '
            f'grayordinates, mask has {subject_mask.size}')
    Y = Y[:, subject_mask]
    logger.info('sub-%02d ses-%03d BOLD: %s, mask vertices: %d',
                subject, session, Y.shape, int(subject_mask.sum()))

    n_trs_per_run = [int(value) for value in metadata['n_trs_per_run']]
    run_onsets = np.concatenate(
        [[0], np.cumsum(n_trs_per_run)]).astype(int)[:-1]
    screen_parts = []
    active_parts = []
    for run_index, run in enumerate(metadata['practice_runs']):
        p, active = load_screen_for_run(
            subject, session, run, n_trs_per_run[run_index],
            PATHS['per_run_combined_repvars_and_bids_events'],
            annotation_root, logger)
        screen_parts.append(p)
        active_parts.append(active)
    screen_p = np.concatenate(screen_parts)
    active_mask = np.concatenate(active_parts)
    if active_mask.size != Y.shape[0]:
        raise ValueError(
            f'sub-{subject:02d} ses-{session:03d}: screen mask has '
            f'{active_mask.size} TRs, BOLD has {Y.shape[0]}')
    Y = zscore_runs_active(Y, run_onsets, active_mask, logger)
    logger.info('sub-%02d ses-%03d gameplay TRs: %d of %d',
                subject, session, int(active_mask.sum()), active_mask.size)
    return Y, active_mask, screen_p


def pearson_corr(a, b):
    """Pearson correlation between two vectors, robust to zero variance."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        return np.nan
    return float(np.dot(a, b) / denominator)


def attempt_window(attempt, active_mask, logger):
    """Return the TR range and active TR indices for one attempt."""
    start = int(attempt['session_tr_start'])
    end = int(attempt['session_tr_end'])
    if start < 0 or end >= active_mask.size or end < start:
        logger.warning('Skipping %s: TR range [%d, %d] outside BOLD',
                       attempt['attempt_id'], start, end)
        return None
    tr_index = np.arange(start, end + 1)
    active_index = tr_index[active_mask[tr_index]]
    return tr_index, active_index


def split_half_reliability(Y, active_index):
    """Odd/even split-half correlation and Spearman-Brown correction."""
    odd = active_index[0::2]
    even = active_index[1::2]
    if len(odd) < MIN_HALF_TRS or len(even) < MIN_HALF_TRS:
        return np.nan, np.nan, 'short_split_half'

    odd_pattern = Y[odd].mean(axis=0, dtype=np.float64)
    even_pattern = Y[even].mean(axis=0, dtype=np.float64)
    split_r = pearson_corr(odd_pattern, even_pattern)
    if not np.isfinite(split_r):
        return np.nan, np.nan, 'undefined_split_half'
    split_sb = (2 * split_r / (1 + split_r)
                if 1 + split_r > 0 else np.nan)
    if not np.isfinite(split_sb):
        return np.nan, np.nan, 'undefined_split_half'
    return split_r, split_sb, ''


def build_attempt_record(Y, attempt, tr_index, active_index, screen_p):
    """Build one attempt row and, when reliable, its mean pattern."""
    n_trs = len(tr_index)
    n_active = len(active_index)
    record = attempt.to_dict()
    record.update({
        'attempt_key': (f'sub-{int(attempt["subject"]):02d}_'
                        f'ses-{int(attempt["session"]):03d}_'
                        f'run-{int(attempt["run"]):02d}_'
                        f'{attempt["attempt_id"]}'),
        'session_run': f'{int(attempt["session"])}_run-'
                       f'{int(attempt["run"]):02d}',
        'session_rep': (f'{int(attempt["session"])}_run-'
                        f'{int(attempt["run"]):02d}_{attempt["rep_id"]}'),
        'n_trs_observed': n_trs,
        'n_active_trs_screen': n_active,
        'active_fraction_screen': n_active / n_trs if n_trs else np.nan,
        'mean_screen_gameplay': (float(screen_p[tr_index].mean())
                                 if n_trs else np.nan),
        'included': False,
        'exclusion_reason': '',
        'split_half_r': np.nan,
        'split_half_sb': np.nan,
    })

    if n_active < MIN_ATTEMPT_TRS:
        record['exclusion_reason'] = 'few_active_trs'
        return record, None, None

    split_r, split_sb, reason = split_half_reliability(Y, active_index)
    if reason:
        record['exclusion_reason'] = reason
        return record, None, None
    pattern = Y[active_index].mean(axis=0, dtype=np.float64)

    record.update({
        'included': True,
        'split_half_r': split_r,
        'split_half_sb': split_sb,
    })
    pattern_row = {
        'attempt_id': attempt['attempt_id'],
        'attempt_key': record['attempt_key'],
        'level': attempt['level'],
        'session': int(attempt['session']),
        'run': int(attempt['run']),
        'rep_id': attempt['rep_id'],
        'outcome': attempt['outcome'],
        'session_run': record['session_run'],
        'session_rep': record['session_rep'],
        'split_half_sb': split_sb,
        'n_active_trs_screen': n_active,
        'session_tr_start': int(attempt['session_tr_start']),
        'session_tr_end': int(attempt['session_tr_end']),
    }
    return record, pattern.astype(np.float32), pattern_row


def extract_attempt_metrics(Y, active_mask, screen_p, attempts, logger):
    """Compute reliability and mean patterns for every attempt in one session."""
    rows = []
    patterns = []
    pattern_rows = []

    for _, attempt in attempts.iterrows():
        window = attempt_window(attempt, active_mask, logger)
        if window is None:
            continue
        tr_index, active_index = window
        record, pattern, pattern_row = build_attempt_record(
            Y, attempt, tr_index, active_index, screen_p)
        rows.append(record)
        if pattern is not None:
            patterns.append(pattern)
            pattern_rows.append(pattern_row)

    if not rows:
        raise ValueError('No attempts with valid TR ranges in this session')
    metrics = pd.DataFrame(rows)
    pattern_meta = pd.DataFrame(pattern_rows)
    if patterns:
        pattern_array = np.vstack(patterns)
    else:
        pattern_array = np.empty((0, Y.shape[1]), dtype=np.float32)
    logger.info(
        'Attempts: %d, reliable patterns: %d, median split-half: %.3f',
        len(metrics), len(pattern_array), safe_median(metrics['split_half_sb']))
    return metrics, pattern_array, pattern_meta


def fisher_mean_z(values):
    """Mean Fisher z for a set of correlation values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    clipped = np.clip(values, -1 + 1e-6, 1 - 1e-6)
    return float(np.mean(np.arctanh(clipped)))


def similarity_stats(r_pairs, levels, runs, reps, upper, valid):
    """Summarise same-level versus different-level pattern correlations."""
    same_level = (levels[:, None] == levels[None, :])[upper]
    same_run = (runs[:, None] == runs[None, :])[upper]
    same_rep = (reps[:, None] == reps[None, :])[upper]
    pair_valid = valid & ~same_rep
    same_level_pair = pair_valid & same_level
    different_level_pair = pair_valid & ~same_level
    same_run_pair = same_run

    same_level_same_run = same_level_pair & same_run_pair
    different_level_same_run = different_level_pair & same_run_pair
    same_level_diff_run = same_level_pair & ~same_run_pair
    different_level_diff_run = different_level_pair & ~same_run_pair

    same_z = fisher_mean_z(r_pairs[same_level_pair])
    different_z = fisher_mean_z(r_pairs[different_level_pair])
    same_run_same_z = fisher_mean_z(r_pairs[same_level_same_run])
    same_run_different_z = fisher_mean_z(r_pairs[different_level_same_run])

    def delta_z(first, second):
        return (first - second if np.isfinite(first) and np.isfinite(second)
                else np.nan)

    return {
        'same_level_r': np.tanh(same_z) if np.isfinite(same_z) else np.nan,
        'different_level_r': (np.tanh(different_z)
                              if np.isfinite(different_z) else np.nan),
        'delta_z': delta_z(same_z, different_z),
        'same_run_same_level_r': (np.tanh(same_run_same_z)
                                  if np.isfinite(same_run_same_z) else np.nan),
        'same_run_different_level_r': (np.tanh(same_run_different_z)
                                       if np.isfinite(same_run_different_z)
                                       else np.nan),
        'same_run_delta_z': delta_z(same_run_same_z, same_run_different_z),
        'n_same_pairs': int(same_level_pair.sum()),
        'n_different_pairs': int(different_level_pair.sum()),
        'n_same_run_same_pairs': int(same_level_same_run.sum()),
        'n_same_run_different_pairs': int(different_level_same_run.sum()),
        'n_same_level_diff_run_pairs': int(same_level_diff_run.sum()),
        'n_different_level_diff_run_pairs': int(different_level_diff_run.sum()),
    }


def permutation_test(r_pairs, levels, runs, reps, n_permutations, rng):
    """Permute level labels within run and compare same-level similarity."""
    n_attempts = len(levels)
    upper = np.triu_indices(n_attempts, k=1)
    valid = np.isfinite(r_pairs)
    observed = similarity_stats(r_pairs, levels, runs, reps, upper, valid)
    if not np.isfinite(observed['delta_z']):
        return observed, np.nan, np.nan

    null_all = []
    null_same_run = []
    for _ in range(n_permutations):
        permuted = levels.copy()
        for run in np.unique(runs):
            indices = np.flatnonzero(runs == run)
            if len(indices) > 1:
                permuted[indices] = rng.permutation(levels[indices])
        stats = similarity_stats(r_pairs, permuted, runs, reps, upper, valid)
        null_all.append(stats['delta_z'])
        null_same_run.append(stats['same_run_delta_z'])

    def p_value(observed_value, null_values):
        null_values = np.asarray(null_values, dtype=float)
        null_values = null_values[np.isfinite(null_values)]
        if not np.isfinite(observed_value) or not null_values.size:
            return np.nan
        return float((1 + np.sum(null_values >= observed_value)) /
                     (1 + null_values.size))

    return observed, p_value(observed['delta_z'], null_all), p_value(
        observed['same_run_delta_z'], null_same_run)


def compute_similarity_summary(patterns, pattern_meta, n_permutations, rng):
    """Correlation matrix and same-level permutation statistics."""
    if len(patterns) >= 2:
        with np.errstate(invalid='ignore', divide='ignore'):
            correlation = np.corrcoef(patterns)
        upper = np.triu_indices(len(patterns), k=1)
        r_pairs = correlation[upper]
        return permutation_test(
            r_pairs,
            pattern_meta['level'].astype(str).to_numpy(),
            pattern_meta['session_run'].astype(str).to_numpy(),
            pattern_meta['session_rep'].astype(str).to_numpy(),
            n_permutations, rng)

    observed = similarity_stats(
        np.array([], dtype=float),
        np.array([], dtype=object), np.array([], dtype=object),
        np.array([], dtype=object), (np.array([], dtype=int),
                                     np.array([], dtype=int)),
        np.array([], dtype=bool))
    return observed, np.nan, np.nan


def apply_subject_rule(row):
    """Add reliability, coverage and similarity pass flags to a summary row."""
    reliability_pass = bool(
        np.isfinite(row['median_split_half_sb']) and
        row['median_split_half_sb'] >= MIN_MEDIAN_SPLIT_HALF)
    coverage_pass = bool(
        row['n_levels_ge2_reliable'] >= MIN_RELIABLE_LEVELS)
    similarity_pass = bool(
        np.isfinite(row['same_level_r']) and
        row['same_level_r'] >= MIN_SAME_LEVEL_R and
        np.isfinite(row['delta_z']) and row['delta_z'] > 0 and
        np.isfinite(row['p_all']) and row['p_all'] < 0.05)
    row['reliability_pass'] = reliability_pass
    row['coverage_pass'] = coverage_pass
    row['similarity_pass'] = similarity_pass
    row['subject_pass'] = bool(reliability_pass and coverage_pass and
                               similarity_pass)
    return row


def summarize_subject(subject, attempt_df, pattern_meta, patterns,
                      n_permutations, rng, logger):
    """Build one subject summary row and the per-subject similarity statistics."""
    included = attempt_df[attempt_df['included']]
    levels = included['level'].astype(str).to_numpy()
    level_counts = pd.Series(levels).value_counts()
    n_levels_ge2 = int((level_counts >= 2).sum())

    observed, p_all, p_same_run = compute_similarity_summary(
        patterns, pattern_meta, n_permutations, rng)

    median_split = safe_median(included['split_half_sb'])
    row = {
        'subject': subject,
        'n_attempts': len(attempt_df),
        'n_included': len(included),
        'n_levels_included': int(included['level'].nunique()),
        'n_levels_ge2_reliable': n_levels_ge2,
        'median_split_half_sb': median_split,
        'median_active_trs': safe_median(included['n_active_trs_screen']),
        'median_attempt_trs': safe_median(included['n_trs_observed']),
        'same_level_r': observed['same_level_r'],
        'different_level_r': observed['different_level_r'],
        'delta_z': observed['delta_z'],
        'p_all': p_all,
        'same_run_same_level_r': observed['same_run_same_level_r'],
        'same_run_different_level_r': observed['same_run_different_level_r'],
        'same_run_delta_z': observed['same_run_delta_z'],
        'p_same_run': p_same_run,
        'n_same_pairs': observed['n_same_pairs'],
        'n_different_pairs': observed['n_different_pairs'],
        'n_same_run_same_pairs': observed['n_same_run_same_pairs'],
        'n_same_run_different_pairs': observed['n_same_run_different_pairs'],
    }
    apply_subject_rule(row)
    logger.info(
        'sub-%02d: included=%d, levels>=2=%d, median split-half=%.3f, '
        'same=%.3f, different=%.3f, p=%.4f',
        subject, len(included), n_levels_ge2, median_split,
        observed['same_level_r'], observed['different_level_r'], p_all)
    return row


def summarize_levels(attempt_df):
    """One row per subject and level with inclusion and reliability counts."""
    if attempt_df.empty:
        return pd.DataFrame()
    rows = []
    for (subject, level), group in attempt_df.groupby(['subject', 'level']):
        outcomes = group['outcome'].value_counts()
        included = group[group['included']]
        rows.append({
            'subject': subject,
            'level': level,
            'n_attempts': len(group),
            'n_included': len(included),
            'n_completed': int(outcomes.get('completed', 0)),
            'n_failed': int(outcomes.get('failed', 0)),
            'n_ambiguous': int(outcomes.get('ambiguous', 0)),
            'median_n_trs': safe_median(group['n_trs_observed']),
            'median_active_trs': safe_median(group['n_active_trs_screen']),
            'median_split_half_sb': safe_median(included['split_half_sb']),
        })
    return pd.DataFrame(rows)


def save_patterns(output_dir, subject, patterns, pattern_meta, vertices,
                  mask_name, logger):
    """Save the expensive attempt-by-vertex patterns for later analyses."""
    if not len(patterns):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'sub-{subject:02d}_attempt_patterns.npz'
    np.savez_compressed(
        path,
        patterns=patterns.astype(np.float32),
        attempt_id=pattern_meta['attempt_id'].to_numpy(dtype=str),
        attempt_key=pattern_meta['attempt_key'].to_numpy(dtype=str),
        level=pattern_meta['level'].to_numpy(dtype=str),
        session=pattern_meta['session'].to_numpy(dtype=np.int16),
        run=pattern_meta['run'].to_numpy(dtype=np.int16),
        rep_id=pattern_meta['rep_id'].to_numpy(dtype=str),
        outcome=pattern_meta['outcome'].to_numpy(dtype=str),
        split_half_sb=pattern_meta['split_half_sb'].to_numpy(dtype=np.float32),
        n_active_trs_screen=pattern_meta['n_active_trs_screen'].to_numpy(
            dtype=np.int16),
        vertices=vertices.astype(np.int32),
        mask_name=str(mask_name),
    )
    logger.info('Saved patterns: %s (%s)', path, patterns.shape)


def write_summary_markdown(path, group_name, mask_path, subject_df,
                           level_df, n_subjects, logger):
    passed = subject_df[subject_df['subject_pass']]['subject'].astype(int).tolist()
    required_subjects = min(MIN_SUBJECTS_PASS, n_subjects)
    decision = 'PASS' if len(passed) >= required_subjects else 'FAIL'
    lines = [
        '# PMe reliability audit, Stage B',
        '',
        f'Group: `{group_name}`. Mask: `{mask_path}`.',
        '',
        '## Decision rule',
        '',
        f'Provisional rule fixed before inspecting held-out BOLD: a subject '
        f'passes when it has at least {MIN_RELIABLE_LEVELS} levels with at least '
        f'two attempts of {MIN_ATTEMPT_TRS} or more Screen/Gameplay TRs, a '
        f'median split-half reliability of at least {MIN_MEDIAN_SPLIT_HALF:.2f}, '
        f'and same-level similarity of at least {MIN_SAME_LEVEL_R:.2f} that '
        f'exceeds different-level similarity with a within-run label-shuffle '
        f'permutation p below 0.05. The group passes with at least '
        f'{required_subjects} of {n_subjects} subjects.',
        '',
        f'**Decision: {decision}.** Subjects meeting all criteria: '
        f'{passed if passed else "none"}.',
        '',
        '## Subject summary',
        '',
        frame_to_markdown(subject_df),
        '',
        '## Notes',
        '',
        '- Attempts come from the Stage A table, which splits each level presentation at raw life-loss events and starts the next attempt after the 3 s death pause.',
        '- The Screen/Gameplay mask is rebuilt from the validated annotated-event files, because the saved downsampled TSVs do not contain `screen_gameplay`. If the saved fit used a different mask, its z-scoring will differ slightly.',
        '- `n_active_trs_saved_fit` is the retained test-TR count in the saved fit significance file. A mismatch with `n_active_trs_total` means the reconstructed mask differs from the fit-time mask.',
        '- BOLD is z-scored per run on Screen/Gameplay TRs, and attempts use only gameplay TRs.',
        '- Within-attempt split-half reliability uses odd versus even active TRs and Spearman-Brown correction.',
        '- Same-level similarity is the Fisher-averaged Pearson correlation between attempt-mean patterns, excluding pairs from the same repetition. The permutation shuffles level labels within run, preserving run composition.',
        '- The same-run columns repeat the contrast within run and are the stricter control. The pass rule uses the pooled test because same-run same-level pairs are sparse.',
        '- Same-level pairs are enriched for failed attempts and similar durations. This audit does not control outcome or attempt duration, so a positive result is not yet evidence for level-specific coding.',
        '- The mask is the group 4-of-5 partial-conjunction map, restricted to cortical vertices.',
        f'- Attempts with fewer than {MIN_ATTEMPT_TRS} gameplay TRs are excluded from reliability and similarity estimates.',
        '- UMAP and decoding are not run here. This audit only asks whether the attempt-level BOLD patterns are reliable enough to support them.',
        '',
        '## Level detail',
        '',
        frame_to_markdown(level_df),
    ]
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Wrote summary: %s', path)


def audit_subject(subject_cfg, group_mask, attempts, args, mask_name, rng,
                  logger):
    """Load one subject and return its summary and per-attempt tables."""
    subject = int(subject_cfg['subject'])
    subject_mask = load_subject_mask(subject_cfg, group_mask, logger)
    saved_active = load_saved_active_count(subject_cfg, logger)
    subject_active_total = 0
    subject_attempts = []
    subject_patterns = []
    subject_meta = []

    for session in subject_cfg['test_sessions']:
        session = int(session)
        session_attempts = attempts[
            (attempts['subject'] == subject) &
            (attempts['session'] == session)].copy()
        if session_attempts.empty:
            logger.warning('No attempts for sub-%02d ses-%03d',
                           subject, session)
            continue
        logger.info('Auditing sub-%02d ses-%03d (%d attempts)',
                    subject, session, len(session_attempts))
        metadata = load_practice_metadata(
            subject, session, PATHS['practice_phase_metadata'])
        Y, active_mask, screen_p = load_session_bold(
            subject, session, metadata, subject_mask,
            args.annotation_root, logger)
        subject_active_total += int(active_mask.sum())
        metrics, patterns, pattern_meta = extract_attempt_metrics(
            Y, active_mask, screen_p, session_attempts, logger)
        subject_attempts.append(metrics)
        if len(patterns):
            subject_patterns.append(patterns)
            subject_meta.append(pattern_meta)
        del Y

    if not subject_attempts:
        logger.warning('No attempt metrics for sub-%02d', subject)
        return None
    subject_attempt_df = pd.concat(subject_attempts, ignore_index=True)

    if subject_patterns:
        patterns = np.vstack(subject_patterns)
        pattern_meta = pd.concat(subject_meta, ignore_index=True)
    else:
        patterns = np.empty((0, int(subject_mask.sum())), dtype=np.float32)
        pattern_meta = pd.DataFrame()
    save_patterns(
        args.output_dir / 'patterns', subject, patterns, pattern_meta,
        np.flatnonzero(subject_mask), mask_name, logger)

    subject_row = summarize_subject(
        subject, subject_attempt_df, pattern_meta, patterns,
        args.n_permutations, rng, logger)
    subject_row['n_active_trs_total'] = subject_active_total
    subject_row['n_active_trs_saved_fit'] = saved_active
    if np.isfinite(saved_active):
        subject_row['screen_mask_matches_fit'] = bool(
            subject_active_total == saved_active)
    else:
        subject_row['screen_mask_matches_fit'] = np.nan
    if np.isfinite(saved_active) and subject_active_total != saved_active:
        logger.warning(
            'sub-%02d: reconstructed gameplay mask retains %d TRs, saved '
            'fit significance used %d; z-scoring may differ',
            subject, subject_active_total, saved_active)
    return subject_row, subject_attempt_df, summarize_levels(subject_attempt_df)


def run_audit(group, group_mask, attempts, args, mask_name, logger):
    """Run the audit for every subject in the group."""
    rng = np.random.default_rng(RANDOM_SEED)
    subject_rows = []
    attempt_tables = []
    level_tables = []
    for subject_cfg in group['subjects']:
        result = audit_subject(
            subject_cfg, group_mask, attempts, args, mask_name, rng, logger)
        if result is None:
            continue
        subject_row, attempt_df, level_df = result
        subject_rows.append(subject_row)
        attempt_tables.append(attempt_df)
        level_tables.append(level_df)

    subject_df = pd.DataFrame(subject_rows)
    attempt_df = pd.concat(attempt_tables, ignore_index=True)
    level_df = pd.concat(level_tables, ignore_index=True)
    return subject_df, attempt_df, level_df


def write_outputs(output_dir, group_name, mask_path, subject_df, attempt_df,
                  level_df, n_subjects, logger):
    """Write the audit tables, Markdown summary and completion log."""
    subject_df.to_csv(output_dir / 'subject_reliability_summary.csv',
                      index=False)
    level_df.to_csv(output_dir / 'level_reliability_summary.csv', index=False)
    attempt_df.to_csv(output_dir / 'attempt_reliability.csv', index=False)

    write_summary_markdown(
        output_dir / 'audit_summary.md', group_name, mask_path,
        subject_df, level_df, n_subjects, logger)

    logger.info('Subjects passing: %d / %d',
                int(subject_df['subject_pass'].sum()), len(subject_df))
    logger.info('Done.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage B audit of held-out BOLD reliability within attempts. '
                    'Writes tables and a summary; see plot_pme_reliability.py '
                    'for figures.')
    parser.add_argument('--group', default=DEFAULT_GROUP,
                        choices=sorted(GROUP_REGISTRY))
    parser.add_argument('--attempt-table', type=Path, default=STAGE_A_TABLE)
    parser.add_argument(
        '--mask', type=Path, default=None,
        help='Group 4-of-5 dscalar mask; defaults to group_r_full_sig_conj_min4.')
    parser.add_argument(
        '--annotation-root', type=Path,
        default=PATHS['screen_gameplay_annotations'],
        help='Root of the validated Screen/Gameplay annotated-event files.')
    parser.add_argument(
        '--output-dir', type=Path,
        default=PATHS['variance_partitioning'].parent / 'pme_reliability')
    parser.add_argument('--n-permutations', type=int, default=N_PERMUTATIONS)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)

    group = GROUP_REGISTRY[args.group]
    mask_path = (args.mask if args.mask is not None else
                 PATHS['figures'] / 'ciftis' / group['output_label'] /
                 DEFAULT_MASK_NAME)
    logger.info('Group: %s', args.group)
    logger.info('Output: %s', args.output_dir)
    logger.info('Mask: %s', mask_path)

    group_mask = load_mask(mask_path, logger)
    attempts = load_attempt_table(args.attempt_table, group, logger)

    subject_df, attempt_df, level_df = run_audit(
        group, group_mask, attempts, args, mask_path.name, logger)
    write_outputs(args.output_dir, args.group, mask_path, subject_df,
                  attempt_df, level_df, len(group['subjects']), logger)


if __name__ == '__main__':
    main()
