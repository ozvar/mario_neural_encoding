"""Stage A behavioural audit of level-attempt epochs for the PMe analysis.

Builds one row per attempt from the combined repvars TSVs and writes repetition,
duration and outcome tables plus a Markdown summary. No fMRI is loaded. Figures
are produced by plot_pme_audit.py.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mario_encoding.config import FEATURE_SPACES, PARAMETERS, PATHS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY
from mario_encoding.utils.data_loading import load_practice_metadata

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}

TR = float(PARAMETERS['TR'])
FRAME_RATE = float(PARAMETERS['frame_rate'])
DEATH_PAUSE_S = (PARAMETERS['event_frame_duration_dict']['event_hit_life_lost']
                 / FRAME_RATE)
ACTIVITY_THRESHOLD = 0.01
FEATURE_COLS = FEATURE_SPACES['motor'] + FEATURE_SPACES['scene']

# Provisional decision rule, fixed before inspecting the counts.
MIN_SUBJECTS_PASS = 4
MIN_LEVELS_GE2 = 10
MIN_LEVELS_GE3 = 5
MIN_MEDIAN_TRS = 8


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


def find_run_file(directory, subject, session, run, suffix):
    path = (Path(directory) / f'sub-{subject:02d}' / f'ses-{session:03d}' /
            f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_{suffix}.tsv')
    return path


def load_downsampled(run_path, logger):
    """Load a downsampled TR file and return activity metadata."""
    df = pd.read_csv(run_path, sep='\t')
    available = [col for col in FEATURE_COLS if col in df.columns]
    missing = sorted(set(FEATURE_COLS) - set(available))
    if missing:
        logger.warning('%s: %d configured features missing from downsampled file',
                       run_path.name, len(missing))

    if available:
        activity = np.abs(df[available].to_numpy(dtype=float)).sum(axis=1)
        active_proxy = activity > ACTIVITY_THRESHOLD
    else:
        active_proxy = np.zeros(len(df), dtype=bool)

    result = {
        'n_trs': len(df),
        'active_proxy': active_proxy,
        'screen_available': 'screen_gameplay' in df.columns,
        'active_screen': None,
    }
    if result['screen_available']:
        result['active_screen'] = df['screen_gameplay'].to_numpy(dtype=float) > 0.0
    return result


def classify_final_outcome(n_trs, is_last_rep, phase):
    """Classify the post-death or no-death segment of a level presentation."""
    if phase != 'practice':
        return 'ambiguous', 'not_practice'
    if n_trs < 2:
        return 'ambiguous', 'short_epoch'
    if is_last_rep:
        return 'ambiguous', 'end_of_run'
    return 'completed', 'level_complete'


def count_outcomes(group):
    """Return completed, failed and ambiguous counts for a group of attempts."""
    outcomes = group['outcome'].value_counts()
    return {outcome: int(outcomes.get(outcome, 0))
            for outcome in ('completed', 'failed', 'ambiguous')}


def get_life_loss_times(group):
    """Return sorted raw life-loss times within one level presentation."""
    life_mask = group['event_hit_life_lost'].to_numpy(dtype=float) > 0
    return np.sort(group.loc[life_mask, 'frame_time_in_run'].unique())


def load_run_inputs(subject, session, run, metadata, run_index,
                    combined_dir, downsampled_dir, logger):
    """Load and validate one run's frame and downsampled tables."""
    combined_path = find_run_file(
        combined_dir, subject, session, run, 'desc-combined')
    downsampled_path = find_run_file(
        downsampled_dir, subject, session, run, 'desc-downsampled')
    if not combined_path.exists():
        logger.warning('Missing combined file: %s', combined_path)
        return None
    if not downsampled_path.exists():
        logger.warning('Missing downsampled file: %s', downsampled_path)
        return None

    df = pd.read_csv(combined_path, sep='\t')
    required = {'rep_id', 'rep_onset', 'rep_duration', 'level',
                'frame_time_in_run', 'frame_index'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'{combined_path.name} missing columns: {missing}')
    if 'event_hit_life_lost' not in df.columns:
        logger.warning('%s: no event_hit_life_lost column; using zeros',
                       combined_path.name)
        df['event_hit_life_lost'] = 0

    down = load_downsampled(downsampled_path, logger)
    if down['n_trs'] != metadata['n_trs_per_run'][run_index]:
        logger.warning(
            'sub-%02d ses-%03d run-%02d: metadata n_trs=%d, file n_trs=%d',
            subject, session, run, metadata['n_trs_per_run'][run_index],
            down['n_trs'])

    if 'phase' in df.columns:
        phases = sorted(df['phase'].dropna().astype(str).unique())
        if phases != ['practice']:
            logger.warning('%s: unexpected phases %s',
                           combined_path.name, phases)

    df['TR_bin'] = np.floor(df['frame_time_in_run'] / TR).astype(int)
    last_onset = df.groupby('rep_id')['rep_onset'].first().max()
    return df, down, last_onset


def segment_presentation(rep_onset, rep_end, life_times):
    """Split one level presentation into attempts at life-loss events."""
    segments = []
    segment_start = rep_onset
    for life_time in life_times:
        if life_time > segment_start + (1.0 / FRAME_RATE):
            segments.append((segment_start, float(life_time),
                             'failed', 'life_loss', 1))
        segment_start = float(life_time) + DEATH_PAUSE_S
    if segment_start < rep_end - (1.0 / FRAME_RATE):
        segments.append((segment_start, rep_end,
                         'final', 'level_end', 0))
    if not segments:
        segments.append((rep_onset, rep_end,
                         'final', 'level_end', 0))
    return segments


def build_attempt_row(subject, session, run, rep_id, group, segment,
                      down, run_offset, rep_onset, rep_duration,
                      is_last_rep, phase, attempt_index):
    """Build one attempt row from one segment of a level presentation."""
    start_s, end_s, kind, detail, life_count = segment
    if kind == 'failed':
        mask = ((group['frame_time_in_run'] >= start_s) &
                (group['frame_time_in_run'] <= end_s))
    else:
        mask = ((group['frame_time_in_run'] >= start_s) &
                (group['frame_time_in_run'] < end_s))
    seg = group.loc[mask]
    if seg.empty:
        return None

    start_tr = int(seg['TR_bin'].min())
    end_tr = int(seg['TR_bin'].max())
    n_trs = end_tr - start_tr + 1
    active = down['active_proxy'][start_tr:end_tr + 1]
    n_active_proxy = int(active.sum())
    active_fraction_proxy = (
        n_active_proxy / n_trs if n_trs > 0 else np.nan)

    n_active_screen = np.nan
    active_fraction_screen = np.nan
    if down['screen_available']:
        screen = down['active_screen'][start_tr:end_tr + 1]
        n_active_screen = int(screen.sum())
        active_fraction_screen = (
            n_active_screen / n_trs if n_trs > 0 else np.nan)

    if kind == 'failed':
        outcome, outcome_detail = 'failed', detail
        life_loss_to_end_s = 0.0
    else:
        outcome, outcome_detail = classify_final_outcome(
            n_trs, is_last_rep, phase)
        life_loss_to_end_s = np.nan

    return {
        'subject': subject,
        'session': session,
        'run': run,
        'rep_id': rep_id,
        'attempt_id': f'{rep_id}_att-{attempt_index:02d}',
        'attempt_index': attempt_index,
        'level': str(group['level'].iloc[0]),
        'world': (str(group['world'].iloc[0])
                  if 'world' in group else ''),
        'stage': (str(group['stage'].iloc[0])
                  if 'stage' in group else ''),
        'phase': phase,
        'rep_onset_s': rep_onset,
        'rep_duration_s': rep_duration,
        'attempt_start_s': start_s,
        'attempt_end_s': end_s,
        'start_tr_in_run': start_tr,
        'end_tr_in_run': end_tr,
        'n_trs': n_trs,
        'n_frames': int(len(seg)),
        'n_active_trs_proxy': n_active_proxy,
        'active_fraction_proxy': active_fraction_proxy,
        'screen_gameplay_available': down['screen_available'],
        'n_active_trs_screen': n_active_screen,
        'active_fraction_screen': active_fraction_screen,
        'life_loss_count': life_count,
        'life_loss_to_end_s': life_loss_to_end_s,
        'is_last_rep_in_run': is_last_rep,
        'outcome': outcome,
        'outcome_detail': outcome_detail,
        'session_tr_start': int(run_offset) + start_tr,
        'session_tr_end': int(run_offset) + end_tr,
    }


def build_attempt_table(subject, session, metadata, combined_dir,
                        downsampled_dir, logger):
    """Build one row per attempt, splitting level presentations at life loss."""
    rows = []
    run_offsets = np.concatenate([[0], np.cumsum(metadata['n_trs_per_run'])])
    any_screen = False

    for run_index, run in enumerate(metadata['practice_runs']):
        run_inputs = load_run_inputs(
            subject, session, run, metadata, run_index,
            combined_dir, downsampled_dir, logger)
        if run_inputs is None:
            continue
        df, down, last_onset = run_inputs
        any_screen = any_screen or down['screen_available']

        for rep_id, group in df.groupby('rep_id', sort=False):
            group = group.sort_values('frame_index').reset_index(drop=True)
            rep_onset = float(group['rep_onset'].iloc[0])
            rep_duration = float(group['rep_duration'].iloc[0])
            if not np.isfinite(rep_onset) or not np.isfinite(rep_duration):
                logger.warning(
                    'sub-%02d ses-%03d run-%02d: skipping %s with missing timing',
                    subject, session, run, rep_id)
                continue
            rep_end = rep_onset + rep_duration
            phase = str(group['phase'].iloc[0]) if 'phase' in group else ''
            is_last_rep = bool(np.isclose(rep_onset, last_onset))
            life_times = get_life_loss_times(group)
            segments = segment_presentation(rep_onset, rep_end, life_times)

            rep_rows = []
            for attempt_index, segment in enumerate(segments, start=1):
                row = build_attempt_row(
                    subject, session, run, rep_id, group, segment, down,
                    run_offsets[run_index], rep_onset, rep_duration,
                    is_last_rep, phase, attempt_index)
                if row is not None:
                    rep_rows.append(row)
            for rep_row in rep_rows:
                rep_row['n_attempts_in_rep'] = len(rep_rows)
            rows.extend(rep_rows)

    if not any_screen and rows:
        logger.warning(
            'Screen/Gameplay metadata absent from downsampled files; '
            'active-TR counts use the feature-activity proxy only.')
    return pd.DataFrame(rows)


def summarize_levels(attempt_df):
    """One row per subject and level."""
    if attempt_df.empty:
        return pd.DataFrame()
    rows = []
    for (subject, level), group in attempt_df.groupby(['subject', 'level']):
        counts = count_outcomes(group)
        rows.append({
            'subject': subject,
            'level': level,
            'n_attempts': len(group),
            'n_completed': counts['completed'],
            'n_failed': counts['failed'],
            'n_ambiguous': counts['ambiguous'],
            'median_n_trs': safe_median(group['n_trs']),
            'min_n_trs': group['n_trs'].min(),
            'max_n_trs': group['n_trs'].max(),
            'median_active_fraction_proxy':
                safe_median(group['active_fraction_proxy']),
            'median_active_fraction_screen':
                safe_median(group['active_fraction_screen']),
        })
    return pd.DataFrame(rows)


def summarize_subjects(attempt_df, level_df):
    """One row per subject with repetition and duration counts."""
    if attempt_df.empty:
        return pd.DataFrame()
    rows = []
    for subject, group in attempt_df.groupby('subject'):
        levels = level_df[level_df['subject'] == subject]
        counts = levels['n_attempts'].to_numpy()
        outcome_counts = count_outcomes(group)
        rows.append({
            'subject': subject,
            'n_attempts': len(group),
            'n_levels': len(levels),
            'n_levels_ge1': int((counts >= 1).sum()),
            'n_levels_ge2': int((counts >= 2).sum()),
            'n_levels_ge3': int((counts >= 3).sum()),
            'n_levels_ge4': int((counts >= 4).sum()),
            'n_levels_ge5': int((counts >= 5).sum()),
            'max_attempts_per_level': int(counts.max()) if len(counts) else 0,
            'median_n_trs': safe_median(group['n_trs']),
            'median_active_fraction_proxy':
                safe_median(group['active_fraction_proxy']),
            'median_active_fraction_screen':
                safe_median(group['active_fraction_screen']),
            'n_attempts_ge8tr': int((group['n_trs'] >= 8).sum()),
            'n_attempts_ge10tr': int((group['n_trs'] >= 10).sum()),
            'n_attempts_ge15tr': int((group['n_trs'] >= 15).sum()),
            'n_completed': outcome_counts['completed'],
            'n_failed': outcome_counts['failed'],
            'n_ambiguous': outcome_counts['ambiguous'],
            'screen_gameplay_available':
                bool(group['screen_gameplay_available'].any()),
        })
    return pd.DataFrame(rows)


def select_passing_subjects(subject_df):
    """Return subjects meeting the Stage A repetition and duration rule."""
    passed = []
    for _, row in subject_df.iterrows():
        if (row['n_levels_ge2'] >= MIN_LEVELS_GE2 and
                row['n_levels_ge3'] >= MIN_LEVELS_GE3 and
                row['median_n_trs'] >= MIN_MEDIAN_TRS):
            passed.append(int(row['subject']))
    return passed


def write_summary_markdown(path, group_name, subject_df, level_df, logger):
    passed = select_passing_subjects(subject_df)
    required_subjects = min(MIN_SUBJECTS_PASS, len(subject_df))
    decision = 'PASS' if len(passed) >= required_subjects else 'FAIL'

    lines = [
        '# PMe repetition audit, Stage A',
        '',
        f'Group: `{group_name}`. Generated from combined repvars TSVs.',
        '',
        '## Decision rule',
        '',
        f'Provisional rule fixed before inspection: at least {required_subjects} '
        f'of {len(subject_df)} subjects with at least {MIN_LEVELS_GE2} levels '
        f'having 2 or more attempts, at least {MIN_LEVELS_GE3} levels having '
        f'3 or more attempts, and a median attempt duration of at least '
        f'{MIN_MEDIAN_TRS} TRs.',
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
        '- Outcome `failed` uses a raw life-loss event near the end of the attempt.',
        '- Outcome `ambiguous` covers end-of-run attempts, short epochs and life loss far from the end.',
        '- `active_fraction_proxy` uses the legacy feature-activity mask because the saved downsampled files do not contain `screen_gameplay`.',
        '- If `screen_gameplay_available` is False, the true fit-time retained-TR count still needs the regenerated screen metadata.',
        '',
        '## Level detail',
        '',
        frame_to_markdown(level_df),
    ]
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Wrote summary: %s', path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage A behavioural audit of level-attempt epochs. '
                    'Writes tables and a summary; see plot_pme_audit.py '
                    'for figures.')
    parser.add_argument('--group', default='GROUP_N5_PRIMARY',
                        choices=sorted(GROUP_REGISTRY))
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PATHS['variance_partitioning'].parent / 'pme_audit')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)

    group = GROUP_REGISTRY[args.group]
    logger.info('Group: %s', args.group)
    logger.info('Output: %s', args.output_dir)

    tables = []
    for subject_cfg in group['subjects']:
        subject = int(subject_cfg['subject'])
        for session in subject_cfg['test_sessions']:
            session = int(session)
            logger.info('Auditing sub-%02d ses-%03d', subject, session)
            metadata = load_practice_metadata(
                subject, session, PATHS['practice_phase_metadata'])
            table = build_attempt_table(
                subject, session, metadata,
                PATHS['per_run_combined_repvars_and_bids_events'],
                PATHS['per_run_downsampled_to_TR'], logger)
            tables.append(table)

    attempt_df = pd.concat(tables, ignore_index=True)
    level_df = summarize_levels(attempt_df)
    subject_df = summarize_subjects(attempt_df, level_df)

    attempt_df.to_csv(args.output_dir / 'attempt_table.csv', index=False)
    level_df.to_csv(args.output_dir / 'level_repetition_summary.csv', index=False)
    subject_df.to_csv(args.output_dir / 'subject_repetition_summary.csv',
                      index=False)

    write_summary_markdown(
        args.output_dir / 'audit_summary.md', args.group,
        subject_df, level_df, logger)

    logger.info('Attempts: %d', len(attempt_df))
    logger.info('Subjects: %d', subject_df['subject'].nunique())
    logger.info('Done.')


if __name__ == '__main__':
    main()
