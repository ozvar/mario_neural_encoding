"""Old/new mask review report for the Screen/Gameplay refit.

Compares the legacy feature-activity mask with the new Screen/Gameplay mask on
identical downsampled TR rows. CPU only, no fMRI. Writes tables and a Markdown
summary under results/mask_review/.

Usage:
    python scripts/python/report_screen_gameplay_mask.py --group GROUP_N5_PRIMARY
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY
from mario_encoding.utils.data_loading import (
    compute_baseline_mask,
    load_practice_features_with_names,
    load_practice_metadata,
    select_and_validate_features,
)
from mario_encoding.utils.screen_annotations import (
    SCREEN_FRACTION_COLUMNS,
    SCREEN_LABELS,
    SCREEN_METADATA_COLUMNS,
    compute_screen_gameplay_mask,
    gameplay_segments,
    load_annotation_provenance,
    validate_screen_metadata,
)

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}

ZERO_VARIANCE_TOL = 1e-10
P_BANDS = [
    ('(0, 0.05]', 0.0, 0.05),
    ('(0.05, 0.1]', 0.05, 0.1),
    ('(0.1, 0.25]', 0.1, 0.25),
    ('(0.25, 0.5]', 0.25, 0.5),
    ('(0.5, 0.75]', 0.5, 0.75),
    ('(0.75, 1)', 0.75, 1.0),
]


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(output_dir / 'report.log'),
                  logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def load_features_and_metadata(subject, sessions, logger):
    """Load selected features, screen metadata and run structure without fMRI."""
    X_list = []
    feature_names = None
    metadata_chunks = {col: [] for col in SCREEN_METADATA_COLUMNS}
    run_onsets_list = [0]
    session_onsets_list = [0]
    run_sessions = []
    cumulative = 0

    for session in sessions:
        meta = load_practice_metadata(
            subject, session, PATHS['practice_phase_metadata'])
        X_sess, feat_names, extra = load_practice_features_with_names(
            subject, session, meta['practice_runs'],
            PATHS['per_run_downsampled_to_TR'],
            metadata_columns=SCREEN_METADATA_COLUMNS)
        if feature_names is None:
            feature_names = feat_names
        elif feat_names != feature_names:
            raise ValueError(
                f'Feature order differs in sub-{subject:02d} ses-{session:03d}')
        X_list.append(X_sess)
        for col in SCREEN_METADATA_COLUMNS:
            metadata_chunks[col].append(extra[col])

        run_onsets_sess = np.array(meta['run_onsets']) + cumulative
        if cumulative == 0:
            run_onsets_list.extend(run_onsets_sess[1:].tolist())
        else:
            run_onsets_list.extend(run_onsets_sess.tolist())
        run_sessions.extend([session] * len(meta['practice_runs']))
        cumulative += meta['n_samples']
        session_onsets_list.append(cumulative)

    X = np.vstack(X_list)
    metadata = {
        col: np.concatenate(chunks) for col, chunks in metadata_chunks.items()
    }
    return {
        'X': X,
        'feature_names': feature_names,
        'metadata': metadata,
        'run_onsets': np.array(run_onsets_list),
        'session_onsets': np.array(session_onsets_list[:-1]),
        'run_sessions': run_sessions,
    }


def screen_categories(metadata):
    """Dominant screen label per TR, or no_frames when nothing was annotated."""
    fractions = np.column_stack([metadata[col] for col in SCREEN_FRACTION_COLUMNS])
    n_frames = np.asarray(metadata['screen_n_frames'], dtype=float)
    index = np.argmax(fractions, axis=1)
    labels = np.array(SCREEN_LABELS, dtype=object)[index]
    labels[n_frames == 0] = 'no_frames'
    return labels


def p_band(p):
    if p == 0:
        return '0'
    if p == 1:
        return '1'
    for label, low, high in P_BANDS:
        if low < p <= high:
            return label
    return 'other'


def run_counts(mask_old, mask_new, run_onsets):
    rows = []
    ends = list(run_onsets[1:]) + [len(mask_new)]
    for run_index, (start, end) in enumerate(zip(run_onsets, ends)):
        old = mask_old[start:end].astype(bool)
        new = mask_new[start:end].astype(bool)
        rows.append({
            'run_index': run_index,
            'n_total': int(end - start),
            'n_old': int(old.sum()),
            'n_new': int(new.sum()),
            'n_both': int(np.sum(old & new)),
            'n_old_only': int(np.sum(old & ~new)),
            'n_new_only': int(np.sum(~old & new)),
            'n_neither': int(np.sum(~old & ~new)),
        })
    return rows


def segment_summary(mask_new, run_onsets):
    n_segments = 0
    lengths = []
    gap_pairs = 0
    contiguous_pairs = 0
    ends = list(run_onsets[1:]) + [len(mask_new)]
    for start, end in zip(run_onsets, ends):
        run_mask = mask_new[start:end].astype(bool)
        segments, bounds = gameplay_segments(run_mask, np.array([0]))
        n_segments += len(bounds)
        lengths.extend((bounds[:, 3] - bounds[:, 2]).tolist() if len(bounds) else [])
        retained = np.flatnonzero(run_mask)
        if len(retained) > 1:
            diffs = np.diff(retained)
            contiguous_pairs += int(np.sum(diffs == 1))
            gap_pairs += int(np.sum(diffs > 1))
    return {
        'n_segments': n_segments,
        'mean_length': float(np.mean(lengths)) if lengths else 0.0,
        'median_length': float(np.median(lengths)) if lengths else 0.0,
        'max_length': int(np.max(lengths)) if lengths else 0,
        'n_adjacent_pairs': contiguous_pairs + gap_pairs,
        'n_contiguous_pairs': contiguous_pairs,
        'n_gap_pairs': gap_pairs,
    }


def analyse_subject(subject, group_cfg, logger):
    subject_cfg = next(s for s in group_cfg['subjects'] if s['subject'] == subject)
    splits = {
        'train': subject_cfg['train_sessions'],
        'test': subject_cfg['test_sessions'],
    }
    mask_cfg = PARAMETERS['screen_gameplay_mask']
    per_run = []
    confusion = []
    p_distribution = []
    variance_rows = []
    variance_summary = []
    segments = []
    retained_session = []
    retained_fold = []

    for split, sessions in splits.items():
        data = load_features_and_metadata(subject, sessions, logger)
        X, names, _ = select_and_validate_features(
            data['X'], data['feature_names'], FEATURE_SPACES, logger)
        metadata = data['metadata']
        validate_screen_metadata(pd.DataFrame(metadata))

        old_mask = compute_baseline_mask(X, data['run_onsets'], logger=logger)
        new_mask = compute_screen_gameplay_mask(
            metadata['screen_gameplay'], rule=mask_cfg['rule'],
            threshold=mask_cfg['threshold'], logger=logger)

        for row in run_counts(old_mask, new_mask, data['run_onsets']):
            row.update({'subject': subject, 'split': split,
                        'session': data['run_sessions'][row['run_index']]})
            per_run.append(row)

        categories = screen_categories(metadata)
        p = np.asarray(metadata['screen_gameplay'], dtype=float)
        changes = np.where(old_mask & ~new_mask, 'old_only',
                           np.where(~old_mask & new_mask, 'new_only',
                                    np.where(old_mask & new_mask, 'both',
                                             'neither')))
        frame = pd.DataFrame({
            'change': changes, 'category': categories, 'p': p,
        })
        counts = frame.groupby(['change', 'category']).size().reset_index(name='n')
        for row in counts.to_dict('records'):
            row.update({'subject': subject, 'split': split})
            confusion.append(row)

        bands = np.array([p_band(value) for value in p], dtype=object)
        no_frames = np.asarray(metadata['screen_n_frames']) == 0
        frame_p = pd.DataFrame({'band': bands, 'p': p, 'no_frames': no_frames})
        frame_p.loc[frame_p['p'] == 0, 'band'] = np.where(
            frame_p.loc[frame_p['p'] == 0, 'no_frames'], '0_no_frames', '0_non_gameplay')
        for row in frame_p.groupby('band').size().reset_index(name='n').to_dict('records'):
            row.update({'subject': subject, 'split': split})
            p_distribution.append(row)

        X_var = np.nan_to_num(X, copy=True)
        old_var = X_var[old_mask].var(axis=0) if old_mask.sum() > 1 else np.full(X.shape[1], np.nan)
        new_var = X_var[new_mask].var(axis=0) if new_mask.sum() > 1 else np.full(X.shape[1], np.nan)
        newly_zero = []
        for name, v_old, v_new in zip(names, old_var, new_var):
            variance_rows.append({
                'subject': subject, 'split': split, 'feature': name,
                'var_old': float(v_old), 'var_new': float(v_new),
                'zero_old': bool(v_old <= ZERO_VARIANCE_TOL),
                'zero_new': bool(v_new <= ZERO_VARIANCE_TOL),
            })
            if v_old > ZERO_VARIANCE_TOL and v_new <= ZERO_VARIANCE_TOL:
                newly_zero.append(name)
        variance_summary.append({
            'subject': subject, 'split': split,
            'n_features': len(names),
            'n_zero_old': int(np.sum(old_var <= ZERO_VARIANCE_TOL)),
            'n_zero_new': int(np.sum(new_var <= ZERO_VARIANCE_TOL)),
            'n_newly_zero': len(newly_zero),
            'newly_zero': ';'.join(newly_zero),
        })

        segment = segment_summary(new_mask, data['run_onsets'])
        segment.update({'subject': subject, 'split': split})
        segments.append(segment)

        for session_index, session in enumerate(sessions):
            if split == 'train':
                start = data['session_onsets'][session_index]
                end = (data['session_onsets'][session_index + 1]
                       if session_index + 1 < len(data['session_onsets']) else len(X))
            else:
                start, end = (data['session_onsets'][session_index],
                              data['session_onsets'][session_index + 1]
                              if session_index + 1 < len(data['session_onsets']) else len(X))
            retained_session.append({
                'subject': subject, 'split': split, 'session': session,
                'n_total': int(end - start),
                'n_old': int(old_mask[start:end].sum()),
                'n_new': int(new_mask[start:end].sum()),
            })

        if split == 'train':
            for session_index, session in enumerate(sessions):
                start = data['session_onsets'][session_index]
                end = (data['session_onsets'][session_index + 1]
                       if session_index + 1 < len(data['session_onsets']) else len(X))
                retained_fold.append({
                    'subject': subject, 'held_out_session': session,
                    'n_train_total': int(len(X) - (end - start)),
                    'n_train_old': int(old_mask.sum() - old_mask[start:end].sum()),
                    'n_train_new': int(new_mask.sum() - new_mask[start:end].sum()),
                    'n_test_new': int(new_mask.sum()),
                })

    return {
        'per_run': per_run,
        'confusion': confusion,
        'p_distribution': p_distribution,
        'variance': variance_rows,
        'variance_summary': variance_summary,
        'segments': segments,
        'retained_session': retained_session,
        'retained_fold': retained_fold,
    }


def coverage_tables(group_cfg):
    path = Path(PATHS['per_run_combined_repvars_and_bids_events']) / 'screen_annotation_coverage.tsv'
    if not path.exists():
        raise FileNotFoundError(
            f'Screen annotation coverage not found: {path}. Run step 2 first.')
    coverage = pd.read_csv(path, sep='\t')
    subjects = {s['subject'] for s in group_cfg['subjects']}
    coverage = coverage[coverage['subject'].isin(subjects)]

    expected = []
    sessions_used = set()
    for subject_cfg in group_cfg['subjects']:
        subject = subject_cfg['subject']
        for session in subject_cfg['train_sessions'] + subject_cfg['test_sessions']:
            sessions_used.add((subject, session))
            meta = load_practice_metadata(
                subject, session, PATHS['practice_phase_metadata'])
            for run in meta['practice_runs']:
                expected.append((subject, session, run))
    expected = set(expected)
    coverage_all = coverage[
        coverage.apply(lambda row: (row['subject'], row['session']) in sessions_used, axis=1)
    ].copy()
    actual_all = set(zip(coverage_all['subject'], coverage_all['session'], coverage_all['run']))
    coverage = coverage_all[
        coverage_all.apply(
            lambda row: (row['subject'], row['session'], row['run']) in expected, axis=1)
    ].copy()
    missing = sorted(expected - actual_all)
    extra = sorted(actual_all - expected)
    failed = coverage[coverage['status'] != 'success']
    summary = {
        'coverage_file': str(path),
        'n_expected_runs': len(expected),
        'n_coverage_rows': len(coverage),
        'n_coverage_rows_all_sessions': len(coverage_all),
        'n_missing_runs': len(missing),
        'n_extra_runs': len(extra),
        'n_failed_runs': int(len(failed)),
        'missing_runs': missing[:20],
        'extra_runs': extra[:20],
        'failed_runs': failed[['subject', 'session', 'run', 'status']].head(20).to_dict('records'),
        'max_onset_delta_s': float(coverage['max_onset_delta_s'].max())
        if 'max_onset_delta_s' in coverage.columns and len(coverage) else None,
        'n_terminal_frames_filled': int(coverage['n_terminal_frames_filled'].sum())
        if 'n_terminal_frames_filled' in coverage.columns else None,
    }
    return coverage, summary


def md_table(table):
    """Render a DataFrame as a fenced plain-text table."""
    return '```text\n' + table.to_string(index=False) + '\n```'


def write_markdown(path, group_name, summary, tables):
    lines = [
        f'# Screen/Gameplay mask review: {group_name}',
        '',
        f'Generated {summary["generated_at_utc"]}.',
        '',
        '## Annotation provenance',
        f'- Annotation root: `{summary["annotation_root"]}`',
        f'- mario commit: `{summary["mario_commit"]}`',
        f'- videogames_utils commit: `{summary["videogames_utils_commit"]}`',
        f'- Coverage: {summary["coverage"]["n_coverage_rows"]} rows for '
        f'{summary["coverage"]["n_expected_runs"]} expected runs; '
        f'{summary["coverage"]["n_missing_runs"]} missing, '
        f'{summary["coverage"]["n_failed_runs"]} failed; '
        f'{summary["coverage"]["n_terminal_frames_filled"]} terminal frames inherited.',
        '',
        '',
        '## Retained TRs by subject and split',
        md_table(tables['retained_split']),
        '',
        '## Old/new confusion',
        md_table(tables['confusion']),
        '',
        '## Feature variance audit',
        md_table(tables['variance_summary']),
        '',
        '## Contiguous gameplay segments',
        md_table(tables['segments']),
        '',
        'The old mask is the legacy feature-activity definition (sum of absolute '
        'feature values above 0.01). The new mask is the configured Screen/Gameplay '
        'rule, p > 0. Feature variance uses NaN-to-zero features, as in the fit.',
        '',
    ]
    reconstruction = summary.get('bk2_reconstruction')
    if reconstruction:
        lines.extend([
            '## Reconstruction note',
            '```json',
            json.dumps(reconstruction, indent=2),
            '```',
            '',
        ])
    Path(path).write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=list(GROUP_REGISTRY))
    parser.add_argument('--output-dir', default=None,
                        help='Default: results/mask_review/{group}')
    args = parser.parse_args()

    group_cfg = GROUP_REGISTRY[args.group]
    output_dir = Path(args.output_dir) if args.output_dir else (
        PATHS['variance_partitioning'].parent / 'mask_review' / args.group.lower())
    logger = setup_logging(output_dir)
    logger.info(f'Building mask review for {args.group}')
    logger.info(f'Output directory: {output_dir}')

    provenance = load_annotation_provenance(PATHS['per_run_downsampled_to_TR'])
    coverage, coverage_summary = coverage_tables(group_cfg)

    all_results = {key: [] for key in (
        'per_run', 'confusion', 'p_distribution', 'variance',
        'variance_summary', 'segments', 'retained_session', 'retained_fold')}
    for subject_cfg in group_cfg['subjects']:
        subject = subject_cfg['subject']
        logger.info(f'Analysing sub-{subject:02d}')
        result = analyse_subject(subject, group_cfg, logger)
        for key in all_results:
            all_results[key].extend(result[key])

    tables = {
        'per_run': pd.DataFrame(all_results['per_run']),
        'confusion': pd.DataFrame(all_results['confusion']),
        'p_distribution': pd.DataFrame(all_results['p_distribution']),
        'variance': pd.DataFrame(all_results['variance']),
        'variance_summary': pd.DataFrame(all_results['variance_summary']),
        'segments': pd.DataFrame(all_results['segments']),
        'retained_session': pd.DataFrame(all_results['retained_session']),
        'retained_fold': pd.DataFrame(all_results['retained_fold']),
    }
    retained_split = (
        tables['per_run'].groupby(['subject', 'split'])[
            ['n_total', 'n_old', 'n_new', 'n_old_only', 'n_new_only']
        ].sum().reset_index()
    )
    tables['retained_split'] = retained_split

    for name, table in tables.items():
        table.to_csv(output_dir / f'{name}.csv', index=False)

    source = provenance.get('source', {})
    summary = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'group': args.group,
        'annotation_root': provenance.get(
            'annotation_root', str(PATHS['screen_gameplay_annotations'])),
        'mario_commit': source.get('commit'),
        'videogames_utils_commit': source.get('videogames_utils_commit'),
        'mask': PARAMETERS['screen_gameplay_mask'],
        'coverage': coverage_summary,
        'n_trs': {key: int(len(table)) for key, table in tables.items()},
        'bk2_reconstruction': provenance.get('bk2_reconstruction'),
    }
    if provenance.get('source', {}).get('dirty'):
        summary['provenance_warning'] = 'annotation checkout was dirty at generation'
    with open(output_dir / 'report_summary.json', 'w') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    write_markdown(output_dir / 'report_summary.md', args.group, summary, tables)

    logger.info('Report complete')
    logger.info(f'  Coverage: {coverage_summary["n_missing_runs"]} missing, '
                f'{coverage_summary["n_failed_runs"]} failed runs')
    logger.info(f'  Retained TRs by split:\n{retained_split.to_string(index=False)}')
    logger.info(f'  Newly zero-variance features: '
                f'{int(tables["variance_summary"]["n_newly_zero"].sum())}')


if __name__ == '__main__':
    main()
