"""
Merge BIDS event annotations with replay variables to create run-level combined TSVs.

This script takes session-level replay variable TSVs and run-level BIDS event annotations,
merging them into comprehensive run-level dataframes with event markers and scene labels.
"""
import argparse

import pandas as pd
import numpy as np
from pathlib import Path
from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY
from mario_encoding.utils.screen_annotations import (
    ScreenAnnotationError,
    carry_provenance,
    create_screen_columns,
    load_or_collect_annotation_provenance,
)

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}



# Canonical set of all possible BIDS event types
# Ensures consistent column structure across all runs regardless of which events occurred
CANONICAL_EVENT_TYPES = [
    'Brick_smashed',
    'Coin_collected',
    'Hit/life_lost',
    'Hit/powerup_lost',
    'Powerup_collected'
]


def get_annotated_events_path(subject, session, run, events_path=PATHS['bids_annotated_tsvs']):
    """Get annotated events TSV file path."""
    base = events_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func'
    return base / f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run:02d}_desc-annotated_events.tsv'


def find_annotated_events(subject=None, session=None, run=None, events_path=PATHS['bids_annotated_tsvs']):
    """Find all annotated events TSV files matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'run-{run:02d}' if run else 'run-*'
    pattern = f'{sub_pattern}/{ses_pattern}/func/*{run_pattern}_desc-annotated_events.tsv'
    return list(events_path.glob(pattern))


def get_screen_annotations_path(subject, session, run,
                                screen_annotations_root=PATHS['screen_gameplay_annotations']):
    """Get the validated Screen/Gameplay annotation path for one run."""
    base = Path(screen_annotations_root) / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func'
    return base / f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run:02d}_desc-annotated_events.tsv'


def find_screen_annotations(subject=None, session=None, run=None,
                            screen_annotations_root=PATHS['screen_gameplay_annotations']):
    """Find validated Screen/Gameplay annotation files matching criteria."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'run-{run:02d}' if run else 'run-*'
    pattern = f'{sub_pattern}/{ses_pattern}/func/*{run_pattern}_desc-annotated_events.tsv'
    return list(Path(screen_annotations_root).glob(pattern))


def load_screen_annotations_tsv(filepath):
    """Load a validated Screen/Gameplay annotation TSV."""
    return pd.read_csv(filepath, sep='\t',
                       dtype={'trial_type': str, 'stim_file': str})


def load_annotated_events_tsv(filepath):
    """Load annotated events TSV file."""
    return pd.read_csv(filepath, sep='\t')


def extract_scene_id(scene_string):
    """Extract scene ID from scene trial_type string."""
    if not scene_string.startswith('scene-'):
        return None
    scene_id = scene_string.split('_')[0].replace('scene-', '')
    return scene_id


def identify_event_types(annotations_df):
    """Return canonical event types (ensures consistent columns across all runs)."""
    return CANONICAL_EVENT_TYPES


def extract_relative_path(absolute_path):
    """Extract relative path starting from 'sub-' for matching with annotations."""
    path_str = str(absolute_path)
    if 'sub-' in path_str:
        idx = path_str.index('sub-')
        return path_str[idx:]
    else:
        raise ValueError(f"Could not find 'sub-' in path: {path_str}")


def match_variables_to_annotations(variables_df, annotations_df):
    """Match variables DataFrame to annotations using stim_file and filename."""
    variables_df = variables_df.copy()
    variables_df['relative_path'] = variables_df['filename'].apply(extract_relative_path)
    rep_mapping = {}
    for _, row in annotations_df[annotations_df['trial_type'] == 'gym-retro_game'].iterrows():
        stim_file = row['stim_file']
        rep_mapping[stim_file] = {
            'onset': row['onset'],
            'duration': row['duration'],
            'phase': row.get('phase', None),
            'n_frames': row.get('n_frames', None)
        }
    matched_reps = []
    for relative_path, group in variables_df.groupby('relative_path'):
        if relative_path in rep_mapping:
            rep_data = rep_mapping[relative_path]
            group = group.copy()
            group['rep_onset'] = rep_data['onset']
            group['rep_duration'] = rep_data['duration']
            group['phase'] = rep_data['phase']
            matched_reps.append(group)
    if not matched_reps:
        raise ValueError("No matching repetitions found between variables and annotations")
    return pd.concat(matched_reps, ignore_index=True)


def add_frame_times(variables_df, frame_rate=60.099826520671044):
    """Add frame_time_in_run column based on rep_onset and frame_index."""
    variables_df = variables_df.copy()
    variables_df['frame_time_in_run'] = (
        variables_df['rep_onset'] + (variables_df['frame_index'] / frame_rate)
    )
    return variables_df


def create_event_columns(variables_df, annotations_df, event_types):
    """Create binary event columns based on frame_start and frame_stop."""
    variables_df = variables_df.copy()
    for event_type in event_types:
        col_name = f"event_{event_type.lower().replace('/', '_')}"
        variables_df[col_name] = 0
    for _, event_row in annotations_df.iterrows():
        trial_type = event_row['trial_type']
        if trial_type not in event_types:
            continue
        frame_start = event_row['frame_start']
        frame_stop = event_row['frame_stop']
        stim_file = event_row['stim_file']
        col_name = f"event_{trial_type.lower().replace('/', '_')}"
        if frame_start == frame_stop:
            mask = (
                (variables_df['relative_path'] == stim_file) &
                (variables_df['frame_index'] == frame_start)
            )
        else:
            mask = (
                (variables_df['relative_path'] == stim_file) &
                (variables_df['frame_index'] >= frame_start) &
                (variables_df['frame_index'] < frame_stop)
            )
        variables_df.loc[mask, col_name] = 1
    return variables_df


def propagate_stim_file_to_annotations(annotations_df):
    """Propagate stim_file from gym-retro_game rows to subsequent rows."""
    annotations_df = annotations_df.copy()
    current_stim_file = None
    for idx, row in annotations_df.iterrows():
        if row['trial_type'] == 'gym-retro_game':
            current_stim_file = row['stim_file']
        else:
            annotations_df.at[idx, 'stim_file'] = current_stim_file
    return annotations_df


def create_scene_column(variables_df, annotations_df):
    """Create categorical scene column based on frame_start and frame_stop."""
    variables_df = variables_df.copy()
    variables_df['scene'] = pd.NA
    variables_df['scene'] = variables_df['scene'].astype('string')
    scene_annotations = annotations_df[annotations_df['trial_type'].str.startswith('scene-', na=False)]
    for _, scene_row in scene_annotations.iterrows():
        scene_string = scene_row['trial_type']
        scene_id = extract_scene_id(scene_string)
        if scene_id is None:
            continue
        frame_start = scene_row['frame_start']
        frame_stop = scene_row['frame_stop']
        stim_file = scene_row['stim_file']
        mask = (
            (variables_df['relative_path'] == stim_file) &
            (variables_df['frame_index'] >= frame_start) &
            (variables_df['frame_index'] < frame_stop)
        )
        variables_df.loc[mask, 'scene'] = scene_id
    return variables_df


def get_session_combined_repvars_path(subject, session, per_session_combined_repvars_path=PATHS['per_session_combined_repvars_tsvs']):
    """Get path for session-level combined replay variables TSV."""
    base = per_session_combined_repvars_path / f'sub-{subject:02d}' / f'ses-{session:03d}'
    return base / f'sub-{subject:02d}_ses-{session:03d}_desc-combined_repvars.tsv'


def get_run_combined_path(subject, session, run, per_run_combined_path=PATHS['per_run_combined_repvars_and_bids_events']):
    """Get path for run-level combined TSV with BIDS events merged."""
    base = per_run_combined_path / f'sub-{subject:02d}' / f'ses-{session:03d}'
    base.mkdir(parents=True, exist_ok=True)
    return base / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-combined.tsv'


def save_run_combined_tsv(df, subject, session, run, per_run_combined_path=PATHS['per_run_combined_repvars_and_bids_events']):
    """Save run-level combined DataFrame to TSV."""
    filepath = get_run_combined_path(subject, session, run, per_run_combined_path)
    df.to_csv(filepath, sep='\t', index=False)
    return filepath


def merge_variables_and_annotations_for_run(subject, session, run, annotations_filepath=None,
                                            per_session_combined_repvars_path=PATHS['per_session_combined_repvars_tsvs'],
                                            events_path=PATHS['bids_annotated_tsvs'],
                                            screen_annotations_root=PATHS['screen_gameplay_annotations']):
    """Merge legacy annotations and the validated Screen/Gameplay source for one run."""
    session_vars_path = get_session_combined_repvars_path(subject, session, per_session_combined_repvars_path)
    variables_df = pd.read_csv(session_vars_path, sep='\t')
    if annotations_filepath is not None:
        annotations_df = load_annotated_events_tsv(annotations_filepath)
    else:
        annotations_path = get_annotated_events_path(subject, session, run, events_path)
        annotations_df = load_annotated_events_tsv(annotations_path)
    annotations_df = propagate_stim_file_to_annotations(annotations_df)
    event_types = identify_event_types(annotations_df)
    merged_df = match_variables_to_annotations(variables_df, annotations_df)
    merged_df = add_frame_times(merged_df)
    merged_df = create_event_columns(merged_df, annotations_df, event_types)
    merged_df = create_scene_column(merged_df, annotations_df)

    screen_path = get_screen_annotations_path(
        subject, session, run, screen_annotations_root)
    if not screen_path.exists():
        raise ScreenAnnotationError(
            f"Validated Screen/Gameplay annotations not found: {screen_path}")
    screen_df = load_screen_annotations_tsv(screen_path)
    merged_df, coverage = create_screen_columns(
        merged_df, screen_df, frame_rate=PARAMETERS['frame_rate'])
    merged_df = merged_df.drop(columns=['relative_path'])
    return merged_df, coverage


def extract_run_from_events_filename(filepath):
    """Extract run number from events filename, handling non-standard formats."""
    filename = filepath.stem
    if '_run-' in filename:
        try:
            run_part = filename.split('_run-')[1].split('_')[0]
            return int(run_part)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not extract run number from: {filename}") from e
    else:
        raise ValueError(f"No '_run-' pattern found in filename: {filename}")


def find_all_replay_sessions(replay_variables_path=PATHS['replay_variables_jsons']):
    """Find all unique subject/session combinations with replay variables."""
    pattern = '*/*/beh/variables'
    var_dirs = list(replay_variables_path.glob(pattern))
    sessions = []
    for var_dir in var_dirs:
        session_dir = var_dir.parent.parent
        subject_str = session_dir.parent.name
        session_str = session_dir.name
        subject = int(subject_str.split('-')[1])
        session = int(session_str.split('-')[1])
        sessions.append((subject, session))
    return sorted(set(sessions))


def process_all_runs_for_session(subject, session,
                                 per_session_combined_repvars_path=PATHS['per_session_combined_repvars_tsvs'],
                                 events_path=PATHS['bids_annotated_tsvs'],
                                 per_run_combined_path=PATHS['per_run_combined_repvars_and_bids_events'],
                                 screen_annotations_root=PATHS['screen_gameplay_annotations']):
    """Process all runs for a session, merging variables and annotations."""
    annotation_files = find_annotated_events(subject=subject, session=session, events_path=events_path)
    if not annotation_files:
        raise ValueError(f"No annotation files found for sub-{subject:02d}_ses-{session:03d}")
    results = []
    coverage_rows = []
    for annotation_file in annotation_files:
        try:
            run = extract_run_from_events_filename(annotation_file)
        except ValueError as e:
            print(f"  Skipping file: {e}")
            continue
        print(f"  Processing run {run:02d}...")
        try:
            merged_df, coverage = merge_variables_and_annotations_for_run(
                subject, session, run,
                annotations_filepath=annotation_file,
                per_session_combined_repvars_path=per_session_combined_repvars_path,
                events_path=events_path,
                screen_annotations_root=screen_annotations_root,
            )
            filepath = save_run_combined_tsv(merged_df, subject, session, run, per_run_combined_path)
            results.append({
                'subject': subject,
                'session': session,
                'run': run,
                'filepath': filepath,
                'n_frames': len(merged_df),
                'status': 'success'
            })
            coverage.update({
                'subject': subject,
                'session': session,
                'run': run,
                'status': 'success',
            })
            coverage_rows.append(coverage)
            print(f"    Saved {len(merged_df)} frames to {filepath.name}")
        except ScreenAnnotationError:
            raise
        except Exception as e:
            results.append({
                'subject': subject,
                'session': session,
                'run': run,
                'filepath': None,
                'n_frames': 0,
                'status': f'failed: {str(e)}'
            })
            coverage_rows.append({
                'subject': subject,
                'session': session,
                'run': run,
                'status': f'failed: {str(e)}',
            })
            print(f"    Failed: {str(e)}")
    return pd.DataFrame(results), pd.DataFrame(coverage_rows)


def process_all_run_merges(per_session_combined_repvars_path=PATHS['per_session_combined_repvars_tsvs'],
                           events_path=PATHS['bids_annotated_tsvs'],
                           per_run_combined_path=PATHS['per_run_combined_repvars_and_bids_events'],
                           screen_annotations_root=PATHS['screen_gameplay_annotations'],
                           sessions=None):
    """Process sessions and runs, creating merged run-level TSVs."""
    if sessions is None:
        sessions = find_all_replay_sessions()
    all_results = []
    all_coverage = []
    for subject, session in sessions:
        print(f"Processing sub-{subject:02d}_ses-{session:03d}...")
        try:
            results_df, coverage_df = process_all_runs_for_session(
                subject, session, per_session_combined_repvars_path, events_path,
                per_run_combined_path, screen_annotations_root
            )
            all_results.append(results_df)
            all_coverage.append(coverage_df)
        except ScreenAnnotationError:
            raise
        except Exception as e:
            print(f"  Failed entire session: {str(e)}")

    coverage_path = Path(per_run_combined_path) / 'screen_annotation_coverage.tsv'
    if all_coverage:
        coverage = pd.concat(all_coverage, ignore_index=True)
        if 'labels_seen' in coverage.columns:
            coverage['labels_seen'] = coverage['labels_seen'].apply(
                lambda value: '' if not isinstance(value, dict)
                else ';'.join(f'{k}:{v}' for k, v in sorted(value.items())))
        coverage.to_csv(coverage_path, sep='\t', index=False)
        print(f"Saved Screen annotation coverage to: {coverage_path}")

    load_or_collect_annotation_provenance(screen_annotations_root)
    carry_provenance(
        screen_annotations_root, per_run_combined_path, step='step2_merge',
        command='python scripts/python/merge_bids_events_and_repvars.py',
        extra={'coverage_file': str(coverage_path)})

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge BIDS events and validated Screen/Gameplay annotations')
    parser.add_argument(
        '--group', default=None, choices=list(GROUP_REGISTRY),
        help='Restrict processing to the train and test sessions of a group config. '
             'Without this, all sessions found in the replay variables are processed.')
    args = parser.parse_args()

    if args.group is not None:
        group = GROUP_REGISTRY[args.group]
        sessions = sorted({
            (subject_cfg['subject'], session)
            for subject_cfg in group['subjects']
            for session in subject_cfg['train_sessions'] + subject_cfg['test_sessions']
        })
        print(f"Restricting to {args.group}: {len(sessions)} subject-session pairs")
    else:
        sessions = None

    print("Merging BIDS events with replay variables...")
    print(f"Input paths:")
    print(f"  Session repvars: {PATHS['per_session_combined_repvars_tsvs']}")
    print(f"  BIDS events: {PATHS['bids_annotated_tsvs']}")
    print(f"  Screen/Gameplay annotations: {PATHS['screen_gameplay_annotations']}")
    print(f"Output path: {PATHS['per_run_combined_repvars_and_bids_events']}")
    print()

    merge_results = process_all_run_merges(
        per_session_combined_repvars_path=PATHS['per_session_combined_repvars_tsvs'],
        events_path=PATHS['bids_annotated_tsvs'],
        per_run_combined_path=PATHS['per_run_combined_repvars_and_bids_events'],
        screen_annotations_root=PATHS['screen_gameplay_annotations'],
        sessions=sessions,
    )

    if merge_results.empty:
        raise SystemExit("No runs were merged; check the inputs and Screen annotation coverage.")

    print("\n" + "="*80)
    print("Merging complete!")
    print(f"Total runs processed: {len(merge_results)}")
    print(f"Successful: {(merge_results['status'] == 'success').sum()}")
    print(f"Failed: {(merge_results['status'] != 'success').sum()}")

    if (merge_results['status'] != 'success').any():
        print("\nFailed runs:")
        print(merge_results[merge_results['status'] != 'success'][['subject', 'session', 'run', 'status']])
        raise SystemExit(1)
