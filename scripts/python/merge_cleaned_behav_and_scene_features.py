"""
Process run-level combined dataframes to create pruned feature sets.

This script takes run-level TSVs (with replay variables and BIDS events merged),
applies feature engineering transformations, and outputs pruned feature matrices
ready for encoding model analysis.
"""
import pandas as pd
from pathlib import Path
from mario_encoding.config import PATHS, PARAMETERS


# Column groups to drop
COLUMNS_TO_DROP = {
    'metadata': ['filename', 'subject', 'session', 'actions', 'rep_id', 'phase'],
    'timing': ['frame_index', 'rep_onset', 'rep_duration', 'frame_time_in_run'],
    'unused_buttons': ['null', 'SELECT', 'START'],
    'level_descriptors': ['world', 'stage', 'level', 'area', 'scene', 'level_layout', 'levelHi', 'levelLo'],
    'position': ['player_x_posHi', 'player_x_posLo', 'player_y_screen', 'player_y_pos', 'xscrollLo', 'xscrollHi'],
    'redundant_status': ['powerup_appear', 'lives', 'player_sprite', 'player_state', 'walk_animation', 'star_timer'],
    'redundant_scene': ['x Hi (entry)', 'x Lo (entry)', 'x Hi (exit)', 'x Lo (exit)', 'Entry point', 'Exit point', 'Layout'],
    'misc': ['fireball_counter']
}


def merge_scene_features(df, mastersheet_path):
    """Merge scene-level perceptual features from mastersheet onto framewise dataframe."""
    df['World'] = df['scene'].str.extract(r'w(\d+)').astype('Int64')
    df['Level'] = df['scene'].str.extract(r'l(\d+)').astype('Int64')
    df['Scene'] = df['scene'].str.extract(r's(\d+)').astype('Int64')
    scenes_df = pd.read_csv(mastersheet_path)
    scenes_df = scenes_df.dropna(subset=['World', 'Level', 'Scene'])
    scene_feature_cols = [col for col in scenes_df.columns if col not in ['World', 'Level', 'Scene']]
    df_with_scenes = df.merge(scenes_df, on=['World', 'Level', 'Scene'], how='left')
    df_with_scenes[scene_feature_cols] = df_with_scenes[scene_feature_cols].fillna(0).astype(int)
    df_with_scenes = df_with_scenes.drop(['World', 'Level', 'Scene'], axis=1)
    return df_with_scenes


def encode_categorical_features(df):
    """Apply all categorical encodings: powerstate, scrolling, powerup, jump, movement."""
    # Powerstate: one-hot encode (drop small as reference)
    df['mario_big'] = (df['powerstate'] == 1000000).astype(int)
    df['mario_fire'] = (df['powerstate'] == 2000000).astype(int)
    df['mario_star'] = (df['powerstate'] == 10000).astype(int)
    df = df.drop(['powerstate'], axis=1)
    # Scrolling: binarize
    df['screen_scrolling'] = df['scrolling'].isin([17, 21]).astype(int)
    df = df.drop(['scrolling'], axis=1)
    # Powerup visibility: binarize
    df['powerup_visible'] = df['powerup_yes_no'].isin([46, 48]).astype(int)
    df = df.drop('powerup_yes_no', axis=1)
    # Jump airborne: binarize
    df['is_airborne'] = (df['jump_airborne'] > 0).astype(int)
    df = df.drop('jump_airborne', axis=1)
    # Moving direction: one-hot encode (drop stationary as reference)
    df['moving_right'] = (df['moving_direction'] == 1).astype(int)
    df['moving_left'] = (df['moving_direction'] == 2).astype(int)
    df = df.drop(['moving_direction'], axis=1)
    return df


def drop_columns(df, columns_to_drop_dict):
    """Drop all columns specified in nested dictionary structure."""
    all_columns_to_drop = []
    for category, columns in columns_to_drop_dict.items():
        all_columns_to_drop.extend(columns)
    # Only drop columns that exist in dataframe
    columns_to_drop_existing = [col for col in all_columns_to_drop if col in df.columns]
    df = df.drop(columns_to_drop_existing, axis=1)
    return df


def extend_event_duration(df, event_col, n_frames):
    """Extend event by specified number of frames."""
    if event_col not in df.columns:
        return df
    events = df[df[event_col] == 1].index
    if events.size == 0:
        return df
    for event_idx in events:
        if df.index[-1] > event_idx + n_frames:
            df.loc[event_idx:event_idx + n_frames - 1, event_col] = 1
    return df


def extend_event_durations(df, event_durations_dict):
    """Extend duration of multiple events based on dictionary specification."""
    for event_type, n_frames in event_durations_dict.items():
        df = extend_event_duration(df, event_col=event_type, n_frames=n_frames)
    return df


def create_kill_events(df):
    """Create kill event features from enemy_kill RAM columns and drop originals."""
    enemy_kill_cols = ['enemy_kill30', 'enemy_kill31', 'enemy_kill32', 'enemy_kill33', 'enemy_kill34']
    # Stomp (value 4)
    df['event_kill_stomp'] = ((df['enemy_kill30'] == 4) | (df['enemy_kill31'] == 4) | (df['enemy_kill32'] == 4) | (df['enemy_kill33'] == 4) | (df['enemy_kill34'] == 4)).astype(int)
    # Impact (value 34)
    df['event_kill_impact'] = ((df['enemy_kill30'] == 34) | (df['enemy_kill31'] == 34) | (df['enemy_kill32'] == 34) | (df['enemy_kill33'] == 34) | (df['enemy_kill34'] == 34)).astype(int)
    # Kick (value 132)
    df['event_kill_kick'] = ((df['enemy_kill30'] == 132) | (df['enemy_kill31'] == 132) | (df['enemy_kill32'] == 132) | (df['enemy_kill33'] == 132) | (df['enemy_kill34'] == 132)).astype(int)
    # Drop original enemy_kill columns
    enemy_kill_cols_to_drop = enemy_kill_cols + ['enemy_kill35']
    df = df.drop(enemy_kill_cols_to_drop, axis=1)
    return df


def collect_and_prune_behav_features(filepath, mastersheet_path, event_durations_dict, columns_to_drop_dict):
    """
    Complete feature engineering pipeline for Mario fMRI encoding model.
    
    Loads run-level combined data, merges scene features, encodes categorical variables,
    extends event durations, creates kill events, and drops unnecessary columns.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to run-level combined TSV file
    mastersheet_path : str or Path
        Path to scenes mastersheet CSV
    event_durations_dict : dict
        Dictionary mapping event column names to extension durations in frames
    columns_to_drop_dict : dict
        Nested dictionary of column categories to drop
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with pruned features
    """
    df = pd.read_csv(filepath, sep='\t')
    df = merge_scene_features(df, mastersheet_path)
    df = encode_categorical_features(df)
    df = drop_columns(df, columns_to_drop_dict)
    df = extend_event_durations(df, event_durations_dict)
    df = create_kill_events(df)
    return df


def get_run_combined_path(subject, session, run, per_run_combined_path=PATHS['per_run_combined_repvars_and_bids_events']):
    """Get path for run-level combined TSV (input)."""
    base = per_run_combined_path / f'sub-{subject:02d}' / f'ses-{session:03d}'
    return base / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-combined.tsv'


def find_run_combined_files(subject=None, session=None, run=None, per_run_combined_path=PATHS['per_run_combined_repvars_and_bids_events']):
    """Find all run-level combined TSV files matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'run-{run:02d}' if run else 'run-*'
    pattern = f'{sub_pattern}/{ses_pattern}/*{run_pattern}_desc-combined.tsv'
    return list(per_run_combined_path.glob(pattern))


def get_run_pruned_features_path(subject, session, run, per_run_pruned_path=PATHS['per_run_pruned_features']):
    """Get path for run-level pruned features TSV (output)."""
    base = per_run_pruned_path / f'sub-{subject:02d}' / f'ses-{session:03d}'
    base.mkdir(parents=True, exist_ok=True)
    return base / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-pruned_features.tsv'


def save_run_pruned_features_tsv(df, subject, session, run, per_run_pruned_path=PATHS['per_run_pruned_features']):
    """Save run-level pruned features DataFrame to TSV."""
    filepath = get_run_pruned_features_path(subject, session, run, per_run_pruned_path)
    df.to_csv(filepath, sep='\t', index=False)
    return filepath


def extract_subject_session_run_from_path(filepath):
    """Extract subject, session, run numbers from filepath."""
    filepath = Path(filepath)
    filename = filepath.stem
    parts = filename.split('_')
    subject = None
    session = None
    run = None
    for part in parts:
        if part.startswith('sub-'):
            subject = int(part.split('-')[1])
        elif part.startswith('ses-'):
            session = int(part.split('-')[1])
        elif part.startswith('run-'):
            run = int(part.split('-')[1])
    if subject is not None and session is not None and run is not None:
        return subject, session, run
    else:
        raise ValueError(f"Could not parse subject/session/run from filename: {filename}")


def process_run_pruned_features(filepath, mastersheet_path, event_durations_dict, columns_to_drop_dict, per_run_pruned_path=PATHS['per_run_pruned_features']):
    """Process a single run file to create pruned features."""
    subject, session, run = extract_subject_session_run_from_path(filepath)
    print(f"  Processing run {run:02d}...")
    try:
        pruned_df = collect_and_prune_behav_features(
            filepath=filepath,
            mastersheet_path=mastersheet_path,
            event_durations_dict=event_durations_dict,
            columns_to_drop_dict=columns_to_drop_dict
        )
        output_filepath = save_run_pruned_features_tsv(pruned_df, subject, session, run, per_run_pruned_path)
        return {
            'subject': subject,
            'session': session,
            'run': run,
            'filepath': output_filepath,
            'n_frames': len(pruned_df),
            'n_features': len(pruned_df.columns),
            'status': 'success'
        }
    except Exception as e:
        return {
            'subject': subject,
            'session': session,
            'run': run,
            'filepath': None,
            'n_frames': 0,
            'n_features': 0,
            'status': f'failed: {str(e)}'
        }


def process_all_run_pruned_features(mastersheet_path, event_durations_dict, columns_to_drop_dict,
                                    per_run_combined_path=PATHS['per_run_combined_repvars_and_bids_events'],
                                    per_run_pruned_path=PATHS['per_run_pruned_features']):
    """Process all run-level combined files to create pruned feature sets."""
    combined_files = find_run_combined_files(per_run_combined_path=per_run_combined_path)
    if not combined_files:
        raise ValueError(f"No combined run files found in {per_run_combined_path}")
    print(f"Found {len(combined_files)} run files to process")
    print()
    results = []
    current_session = None
    for filepath in sorted(combined_files):
        subject, session, run = extract_subject_session_run_from_path(filepath)
        if (subject, session) != current_session:
            current_session = (subject, session)
            print(f"Processing sub-{subject:02d}_ses-{session:03d}...")
        result = process_run_pruned_features(
            filepath=filepath,
            mastersheet_path=mastersheet_path,
            event_durations_dict=event_durations_dict,
            columns_to_drop_dict=columns_to_drop_dict,
            per_run_pruned_path=per_run_pruned_path
        )
        results.append(result)
        if result['status'] == 'success':
            print(f"    Saved {result['n_frames']} frames, {result['n_features']} features to {Path(result['filepath']).name}")
        else:
            print(f"    Failed: {result['status']}")
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Get mastersheet path - adjust if needed
    mastersheet_path = PATHS['src_python'].parent.parent.parent / 'mario.scenes' / 'sourcedata' / 'scenes_info' / 'scenes_mastersheet.csv'
    
    print("Processing pruned features for all runs...")
    print(f"Input path: {PATHS['per_run_combined_repvars_and_bids_events']}")
    print(f"Output path: {PATHS['per_run_pruned_features']}")
    print(f"Mastersheet: {mastersheet_path}")
    print()
    
    results = process_all_run_pruned_features(
        mastersheet_path=mastersheet_path,
        event_durations_dict=PARAMETERS['event_frame_duration_dict'],
        columns_to_drop_dict=COLUMNS_TO_DROP
    )
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Total runs processed: {len(results)}")
    print(f"Successful: {(results['status'] == 'success').sum()}")
    print(f"Failed: {(results['status'] != 'success').sum()}")
    
    if (results['status'] != 'success').any():
        print("\nFailed runs:")
        print(results[results['status'] != 'success'][['subject', 'session', 'run', 'status']])
