import json
import pandas as pd
import numpy as np
from pathlib import Path
from mario_encoding.config import PATHS


def get_subject_session_path(subject, session, datatype='func', raw_data_path=PATHS['raw_data']):
    """Get subject/session directory path."""
    return raw_data_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / datatype


def get_gamelog_path(subject, session, level, rep, raw_data_path=PATHS['raw_data']):
    """Get gamelog file path."""
    base = get_subject_session_path(subject, session, 'gamelogs', raw_data_path)
    return base / f'sub-{subject:02d}_ses-{session:03d}_task-mario_level-{level}_rep-{rep:03d}.bk2'


def get_events_path(subject, session, run, raw_data_path=PATHS['raw_data']):
    """Get events TSV file path."""
    base = get_subject_session_path(subject, session, 'func', raw_data_path)
    return base / f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run:02d}_events.tsv'


def find_gamelogs(subject=None, session=None, level=None, raw_data_path=PATHS['raw_data']):
    """Find gamelogs matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    level_pattern = f'*_level-{level}_*.bk2' if level else '*.bk2'
    
    pattern = f'{sub_pattern}/{ses_pattern}/gamelogs/{level_pattern}'
    return list(raw_data_path.glob(pattern))


def find_events(subject=None, session=None, run=None, raw_data_path=PATHS['raw_data']):
    """Find all events TSV files matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'*_run-{run:02d}_events.tsv' if run else '*_events.tsv'
    
    pattern = f'{sub_pattern}/{ses_pattern}/func/{run_pattern}'
    return list(raw_data_path.glob(pattern))


# ============================================================================
# Scene clip JSON I/O
# ============================================================================


def get_scene_clip_path(subject, session, run, level, scene, clip, 
                        scene_clips_path=PATHS['scene_clip_jsons']):
    """Get scene clip JSON file path."""
    base = scene_clips_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'beh' / 'variables'
    return base / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_level-{level}_scene-{scene}_clip-{clip}.json'


def find_scene_clips(subject=None, session=None, run=None, level=None, 
                     scene_clips_path=PATHS['scene_clip_jsons']):
    """Find all scene clip JSONs matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'run-{run:02d}' if run else 'run-*'
    level_pattern = f'level-{level}' if level else 'level-*'
    
    pattern = f'{sub_pattern}/{ses_pattern}/beh/variables/{run_pattern}_{level_pattern}_*.json'
    return list(scene_clips_path.glob(pattern))


def load_scene_clip_json(filepath):
    """Load scene clip JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def expand_scenes_scalar_metadata(data, scalar_keys=['filename', 'level', 'subject', 'session', 'actions', 'metadata']):
    """Expand scalar metadata values to match frame count."""
    # Find frame count from first list-type value
    frame_count = None
    for key, value in data.items():
        if isinstance(value, list) and key not in scalar_keys:
            frame_count = len(value)
            break
    if frame_count is None:
        raise ValueError("No frame-wise data found to determine frame count")
    # Expand scalar values
    expanded_data = {}
    for key, value in data.items():
        if key in scalar_keys:
            expanded_data[key] = [value] * frame_count
        else:
            expanded_data[key] = value
    
    return expanded_data


def extract_scene_code_from_path(filepath):
    """Extract formatted scene code from filepath."""
    filepath = Path(filepath)
    filename = filepath.stem
    
    # Parse: sub-01_ses-001_run-01_level-w1l1_scene-0_clip-00101010000122
    parts = filename.split('_')
    level = None
    scene = None
    clip = None
    
    for part in parts:
        if part.startswith('level-'):
            level = part.split('-', 1)[1]
        elif part.startswith('scene-'):
            scene = part.split('-', 1)[1]
        elif part.startswith('clip-'):
            clip = part.split('-', 1)[1]
    
    if level and scene and clip:
        return f'scene-{level}s{scene}_code-{clip}'
    else:
        raise ValueError(f"Could not parse scene code from filename: {filename}")


def parse_scene_clip_df(data, filepath):
    """Parse scene clip JSON data into DataFrame with frame-level features."""
    # Expand scalar metadata
    expanded_data = expand_scenes_scalar_metadata(data)
    # Convert to DataFrame
    df = pd.DataFrame(expanded_data)
    # Add scene code column
    scene_code = extract_scene_code_from_path(filepath)
    df['scene_code'] = scene_code
    
    return df


def load_and_parse_scene_clip(filepath):
    """Load and parse scene clip JSON into DataFrame (convenience wrapper)."""
    data = load_scene_clip_json(filepath)
    df = parse_scene_clip_df(data, filepath)
    return df

# ============================================================================
# Replay variables JSON I/O
# ============================================================================

def get_replay_variables_path(subject, session, level, rep, 
                               replay_variables_path=PATHS['replay_variables_jsons']):
    """Get replay variables JSON file path."""
    base = replay_variables_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'beh' / 'variables'
    return base / f'sub-{subject:02d}_ses-{session:03d}_task-mario_level-{level}_rep-{rep:03d}.json'


def find_replay_variables(subject=None, session=None, level=None, 
                          replay_variables_path=PATHS['replay_variables_jsons']):
    """Find all replay variables JSONs matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    level_pattern = f'level-{level}' if level else 'level-*'
    
    pattern = f'{sub_pattern}/{ses_pattern}/beh/variables/*{level_pattern}_*.json'
    return list(replay_variables_path.glob(pattern))


def load_replay_variables_json(filepath):
    """Load replay variables JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def expand_replay_scalar_metadata(data, scalar_keys=['filename', 'level', 'subject', 'session', 'actions']):
    """Expand scalar metadata values to match frame count."""
    # Find frame count from first list-type value
    frame_count = None
    for key, value in data.items():
        if isinstance(value, list) and key not in scalar_keys:
            frame_count = len(value)
            break
    
    if frame_count is None:
        raise ValueError("No frame-wise data found to determine frame count")
    
    # Expand scalar values
    expanded_data = {}
    for key, value in data.items():
        if key in scalar_keys:
            expanded_data[key] = [value] * frame_count
        else:
            expanded_data[key] = value
    
    return expanded_data


def extract_rep_id_from_path(filepath):
    """Extract repetition identifier from filepath."""
    filepath = Path(filepath)
    filename = filepath.stem
    
    # Parse: sub-01_ses-001_task-mario_level-w1l1_rep-000
    parts = filename.split('_')
    level = None
    rep = None
    
    for part in parts:
        if part.startswith('level-'):
            level = part.split('-', 1)[1]
        elif part.startswith('rep-'):
            rep = part.split('-', 1)[1]
    
    if level and rep:
        return f'{level}_rep-{rep}'
    else:
        raise ValueError(f"Could not parse rep ID from filename: {filename}")


def parse_replay_variables_df(data, filepath):
    """Parse replay variables JSON data into DataFrame with frame-level features."""
    # Expand scalar metadata
    expanded_data = expand_replay_scalar_metadata(data)
    
    # Convert to DataFrame
    df = pd.DataFrame(expanded_data)
    
    # Add rep ID column
    rep_id = extract_rep_id_from_path(filepath)
    df['rep_id'] = rep_id
    
    return df


def load_and_parse_replay_variables(filepath):
    """Load and parse replay variables JSON into DataFrame (convenience wrapper)."""
    data = load_replay_variables_json(filepath)
    df = parse_replay_variables_df(data, filepath)
    return df


def merge_replay_variables_for_session(subject, session, 
                                       replay_variables_path=PATHS['replay_variables_jsons']):
    """Merge all replay variables for a session into single DataFrame."""
    # Find all replay variables for this session
    replay_files = find_replay_variables(subject=subject, session=session, 
                                         replay_variables_path=replay_variables_path)
    if not replay_files:
        raise ValueError(f"No replay variables found for sub-{subject:02d}_ses-{session:03d}")
    # Load and parse each file
    dfs = []
    for filepath in replay_files:
        df = load_and_parse_replay_variables(filepath)
        df['frame_index'] = range(len(df))
        dfs.append(df)
    # Concatenate all repetitions
    merged_df = pd.concat(dfs, ignore_index=True)
    
    return merged_df


def get_session_framewise_path(subject, session, 
                                framewise_path=PATHS['per_session_framewise_tsvs']):
    """Get path for merged session-level framewise TSV."""
    base = framewise_path / f'sub-{subject:02d}' / f'ses-{session:03d}'
    base.mkdir(parents=True, exist_ok=True)
    return base / f'sub-{subject:02d}_ses-{session:03d}_desc-framewise_variables.tsv'


def save_session_framewise_tsv(df, subject, session, 
                                framewise_path=PATHS['per_session_framewise_tsvs']):
    """Save merged session-level framewise DataFrame to TSV."""
    filepath = get_session_framewise_path(subject, session, framewise_path)
    df.to_csv(filepath, sep='\t', index=False)
    return filepath


def find_all_replay_sessions(replay_variables_path=PATHS['replay_variables_jsons']):
    """Find all unique subject/session combinations with replay variables."""
    # Find all variables directories: sub-XX/ses-XXX/beh/variables/
    pattern = '*/*/beh/variables'
    var_dirs = list(replay_variables_path.glob(pattern))
    
    sessions = []
    for var_dir in var_dirs:
        # Extract subject and session from path: .../sub-01/ses-001/beh/variables
        session_dir = var_dir.parent.parent
        subject_str = session_dir.parent.name  # 'sub-01'
        session_str = session_dir.name          # 'ses-001'
        
        subject = int(subject_str.split('-')[1])
        session = int(session_str.split('-')[1])
        sessions.append((subject, session))
    
    return sorted(set(sessions))


def process_all_replay_sessions(replay_variables_path=PATHS['replay_variables_jsons'],
                         framewise_path=PATHS['per_session_framewise_tsvs']):
    """Merge and save framewise variables for all sessions."""
    sessions = find_all_replay_sessions(replay_variables_path)
    
    results = []
    for subject, session in sessions:
        print(f"Processing sub-{subject:02d}_ses-{session:03d}...")
        
        try:
            # Merge variables for session
            df = merge_replay_variables_for_session(subject, session, replay_variables_path)
            
            # Save to disk
            filepath = save_session_framewise_tsv(df, subject, session, framewise_path)
            
            results.append({
                'subject': subject,
                'session': session,
                'filepath': filepath,
                'n_frames': len(df),
                'status': 'success'
            })
            print(f"  Saved {len(df)} frames to {filepath}")
            
        except Exception as e:
            results.append({
                'subject': subject,
                'session': session,
                'filepath': None,
                'n_frames': 0,
                'status': f'failed: {str(e)}'
            })
            print(f"  Failed: {str(e)}")
    
    return pd.DataFrame(results)


# ============================================================================
# Annotated events TSV I/O
# ============================================================================

def get_annotated_events_path(subject, session, run, 
                               events_path=PATHS['bids_annotated_tsvs']):
    """Get annotated events TSV file path."""
    base = events_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func'
    return base / f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run:02d}_desc-annotated_events.tsv'


def find_annotated_events(subject=None, session=None, run=None,
                          events_path=PATHS['bids_annotated_tsvs']):
    """Find all annotated events TSV files matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'run-{run:02d}' if run else 'run-*'
    
    pattern = f'{sub_pattern}/{ses_pattern}/func/*{run_pattern}_desc-annotated_events.tsv'
    return list(events_path.glob(pattern))


def load_annotated_events_tsv(filepath):
    """Load annotated events TSV file."""
    return pd.read_csv(filepath, sep='\t')


def parse_annotated_events_df(events_df):
    """Parse annotated events DataFrame (already in correct format from TSV)."""
    return events_df


def load_and_parse_annotated_events(filepath):
    """Load and parse annotated events TSV (convenience wrapper)."""
    df = load_annotated_events_tsv(filepath)
    return parse_annotated_events_df(df)


# ============================================================================
# Merge framewise variables and BIDS-like annotations TSV I/O
# ============================================================================

def extract_scene_id(scene_string):
    """Extract scene ID from scene trial_type string."""
    # From: 'scene-w1l1s0_code-00101000000122'
    # To: 'w1l1s0'
    if not scene_string.startswith('scene-'):
        return None
    # Split by '_' and take first part, then remove 'scene-' prefix
    scene_id = scene_string.split('_')[0].replace('scene-', '')
    return scene_id


def identify_event_types(annotations_df):
    """Identify event trial types (excluding buttons, scenes, gym-retro_game)."""
    button_actions = {'A', 'B', 'LEFT', 'RIGHT', 'UP', 'DOWN'}
    exclude_types = {'gym-retro_game'}
    all_trial_types = set(annotations_df['trial_type'].unique())
    # Filter out buttons, scenes, and gym-retro_game
    event_types = []
    for trial_type in all_trial_types:
        if (trial_type not in button_actions and 
            trial_type not in exclude_types and
            not trial_type.startswith('scene-')):
            event_types.append(trial_type)
    
    return sorted(event_types)


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
    # Extract relative paths from variables filenames
    variables_df = variables_df.copy()
    variables_df['relative_path'] = variables_df['filename'].apply(extract_relative_path)
    # Create mapping from stim_file to rep data
    rep_mapping = {}
    for _, row in annotations_df[annotations_df['trial_type'] == 'gym-retro_game'].iterrows():
        stim_file = row['stim_file']
        rep_mapping[stim_file] = {
            'onset': row['onset'],
            'duration': row['duration'],
            'phase': row.get('phase', None),
            'n_frames': row.get('n_frames', None)
        }
    # Match and add metadata to variables
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
    # Initialize event columns
    for event_type in event_types:
        col_name = f"event_{event_type.lower().replace('/', '_')}"
        variables_df[col_name] = 0
    # Fill event columns using frame_start and frame_stop
    for _, event_row in annotations_df.iterrows():
        trial_type = event_row['trial_type']
        
        if trial_type not in event_types:
            continue
        frame_start = event_row['frame_start']
        frame_stop = event_row['frame_stop']
        stim_file = event_row['stim_file']
        # Find matching frames in variables
        col_name = f"event_{trial_type.lower().replace('/', '_')}"
        # Handle single-frame events (frame_start == frame_stop)
        if frame_start == frame_stop:
            mask = (
                (variables_df['relative_path'] == stim_file) &
                (variables_df['frame_index'] == frame_start)
            )
        else:
            # Multi-frame events use exclusive upper bound
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
            # Fill in stim_file for non-gym-retro_game rows
            annotations_df.at[idx, 'stim_file'] = current_stim_file
    
    return annotations_df


def create_scene_column(variables_df, annotations_df):
    """Create categorical scene column based on frame_start and frame_stop."""
    variables_df = variables_df.copy()
    variables_df['scene'] = pd.NA
    variables_df['scene'] = variables_df['scene'].astype('string')
    # Fill scene column
    scene_annotations = annotations_df[annotations_df['trial_type'].str.startswith('scene-', na=False)]
    for _, scene_row in scene_annotations.iterrows():
        scene_string = scene_row['trial_type']
        scene_id = extract_scene_id(scene_string)
        
        if scene_id is None:
            continue
        
        frame_start = scene_row['frame_start']
        frame_stop = scene_row['frame_stop']
        stim_file = scene_row['stim_file']
        # Find matching frames
        mask = (
            (variables_df['relative_path'] == stim_file) &
            (variables_df['frame_index'] >= frame_start) &
            (variables_df['frame_index'] < frame_stop)
        )
        variables_df.loc[mask, 'scene'] = scene_id
    
    return variables_df


def merge_variables_and_annotations_for_run(subject, session, run,
                                            annotations_filepath=None,
                                            framewise_path=PATHS['per_session_framewise_tsvs'],
                                            events_path=PATHS['bids_annotated_tsvs']):
    """Merge variables and annotations for a single run."""
    # Load session-level variables
    session_vars_path = get_session_framewise_path(subject, session, framewise_path)
    variables_df = pd.read_csv(session_vars_path, sep='\t')
    # Load run-level annotations
    if annotations_filepath is not None:
        # Use provided filepath (handles non-standard names)
        annotations_df = load_annotated_events_tsv(annotations_filepath)
    else:
        # Construct standard path
        annotations_path = get_annotated_events_path(subject, session, run, events_path)
        annotations_df = load_annotated_events_tsv(annotations_path)
    #Propagate stim_file first before any other processing (for correct scene labelling later)
    annotations_df = propagate_stim_file_to_annotations(annotations_df)
    # Identify event types
    event_types = identify_event_types(annotations_df)
    # Match variables to annotations
    merged_df = match_variables_to_annotations(variables_df, annotations_df)
    # Add frame times
    merged_df = add_frame_times(merged_df)
    # Create event columns
    merged_df = create_event_columns(merged_df, annotations_df, event_types)
    # Create scene column
    merged_df = create_scene_column(merged_df, annotations_df)
    # Drop relative_path helper column
    merged_df = merged_df.drop(columns=['relative_path'])
    
    return merged_df


def get_run_framewise_path(subject, session, run,
                           framewise_path=PATHS['per_session_framewise_tsvs']):
    """Get path for run-level merged framewise TSV."""
    base = framewise_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'run_framewise'
    base.mkdir(parents=True, exist_ok=True)
    return base / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-framewise_merged.tsv'


def save_run_framewise_tsv(df, subject, session, run,
                           framewise_path=PATHS['per_session_framewise_tsvs']):
    """Save run-level merged framewise DataFrame to TSV."""
    filepath = get_run_framewise_path(subject, session, run, framewise_path)
    df.to_csv(filepath, sep='\t', index=False)
    return filepath


def extract_run_from_events_filename(filepath):
    """Extract run number from events filename, handling non-standard formats."""
    filename = filepath.stem
    # Look for _run-XX pattern
    if '_run-' in filename:
        try:
            # Split on '_run-' and take the next part before next '_'
            run_part = filename.split('_run-')[1].split('_')[0]
            return int(run_part)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not extract run number from: {filename}") from e
    else:
        raise ValueError(f"No '_run-' pattern found in filename: {filename}")


def process_all_runs_for_session(subject, session,
                                 framewise_path=PATHS['per_session_framewise_tsvs'],
                                 events_path=PATHS['bids_annotated_tsvs']):
    """Process all runs for a session, merging variables and annotations."""
    # Find all annotation files for this session
    annotation_files = find_annotated_events(subject=subject, session=session, 
                                             events_path=events_path)
    if not annotation_files:
        raise ValueError(f"No annotation files found for sub-{subject:02d}_ses-{session:03d}")
    results = []
    for annotation_file in annotation_files:
        # Extract run number from filename
        try:
            run = extract_run_from_events_filename(annotation_file)
        except ValueError as e:
            print(f"  Skipping file: {e}")
            continue
        print(f"  Processing run {run:02d}...")
        try:
            # Merge variables and annotations
            merged_df = merge_variables_and_annotations_for_run(
                subject, session, run, 
                annotations_filepath=annotation_file,
                framewise_path=framewise_path,
                events_path=events_path
            )
            # Save to disk
            filepath = save_run_framewise_tsv(merged_df, subject, session, run, framewise_path)
            results.append({
                'subject': subject,
                'session': session,
                'run': run,
                'filepath': filepath,
                'n_frames': len(merged_df),
                'status': 'success'
            })
            print(f"    Saved {len(merged_df)} frames to {filepath.name}")
            
        except Exception as e:
            results.append({
                'subject': subject,
                'session': session,
                'run': run,
                'filepath': None,
                'n_frames': 0,
                'status': f'failed: {str(e)}'
            })
            print(f"    Failed: {str(e)}")
    
    return pd.DataFrame(results)


def process_all_run_merges(framewise_path=PATHS['per_session_framewise_tsvs'],
                           events_path=PATHS['bids_annotated_tsvs']):
    """Process all sessions and runs, creating merged run-level TSVs."""
    sessions = find_all_replay_sessions()
    all_results = []
    for subject, session in sessions:
        print(f"Processing sub-{subject:02d}_ses-{session:03d}...")
        try:
            results_df = process_all_runs_for_session(
                subject, session, framewise_path, events_path
            )
            all_results.append(results_df)
        except Exception as e:
            print(f"  Failed entire session: {str(e)}")
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()
