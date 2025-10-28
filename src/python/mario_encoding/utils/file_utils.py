import json
import pandas as pd
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


def expand_scalar_metadata(data, scalar_keys=['filename', 'level', 'subject', 'session', 'actions', 'metadata']):
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
    expanded_data = expand_scalar_metadata(data)
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


def expand_scalar_metadata(data, scalar_keys=['filename', 'level', 'subject', 'session', 'actions']):
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
    expanded_data = expand_scalar_metadata(data)
    
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
        dfs.append(df)
    # Concatenate all repetitions
    merged_df = pd.concat(dfs, ignore_index=True)
    
    return merged_df


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
