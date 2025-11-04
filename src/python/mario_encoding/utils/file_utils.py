"""
General file utilities for finding and accessing Mario dataset files.

Contains utilities for locating gamelogs, events files, and scene clip data.
Does not include processing logic - see standalone processing scripts.
"""
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


def expand_scenes_scalar_metadata(data, scalar_keys=['filename', 'level', 'subject', 'session', 'actions', 'metadata']):
    """Expand scalar metadata values to match frame count."""
    frame_count = None
    for key, value in data.items():
        if isinstance(value, list) and key not in scalar_keys:
            frame_count = len(value)
            break
    if frame_count is None:
        raise ValueError("No frame-wise data found to determine frame count")
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
    expanded_data = expand_scenes_scalar_metadata(data)
    df = pd.DataFrame(expanded_data)
    scene_code = extract_scene_code_from_path(filepath)
    df['scene_code'] = scene_code
    return df


def load_and_parse_scene_clip(filepath):
    """Load and parse scene clip JSON into DataFrame (convenience wrapper)."""
    data = load_scene_clip_json(filepath)
    df = parse_scene_clip_df(data, filepath)
    return df
