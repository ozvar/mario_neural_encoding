"""
Process replay variables JSONs into session-level combined TSVs.

This script consolidates all replay variable data for each session into a single
TSV file, preparing data for subsequent merging with BIDS event annotations.
"""
import json
import pandas as pd
from pathlib import Path
from mario_encoding.config import PATHS


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


def extract_rep_id_from_path(filepath):
    """Extract repetition identifier from filepath."""
    filepath = Path(filepath)
    filename = filepath.stem
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
    expanded_data = expand_replay_scalar_metadata(data)
    df = pd.DataFrame(expanded_data)
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
    replay_files = find_replay_variables(subject=subject, session=session, 
                                         replay_variables_path=replay_variables_path)
    if not replay_files:
        raise ValueError(f"No replay variables found for sub-{subject:02d}_ses-{session:03d}")
    dfs = []
    for filepath in replay_files:
        df = load_and_parse_replay_variables(filepath)
        df['frame_index'] = range(len(df))
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


def get_session_combined_repvars_path(subject, session, 
                                      per_session_combined_repvars_path=PATHS['per_session_combined_repvars_tsvs']):
    """Get path for merged session-level combined replay variables TSV."""
    base = per_session_combined_repvars_path / f'sub-{subject:02d}' / f'ses-{session:03d}'
    base.mkdir(parents=True, exist_ok=True)
    return base / f'sub-{subject:02d}_ses-{session:03d}_desc-combined_repvars.tsv'


def save_session_combined_repvars_tsv(df, subject, session, 
                                      per_session_combined_repvars_path=PATHS['per_session_combined_repvars_tsvs']):
    """Save merged session-level combined replay variables DataFrame to TSV."""
    filepath = get_session_combined_repvars_path(subject, session, per_session_combined_repvars_path)
    df.to_csv(filepath, sep='\t', index=False)
    return filepath


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


def process_all_replay_sessions(replay_variables_path=PATHS['replay_variables_jsons'],
                                per_session_combined_repvars_path=PATHS['per_session_combined_repvars_tsvs']):
    """Merge and save combined replay variables for all sessions."""
    sessions = find_all_replay_sessions(replay_variables_path)
    results = []
    for subject, session in sessions:
        print(f"Processing sub-{subject:02d}_ses-{session:03d}...")
        try:
            df = merge_replay_variables_for_session(subject, session, replay_variables_path)
            filepath = save_session_combined_repvars_tsv(df, subject, session, per_session_combined_repvars_path)
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


if __name__ == '__main__':
    print("Processing replay sessions...")
    print(f"Input path: {PATHS['replay_variables_jsons']}")
    print(f"Output path: {PATHS['per_session_combined_repvars_tsvs']}")
    print()
    
    replay_results = process_all_replay_sessions(
        replay_variables_path=PATHS['replay_variables_jsons'],
        per_session_combined_repvars_path=PATHS['per_session_combined_repvars_tsvs']
    )
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Total sessions processed: {len(replay_results)}")
    print(f"Successful: {(replay_results['status'] == 'success').sum()}")
    print(f"Failed: {(replay_results['status'] != 'success').sum()}")
    
    if (replay_results['status'] != 'success').any():
        print("\nFailed sessions:")
        print(replay_results[replay_results['status'] != 'success'][['subject', 'session', 'status']])
