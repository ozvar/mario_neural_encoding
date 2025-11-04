import pandas as pd
from pathlib import Path
from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.utils import pandas_styleset


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
    
    Loads framewise behavioral data, merges scene features, encodes categorical variables,
    extends event durations, creates kill events, and drops unnecessary columns.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to framewise merged TSV file
    mastersheet_path : str or Path
        Path to scenes mastersheet CSV
    event_durations_dict : dict
        Dictionary mapping event column names to extension durations in frames
    columns_to_drop_dict : dict
        Nested dictionary of column categories to drop
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with engineered features
    """
    df = pd.read_csv(filepath, sep='\t')
    df = merge_scene_features(df, mastersheet_path)
    df = encode_categorical_features(df)
    df = drop_columns(df, columns_to_drop_dict)
    df = extend_event_durations(df, event_durations_dict)
    df = create_kill_events(df)
    return df


if __name__ == '__main__':
    filepath = '/home/ozvar/Git/cneuromod/mario_neural_encoding/inputs/preprocessed_behav_data/sub-01/ses-001/run_framewise/sub-01_ses-001_run-04_desc-framewise_merged.tsv'
    mastersheet_path = '/home/ozvar/Git/cneuromod/mario.scenes/sourcedata/scenes_info/scenes_mastersheet.csv'
    
    processed_df = collect_and_prune_behav_features(
        filepath=filepath,
        mastersheet_path=mastersheet_path,
        event_durations_dict=PARAMETERS['event_frame_duration_dict'],
        columns_to_drop_dict=COLUMNS_TO_DROP
    )
    
    print(f"Processed dataframe shape: {processed_df.shape}")
    print(f"Columns: {processed_df.columns.tolist()}")
