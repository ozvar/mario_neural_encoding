import pandas as pd
from pathlib import Path
from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.utils import pandas_styleset


metadata_columns_to_drop = [
        'filename',
        'subject',
        'session',
        'actions',
        'rep_id',
        'phase'
        ]

timing_columns_to_drop = [
        'frame_index',
        'rep_onset',
        'rep_duration',
        'frame_time_in_run',
        ]

unused_button_columns_to_drop = [
        'null',
        'SELECT',
        'START'
        ]

level_descriptor_columns_to_drop = [
        'world',
        'stage',
        'level',
        'area',
        'scene',
        'level_layout',
        'levelHi',
        'levelLo'
        ]


position_columns_to_drop = [
        'player_x_posHi',
        'player_x_posLo',
        'player_y_screen',
        'player_y_pos',
        'xscrollLo',
        'xscrollHi'
        ]


redundant_status_columns = [
        'powerup_appear', # Unreliable/broken
        'lives', # not visible on screen
        'player_sprite', # unclear, likely redundant with status events
        'player_state',
        'walk_animation',
        'star_timer' # why would we need this?
]


level_info_columns = [
        # constantly onscreen level/score info
        'score',
        'coins',
        'level',
        'time' # this one might be problematic
        ]


not_sure_but_drop = [
        'fireball_counter' # haven't encountered non-zero instance of this yet
        ]


redundant_scene_features = [
        'x Hi (entry)',
        'x Lo (entry)',
        'x Hi (exit)',
        'x Lo (exit)',
        'Entry point',
        'Exit point',
        'Layout'
        ]


def binarize_power_states(df):
    # Create binary indicators for each power state, otherwise mario is small
    df['mario_big'] = (df['powerstate'] == 1000000).astype(int)
    df['mario_fire'] = (df['powerstate'] == 2000000).astype(int)
    df['mario_star'] = (df['powerstate'] == 10000).astype(int)  # Invincibility
    df = df.drop(['powerstate'], axis=1) # Redundant with indicators above

    return df


def binarize_scrolling(df):
    """Convert scrolling to binary: 1 for active scrolling (17, 21), 0 for stationary (16, 20)"""
    df['screen_scrolling'] = df['scrolling'].isin([17, 21]).astype(int)
    df = df.drop(['scrolling'], axis=1)

    return df


def binarize_powerup_on_screen(df):
    df['powerup_visible'] = df['powerup_yes_no'].isin([46, 48]).astype(int)
    df = df.drop('powerup_yes_no', axis=1)

    return df


def binarize_jump_airborne(df):
    df['is_airborne'] = (df['jump_airborne'] > 0).astype(int)
    df = df.drop('jump_airborne', axis=1)

    return df


def one_hot_moving_direction(df):
    # absence of either implies mario is stationary
    df['moving_right'] = (df['moving_direction'] == 1).astype(int)
    df['moving_left'] = (df['moving_direction'] == 2).astype(int)
    df = df.drop(['moving_direction'], axis=1)

    return df


def extend_event_duration(df, event_col, n_frames):
    """Extend event by specified number of frames and return new df"""
    if not event_col in df.columns:
        return df
    events = df[df[event_col] == 1].index
    if events.size == 0:
        return df
    else:
        for event_idx in events:
            if df.index[-1] > event_idx+n_frames:
                df.loc[event_idx:event_idx + n_frames - 1, event_col] = 1

        return df


def extend_event_durations_in_events_dict(df, event_durations_dict):
    for event_type in event_durations_dict:
        df = extend_event_duration(
                df,
                event_col=event_type,
                n_frames=event_durations_dict[event_type]
                )
    return df


def merge_scene_features(df, mastersheet_path):
    # Parse df['scene'] into World, Level, Scene components
    df['World'] = df['scene'].str.extract(r'w(\d+)').astype('Int64')  # Int64 handles NaN
    df['Level'] = df['scene'].str.extract(r'l(\d+)').astype('Int64')
    df['Scene'] = df['scene'].str.extract(r's(\d+)').astype('Int64')
    # Load mastersheet
    scenes_df = pd.read_csv(mastersheet_path)
    # Drop rows where World/Level/Scene are NaN (summary rows)
    scenes_df = scenes_df.dropna(subset=['World', 'Level', 'Scene'])
    scene_feature_cols = [col for col in scenes_df.columns 
                         if col not in ['World', 'Level', 'Scene']]
    # Merge on World, Level, Scene
    df_with_scenes = df.merge(
        scenes_df, 
        on=['World', 'Level', 'Scene'], 
        how='left'
    )
    df_with_scenes[scene_feature_cols] = df_with_scenes[scene_feature_cols].fillna(0).astype(int)
    # Drop the helper features
    df_with_scenes = df_with_scenes.drop(['World', 'Level', 'Scene'], axis=1)

    return df_with_scenes


if __name__ == '__main__':
    fp = '/home/ozvar/Git/cneuromod/mario_neural_encoding/inputs/preprocessed_behav_data/sub-01/ses-001/run_framewise/sub-01_ses-001_run-04_desc-framewise_merged.tsv'
    mastersheet_path = '/home/ozvar/Git/cneuromod/mario.scenes/sourcedata/scenes_info/scenes_mastersheet.csv'
    df = pd.read_csv(fp, sep='\t')
    # Merge df with scene features from scenes mastersheet
    df = merge_scene_features(df, mastersheet_path)
    # Binarize mario powerstate so we know what kind of mario is on screen
    df = binarize_power_states(df)
    # Binarize scrolling state
    df = binarize_scrolling(df)
    # Binarize powerup appearance
    df = binarize_powerup_on_screen(df)
    # Binarize jump airborne (>0 means mario isn't grounded)
    df = binarize_jump_airborne(df)
    # One-hot encode moving direction
    df = one_hot_moving_direction(df)
    # Create copy of dataframe and drop unnecessary columns to play with feature space
    test = df.copy()
    test = test.drop(metadata_columns_to_drop, axis=1)
    test = test.drop(timing_columns_to_drop, axis=1)
    test = test.drop(unused_button_columns_to_drop, axis=1)
    test = test.drop(level_descriptor_columns_to_drop, axis=1)
    test = test.drop(position_columns_to_drop, axis=1)
    test = test.drop(redundant_status_columns, axis=1)
    test = test.drop(redundant_scene_features, axis=1)
    test = test.drop(not_sure_but_drop, axis=1)
    # Create extended kill events from enemy_kill columns
    test = extend_event_durations_in_events_dict(df=test, event_durations_dict=PARAMETERS['event_frame_duration_dict'])
    # Stomp (value 4)
    test['event_kill_stomp'] = ((test['enemy_kill30'] == 4) | 
                                (test['enemy_kill31'] == 4) |
                                (test['enemy_kill32'] == 4) |
                                (test['enemy_kill33'] == 4) |
                                (test['enemy_kill34'] == 4)).astype(int)
    # Impact (value 34)
    test['event_kill_impact'] = ((test['enemy_kill30'] == 34) | 
                                 (test['enemy_kill31'] == 34) |
                                 (test['enemy_kill32'] == 34) |
                                 (test['enemy_kill33'] == 34) |
                                 (test['enemy_kill34'] == 34)).astype(int)

    # Kick (value 132)
    test['event_kill_kick'] = ((test['enemy_kill30'] == 132) | 
                               (test['enemy_kill31'] == 132) |
                               (test['enemy_kill32'] == 132) |
                               (test['enemy_kill33'] == 132) |
                               (test['enemy_kill34'] == 132)).astype(int)
    # Drop original enemy_kill columns (no longer needed)
    enemy_kill_cols_to_drop = ['enemy_kill30', 'enemy_kill31', 'enemy_kill32',
                                'enemy_kill33', 'enemy_kill34', 'enemy_kill35']
    test = test.drop(enemy_kill_cols_to_drop, axis=1)
