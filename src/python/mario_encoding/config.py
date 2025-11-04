from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

PATHS = {
    'raw_data': PROJECT_DIR / '..' / 'mario.replays' / 'sourcedata' / 'mario',
    'replay_variables_jsons': PROJECT_DIR / '..' / 'mario.replays' / 'outputdata' / 'replays',
    'scene_clip_jsons': PROJECT_DIR / '..' / 'mario.scenes' / 'outputdata' / 'scene_clips',
    'bids_annotated_tsvs': PROJECT_DIR / '..' / 'mario.annotations' / 'outputdata' / 'annotated_events',
    'per_session_combined_repvars_tsvs': PROJECT_DIR / 'inputs' / 'per_session_combined_repvars_tsvs',
    'per_run_combined_repvars_and_bids_events': PROJECT_DIR / 'inputs' / 'per_run_combined_repvars_and_bids_events',
    'per_run_pruned_features': PROJECT_DIR / 'inputs' / 'per_run_pruned_features',
    'fmriprep_data': PROJECT_DIR / '..' / 'mario.fmriprep',
    'figures': PROJECT_DIR / 'results' / 'figures',
    'logs': PROJECT_DIR / 'results' / 'logs',
    'src_python': PROJECT_DIR / 'src' / 'python',
}

PARAMETERS = {
        'event_frame_duration_dict': {
            'event_coin_collected': 30,
            'event_brick_smashed': 30,
            'event_powerup_collected': 114,
            'event_hit_powerup_lost': 35,
            'event_hit_life_lost': 180
            },
        'TR': 1.49,
        'frame_rate': 60.099826520671044 #frame rate given by retro emulator.em.get_screen_rate()
        }

# Ensure non-data directories exist
PATHS['figures'].mkdir(parents=True, exist_ok=True)
PATHS['logs'].mkdir(parents=True, exist_ok=True)
