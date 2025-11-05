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
    'per_run_downsampled_to_TR': PROJECT_DIR / 'inputs' / 'per_run_downsampled_to_TR',
    'practice_phase_metadata': PROJECT_DIR / 'inputs' / 'practice_phase_metadata',
    'fmriprep_data': PROJECT_DIR / '..' / 'mario.fmriprep',
    'figures': PROJECT_DIR / 'results' / 'figures',
    'logs': PROJECT_DIR / 'results' / 'logs',
    'models': PROJECT_DIR / 'results' / 'models',
    'cv_scores': PROJECT_DIR / 'results' / 'cv_scores',
    'src_python': PROJECT_DIR / 'src' / 'python',
}

PARAMETERS = {
        # n frames to extend each BIDS-like event annotation by
        # these are determined programmatically and/or via visual inspection of replay files for now 
        'event_frame_duration_dict': {
            'event_coin_collected': 30,
            'event_brick_smashed': 30,
            'event_powerup_collected': 114,
            'event_hit_powerup_lost': 35,
            'event_hit_life_lost': 180
            },

        'TR': 1.49,
        'frame_rate': 60.099826520671044, #frame rate given by retro emulator.em.get_screen_rate()

        # Encoding model parameters
        'encoding_model': {
            'delays': [1, 2, 3, 4],          # FIR delays (TRs)
            'n_alphas': 20,                   # Number of alphas to test
            'alpha_min': 1,                   # Min alpha: 10^1
            'alpha_max': 20,                  # Max alpha: 10^20
            'solver_params': {
                'n_targets_batch': 2000,      # Batch size for grayordinates
                'n_alphas_batch': 5,          # Batch size for alphas
                'n_targets_batch_refit': 2000 # Batch size for refit
                }
            }
}

# Ensure non-data directories exist
PATHS['figures'].mkdir(parents=True, exist_ok=True)
PATHS['logs'].mkdir(parents=True, exist_ok=True)
