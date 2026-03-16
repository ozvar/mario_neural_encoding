from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

PATHS = {
    # Raw data directories
    'raw_data': PROJECT_DIR / '..' / 'mario.replays' / 'sourcedata' / 'mario',
    'replay_variables_jsons': PROJECT_DIR / '..' / 'mario.replays' / 'outputdata' / 'replays',
    'scene_clip_jsons': PROJECT_DIR / '..' / 'mario.scenes' / 'outputdata' / 'scene_clips',
    'bids_annotated_tsvs': PROJECT_DIR / '..' / 'mario.annotations' / 'outputdata' / 'annotated_events',
    # Preprocessed data directories
    'constants': PROJECT_DIR / 'inputs' / 'constants',
    'per_session_combined_repvars_tsvs': PROJECT_DIR / 'inputs' / 'per_session_combined_repvars_tsvs',
    'per_run_combined_repvars_and_bids_events': PROJECT_DIR / 'inputs' / 'per_run_combined_repvars_and_bids_events',
    'per_run_pruned_features': PROJECT_DIR / 'inputs' / 'per_run_pruned_features',
    'per_run_downsampled_to_TR': PROJECT_DIR / 'inputs' / 'per_run_downsampled_to_TR',
    'practice_phase_metadata': PROJECT_DIR / 'inputs' / 'practice_phase_metadata',
    'fmriprep_data': PROJECT_DIR / '..' / 'mario.fmriprep',
    # Results directories
    'variance_partitioning': PROJECT_DIR / 'results' / 'variance_partitioning',
    'pca': PROJECT_DIR / 'results' / 'pca',
    'cv_scores': PROJECT_DIR / 'results' / 'cv_scores',
    'figures': PROJECT_DIR / 'results' / 'figures',
    'figure_assets': PROJECT_DIR / 'results' / 'figures' / 'assets',
    'pycortex': PROJECT_DIR / 'results' / 'figures' / 'pycortex',
    'neurosynth': PROJECT_DIR / 'results' / 'neurosynth_decoding',
    # Logs directories
    'logs': PROJECT_DIR / 'results' / 'logs',
    'fit_logs': PROJECT_DIR / 'results' / 'logs',
    'src_python': PROJECT_DIR / 'src' / 'python',
    # Cifti directories
    'Yeo7_networks': '/home/ozvar/Atlases/HCP_S1200_Atlas/Yeo7_Yeo-Buckner-Choi-Raut.dlabel.nii',
    'ColeAnticevic_networks': '/home/ozvar/Atlases/HCP_S1200_Atlas/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_netassignments_LR.dlabel.nii'
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
    'frame_rate': 60.099826520671044,  # frame rate given by retro emulator.em.get_screen_rate()
    
    # Encoding model parameters
    'encoding_model': {
        'delays': [1, 2, 3, 4, 5],           # FIR delays (in TRs)
        'n_alphas': 20,                   # Number of alpha values to test
        'alpha_min': 1,                   # Minimum alpha: 10^alpha_min
        'alpha_max': 20,                  # Maximum alpha: 10^alpha_max
        'solver_params': {
            'n_targets_batch': 500,       # Batch size for grayordinates (GPU memory)
            'n_alphas_batch': 5,          # Batch size for alphas
            'n_targets_batch_refit': 500  # Batch size for refit
        }
    },
    # Variance partitioning (using banded ridge regression) parameters
    'variance_partitioning': {
        'delays': [1, 2, 3, 4, 5],  # Same as encoding_model
        'solver': 'random_search',
        'solver_params': {
            'n_iter': 100,  # Random search iterations
            'n_targets_batch': 500,  # GPU memory management
            'alphas': np.logspace(1, 20, 20)
        }
    }
}

FEATURE_SPACES = {
    'motor': [
        # Button inputs
        'B', 'A', 'RIGHT', 'LEFT', 'UP', 'DOWN',
    ],
    #'actions': [
    #    # Kinematics
    #    'is_airborne', 'moving_right', 'moving_left',
    #    # Scrolling state
    #    'screen_scrolling'
    #],
    'scene': [
        # Enemy presence/density (coarse scene features)
        'Enemy', '2-Horde', '3-Horde', '4-Horde',
        # Obstacles
        'Roof', 'Gap', 'Multiple gaps', 'Variable gaps', 'Gap enemy', 'Pillar gap',
        # Valleys
        'Valley', 'Pipe valley', 'Empty valley', 'Enemy valley', 'Roof valley',
        # Navigation contexts
        'Stair up', 'Stair down', 'Empty stair valley', 'Enemy stair valley', 'Gap stair valley',
        # Decision affordances
        '2-Path', '3-Path', 'Risk/Reward',
        # Location markers
        'Reward', 'Moving platform', 'Flagpole', 'Beginning', 'Bonus zone', 'Waterworld', 'Checkpoint',
        # Mario powerup states (ambient perceptual)
        'mario_big', 'mario_fire', 'mario_star',
        # Powerup visibility
        'powerup_visible',
        # Enemy visibility (framewise perceptual)
        'enemy_drawn19', 'enemy_drawn17', 'enemy_drawn18', 'enemy_drawn16', 'enemy_drawn15'
    ],
    #'activity': [
    #    # Reward acquisition events
    #    'event_brick_smashed', 'event_coin_collected', 'event_powerup_collected',
    #    # Damage/failure events
    #    'event_hit_life_lost', 'event_hit_powerup_lost',
    #    # Enemy elimination events
    #    'event_kill_stomp', 'event_kill_impact', 'event_kill_kick'
    #]
}

# Ensure non-data directories exist
PATHS['figures'].mkdir(parents=True, exist_ok=True)
PATHS['logs'].mkdir(parents=True, exist_ok=True)
PATHS['models'].mkdir(parents=True, exist_ok=True)
PATHS['variance_partitioning'].mkdir(parents=True, exist_ok=True)
PATHS['pca'].mkdir(parents=True, exist_ok=True)
PATHS['cv_scores'].mkdir(parents=True, exist_ok=True)
