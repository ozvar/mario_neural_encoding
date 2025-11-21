"""Experiment configuration and tracking utilities."""
import json
from datetime import datetime
from pathlib import Path


def generate_experiment_id() -> str:
    """Generate timestamp-based experiment ID."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_experiment_config(feature_spaces: dict, delays: list, n_alphas: int,
                            alpha_min: float, alpha_max: float, 
                            baseline_filtering: bool) -> tuple:
    """
    Create experiment configuration dictionary.
    
    Parameters:
    -----------
    feature_spaces : dict
        Dictionary mapping feature space names to lists of feature names
    delays : list
        FIR delays in TRs
    n_alphas : int
    alpha_min : float
    alpha_max : float
    baseline_filtering : bool
        
    Returns:
    --------
    config : dict
    experiment_id : str
    """
    experiment_id = generate_experiment_id()
    config = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'feature_spaces': feature_spaces,
        'encoding_model': {
            'delays': delays,
            'n_alphas': n_alphas,
            'alpha_min': alpha_min,
            'alpha_max': alpha_max
        },
        'preprocessing': {
            'baseline_filtering': baseline_filtering
        }
    }
    return config, experiment_id


def save_experiment_config(config: dict, output_dir: Path) -> Path:
    """
    Save experiment configuration to JSON file.
    
    Parameters:
    -----------
    config : dict
    output_dir : Path
        
    Returns:
    --------
    config_path : Path
    """
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    return config_path


def load_experiment_config(output_dir: Path) -> dict:
    """
    Load experiment configuration from JSON file.
    
    Parameters:
    -----------
    output_dir : Path
        
    Returns:
    --------
    config : dict
    """
    config_path = output_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
