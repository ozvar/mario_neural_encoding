from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

PATHS = {
    'raw_data': PROJECT_DIR / '..' / 'mario.replays' / 'sourcedata' / 'mario',
    'fmriprep_data': PROJECT_DIR / '..' / 'mario.fmriprep',
    'figures': PROJECT_DIR / 'results' / 'figures',
    'logs': PROJECT_DIR / 'results' / 'logs',
    'src_python': PROJECT_DIR / 'src' / 'python',
}

PARAMETERS = {}

# Ensure non-data directories exist
PATHS['figures'].mkdir(parents=True, exist_ok=True)
PATHS['logs'].mkdir(parents=True, exist_ok=True)
