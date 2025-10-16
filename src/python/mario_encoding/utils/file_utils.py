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
