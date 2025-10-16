from pathlib import Path
from config import PATHS


def get_subject_session_path(subject, session, datatype='func'):
    """Get subject/session directory path."""
    return PATHS['raw_data'] / f'sub-{subject:02d}' / f'ses-{session:03d}' / datatype


def get_gamelog_path(subject, session, level, rep):
    """Get gamelog file path."""
    base = get_subject_session_path(subject, session, 'gamelogs')
    return base / f'sub-{subject:02d}_ses-{session:03d}_task-mario_level-{level}_rep-{rep:03d}.bk2'


def get_events_path(subject, session, run):
    """Get events TSV file path."""
    base = get_subject_session_path(subject, session, 'func')
    return base / f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run:02d}_events.tsv'


def find_gamelogs(subject=None, session=None, level=None):
    """Find gamelogs matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    level_pattern = f'*_level-{level}_*.bk2' if level else '*.bk2'
    
    search_path = PATHS['raw_data'] / sub_pattern / ses_pattern / 'gamelogs'
    return list(search_path.parent.glob(f'gamelogs/{level_pattern}'))
