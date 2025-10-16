import pandas as pd

from mario_encoding.config import PATHS
from mario_encoding.utils import find_events, setup_logger


def check_bk2_files(subject):
    """Check BK2 file existence for a subject across all sessions.
    
    Returns DataFrame with columns: session, bk2_file, exists
    """
    # Find all event TSVs for this subject
    event_files = find_events(subject=subject)
    results = []
    for event_file in event_files:
        # Extract session from filename (e.g., ses-001)
        session = int(event_file.stem.split('_')[1].split('-')[1])
        # Read TSV
        df = pd.read_csv(event_file, sep='\t')
        # Get unique BK2 files from stim_file column
        bk2_files = df['stim_file'].dropna().unique()
        for bk2_rel_path in bk2_files:
            # Construct full path
            bk2_full_path = PATHS['raw_data'] / bk2_rel_path
            results.append({
                'subject': subject,
                'session': session,
                'bk2_file': bk2_rel_path,
                'full_path': str(bk2_full_path),
                'exists': bk2_full_path.exists()
            })
    
    return pd.DataFrame(results)


def log_event_file_counts(subjects, logs_path, raw_data_path, log_name='event_file_counts'):
    """Log count of event files for each subject.
    
    Args:
        subjects: Iterable of subject numbers (e.g., range(1, 7))
        logs_path: Path to logs directory
        raw_data_path: Path to raw data directory
        log_name: Name for log file (default: 'event_file_counts')
    """
    logger = setup_logger(str(logs_path), log_name)
    results = []
    for subject in subjects:
        event_files = find_events(subject=subject, raw_data_path=raw_data_path)
        results.append({
            'subject': subject,
            'n_event_files': len(event_files)
        })
    # Create table
    df = pd.DataFrame(results)
    # Log header
    logger.info("subject | n_event_files")
    logger.info("-" * 25)
    # Log each row
    for _, row in df.iterrows():
        logger.info(f"{row['subject']:<7} | {row['n_event_files']}")
    # Log summary
    logger.info("-" * 25)
    logger.info(f"Total: {df['n_event_files'].sum()} event files across {len(subjects)} subjects")


if __name__ == '__main__':
    log_event_file_counts(
            subjects=range(1, 7),
            logs_path=PATHS['logs'],
            raw_data_path=PATHS['raw_data']
            )
