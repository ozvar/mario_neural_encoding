import pandas as pd
from mario_encoding.config import PATHS
from mario_encoding.utils import find_events, setup_logger


def check_file_status(filepath):
    """Check if file path exists and has actual content (not just git-annex symlink)."""
    if not filepath.exists():
        return False, False
    path_exists = True
    # Check if file has actual content by trying to read it
    # Git-annex retrieved files are symlinks but readable
    # Git-annex non-retrieved files are broken symlinks
    try:
        size = filepath.stat().st_size
        content_available = size > 0
    except (OSError, FileNotFoundError):
        # Broken symlink or inaccessible
        content_available = False
    
    return path_exists, content_available


def check_event_files(subjects, raw_data_path):
    """Check event file existence and content availability for subjects."""
    results = []
    for subject in subjects:
        event_paths = find_events(subject=subject, raw_data_path=raw_data_path)
        for event_path in event_paths:
            parts = event_path.stem.split('_')
            # Parse session
            session = None
            for part in parts:
                if part.startswith('ses-'):
                    session = int(part.split('-')[1])
                    break
            # Parse run
            run = None
            for part in parts:
                if part.startswith('run-'):
                    run = int(part.split('-')[1])
                    break
            if session is None or run is None:
                continue  # Skip malformed filenames
            path_exists, content_available = check_file_status(event_path)
            results.append({
                'subject': subject,
                'session': session,
                'run': run,
                'filepath': str(event_path),
                'path_exists': path_exists,
                'content_available': content_available
            })
    return pd.DataFrame(results)


def check_bk2_files(subjects, raw_data_path):
    """Check BK2 file existence based on available event files."""
    results = []
    for subject in subjects:
        event_paths = find_events(subject=subject, raw_data_path=raw_data_path)
        for event_path in event_paths:
            _, content_available = check_file_status(event_path)
            if not content_available:
                continue
            session = int(event_path.stem.split('_')[1].split('-')[1])
            df = pd.read_csv(event_path, sep='\t')
            bk2_files = df['stim_file'].dropna().unique()
            for bk2_rel_path in bk2_files:
                bk2_full_path = raw_data_path / bk2_rel_path
                path_exists, bk2_content = check_file_status(bk2_full_path)
                results.append({
                    'subject': subject,
                    'session': session,
                    'bk2_file': bk2_rel_path,
                    'path_exists': path_exists,
                    'content_available': bk2_content
                })
    return pd.DataFrame(results)


def log_event_file_status(subjects, logs_path, raw_data_path, log_name='event_files_status'):
    """Log event file status showing path existence and content availability."""
    logger = setup_logger(str(logs_path), log_name)
    df = check_event_files(subjects, raw_data_path)
    logger.info("Event Files Status")
    logger.info("=" * 80)
    logger.info(f"{'Subject':<8} | {'Session':<8} | {'Run':<5} | {'Path Exists':<12} | {'Content Available':<18}")
    logger.info("-" * 80)
    for _, row in df.iterrows():
        path_status = "YES" if row['path_exists'] else "NO"
        content_status = "YES" if row['content_available'] else "NO"
        logger.info(f"{row['subject']:<8} | {row['session']:<8} | {row['run']:<5} | {path_status:<12} | {content_status:<18}")
    logger.info("-" * 80)
    total = len(df)
    paths_exist = df['path_exists'].sum()
    content_avail = df['content_available'].sum()
    logger.info(f"Total event files found: {total}")
    logger.info(f"Paths exist: {paths_exist}/{total} ({paths_exist/total*100:.1f}%)")
    logger.info(f"Content available: {content_avail}/{total} ({content_avail/total*100:.1f}%)")


def log_bk2_file_status(subjects, logs_path, raw_data_path, log_name='bk2_files_status'):
    """Log BK2 file status based on available event files."""
    logger = setup_logger(str(logs_path), log_name)
    df = check_bk2_files(subjects, raw_data_path)
    if df.empty:
        logger.info("No event files with content available to check BK2 files")
        return
    logger.info("BK2 Files Status (based on available event files)")
    logger.info("=" * 80)
    for subject in sorted(df['subject'].unique()):
        subject_df = df[df['subject'] == subject]
        logger.info(f"\nSubject {subject}")
        logger.info("-" * 80)
        for session in sorted(subject_df['session'].unique()):
            session_df = subject_df[subject_df['session'] == session]
            n_total = len(session_df)
            n_paths = session_df['path_exists'].sum()
            n_content = session_df['content_available'].sum()
            logger.info(f"Session {session}: {n_content}/{n_total} BK2 files available")
            missing_content = session_df[~session_df['content_available']]
            for _, row in missing_content.iterrows():
                status = "Path missing" if not row['path_exists'] else "Content not retrieved"
                logger.info(f"  [{status}] {row['bk2_file']}")
    logger.info("\n" + "=" * 80)
    logger.info("Overall Summary")
    logger.info("=" * 80)
    total = len(df)
    paths_exist = df['path_exists'].sum()
    content_avail = df['content_available'].sum()
    logger.info(f"Total BK2 files referenced: {total}")
    logger.info(f"Paths exist: {paths_exist}/{total} ({paths_exist/total*100:.1f}%)")
    logger.info(f"Content available: {content_avail}/{total} ({content_avail/total*100:.1f}%)")


if __name__ == '__main__':
    log_event_file_status(
        subjects=range(1, 7),
        logs_path=PATHS['logs'],
        raw_data_path=PATHS['raw_data']
    )
    log_bk2_file_status(
        subjects=range(1, 7),
        logs_path=PATHS['logs'],
        raw_data_path=PATHS['raw_data']
    )
