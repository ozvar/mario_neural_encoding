import pandas as pd
from mario_encoding.config import PATHS
from mario_encoding.utils import setup_logger


def check_file_status(filepath):
    """Check if file path exists and has actual content (not just git-annex symlink)."""
    if not filepath.exists():
        return False, False
    path_exists = True
    try:
        size = filepath.stat().st_size
        content_available = size > 0
    except (OSError, FileNotFoundError):
        content_available = False
    return path_exists, content_available


def find_surface_bold(subject=None, session=None, run=None, fmriprep_path=PATHS['fmriprep_data']):
    """Find surface BOLD dtseries files matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'*_run-{run}_*' if run else '*_run-*'
    pattern = f'{sub_pattern}/{ses_pattern}/func/{run_pattern}space-fsLR_den-91k_bold.dtseries.nii'
    return list(fmriprep_path.glob(pattern))


def find_confounds(subject=None, session=None, run=None, fmriprep_path=PATHS['fmriprep_data']):
    """Find confounds TSV files matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'*_run-{run}_*' if run else '*_run-*'
    pattern = f'{sub_pattern}/{ses_pattern}/func/{run_pattern}desc-confounds_timeseries.tsv'
    return list(fmriprep_path.glob(pattern))


def check_fmri_files(subjects, fmriprep_path):
    """Check fMRI file existence and content availability for subjects."""
    results = []
    for subject in subjects:
        bold_paths = find_surface_bold(subject=subject, fmriprep_path=fmriprep_path)
        for bold_path in bold_paths:
            parts = bold_path.stem.split('_')
            session = None
            run = None
            for part in parts:
                if part.startswith('ses-'):
                    session = int(part.split('-')[1])
                elif part.startswith('run-'):
                    run = int(part.split('-')[1])
            if session is None or run is None:
                continue
            path_exists, content_available = check_file_status(bold_path)
            results.append({
                'subject': subject,
                'session': session,
                'run': run,
                'file_type': 'surface_bold',
                'filepath': str(bold_path),
                'path_exists': path_exists,
                'content_available': content_available
            })
        confound_paths = find_confounds(subject=subject, fmriprep_path=fmriprep_path)
        for confound_path in confound_paths:
            parts = confound_path.stem.split('_')
            session = None
            run = None
            for part in parts:
                if part.startswith('ses-'):
                    session = int(part.split('-')[1])
                elif part.startswith('run-'):
                    run = int(part.split('-')[1])
            if session is None or run is None:
                continue
            path_exists, content_available = check_file_status(confound_path)
            results.append({
                'subject': subject,
                'session': session,
                'run': run,
                'file_type': 'confounds',
                'filepath': str(confound_path),
                'path_exists': path_exists,
                'content_available': content_available
            })
    return pd.DataFrame(results)


def log_fmri_file_status(subjects, logs_path, fmriprep_path, log_name='fmri_files_status'):
    """Log fMRI file status showing path existence and content availability."""
    logger = setup_logger(str(logs_path), log_name)
    df = check_fmri_files(subjects, fmriprep_path)
    logger.info("fMRI Files Status (Surface BOLD + Confounds)")
    logger.info("=" * 90)
    for subject in sorted(df['subject'].unique()):
        subject_df = df[df['subject'] == subject]
        logger.info(f"\nSubject {subject}")
        logger.info("-" * 90)
        for session in sorted(subject_df['session'].unique()):
            session_df = subject_df[subject_df['session'] == session]
            logger.info(f"  Session {session}:")
            for file_type in ['surface_bold', 'confounds']:
                type_df = session_df[session_df['file_type'] == file_type]
                n_total = len(type_df)
                n_paths = type_df['path_exists'].sum()
                n_content = type_df['content_available'].sum()
                logger.info(f"    {file_type}: {n_content}/{n_total} available")
        # Per-subject summary
        subj_total = len(subject_df)
        subj_paths = subject_df['path_exists'].sum()
        subj_content = subject_df['content_available'].sum()
        logger.info(f"  Subject {subject} Total: {subj_content}/{subj_total} files available ({subj_content/subj_total*100:.1f}%)")
    logger.info("\n" + "=" * 90)
    logger.info("Overall Summary")
    logger.info("=" * 90)
    for file_type in ['surface_bold', 'confounds']:
        type_df = df[df['file_type'] == file_type]
        total = len(type_df)
        paths_exist = type_df['path_exists'].sum()
        content_avail = type_df['content_available'].sum()
        logger.info(f"{file_type}:")
        logger.info(f"  Total files: {total}")
        logger.info(f"  Paths exist: {paths_exist}/{total} ({paths_exist/total*100:.1f}%)")
        logger.info(f"  Content available: {content_avail}/{total} ({content_avail/total*100:.1f}%)")


if __name__ == '__main__':
    log_fmri_file_status(
        subjects=range(1, 7),
        logs_path=PATHS['logs'],
        fmriprep_path=PATHS['fmriprep_data']
    )
