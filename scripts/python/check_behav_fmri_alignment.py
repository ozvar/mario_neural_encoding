import pandas as pd
from mario_encoding.config import PATHS
from mario_encoding.utils import setup_logger
from check_behav_data_files_exist import check_event_files, check_bk2_files
from check_fmri_data_files_exist import check_fmri_files


def get_available_runs(df, file_type=None):
    """Extract set of (session, run) tuples for available files."""
    if file_type:
        df = df[df['file_type'] == file_type]
    available = df[df['content_available'] == True]
    return set(zip(available['session'], available['run']))


def check_bk2_completeness(subject, session, bk2_df):
    """Check if all BK2s are available for a subject/session."""
    session_bk2s = bk2_df[(bk2_df['subject'] == subject) & (bk2_df['session'] == session)]
    if len(session_bk2s) == 0:
        return True  # No BK2s expected, consider complete
    return session_bk2s['content_available'].all()


def log_alignment_status(subjects, logs_path, raw_data_path, fmriprep_path, log_name='behavioral_fmri_alignment'):
    """Log alignment between behavioral and fMRI data with BK2 completeness flags."""
    logger = setup_logger(str(logs_path), log_name)
    behav_df = check_event_files(subjects, raw_data_path)
    fmri_df = check_fmri_files(subjects, fmriprep_path)
    bk2_df = check_bk2_files(subjects, raw_data_path)
    logger.info("Behavioral-fMRI Alignment Status")
    logger.info("=" * 90)
    all_matched = []
    all_matched_incomplete_bk2 = []
    all_behav_only = []
    all_fmri_only = []
    for subject in subjects:
        subj_behav = behav_df[behav_df['subject'] == subject]
        subj_fmri = fmri_df[fmri_df['subject'] == subject]
        subj_bk2 = bk2_df[bk2_df['subject'] == subject]
        behav_runs = get_available_runs(subj_behav)
        fmri_runs = get_available_runs(subj_fmri, file_type='surface_bold')
        matched = behav_runs & fmri_runs
        behav_only = behav_runs - fmri_runs
        fmri_only = fmri_runs - behav_runs
        matched_complete = []
        matched_incomplete = []
        for session, run in matched:
            bk2_complete = check_bk2_completeness(subject, session, subj_bk2)
            if bk2_complete:
                matched_complete.append((session, run))
                all_matched.append((subject, session, run))
            else:
                matched_incomplete.append((session, run))
                all_matched_incomplete_bk2.append((subject, session, run))
        all_behav_only.extend([(subject, s, r) for s, r in behav_only])
        all_fmri_only.extend([(subject, s, r) for s, r in fmri_only])
        logger.info(f"\nSubject {subject}")
        logger.info("-" * 90)
        logger.info(f"  Matched runs (complete): {len(matched_complete)}")
        logger.info(f"  Matched runs (missing BK2s): {len(matched_incomplete)}")
        logger.info(f"  Behavioral only (missing fMRI): {len(behav_only)}")
        logger.info(f"  fMRI only (missing behavioral): {len(fmri_only)}")
        if matched_incomplete:
            logger.info(f"    Matched but incomplete BK2s:")
            for session, run in sorted(matched_incomplete):
                logger.info(f"      Session {session}, Run {run} (⚠ see bk2_files_status.txt)")
        if behav_only:
            logger.info(f"    Behavioral-only runs:")
            for session, run in sorted(behav_only):
                logger.info(f"      Session {session}, Run {run}")
        if fmri_only:
            logger.info(f"    fMRI-only runs:")
            for session, run in sorted(fmri_only):
                logger.info(f"      Session {session}, Run {run}")
        total_runs = len(matched) + len(behav_only) + len(fmri_only)
        match_pct = (len(matched) / total_runs * 100) if total_runs > 0 else 0
        logger.info(f"  Subject {subject} match rate: {len(matched)}/{total_runs} ({match_pct:.1f}%)")
    logger.info("\n" + "=" * 90)
    logger.info("Overall Summary")
    logger.info("=" * 90)
    total_matched_complete = len(all_matched)
    total_matched_incomplete = len(all_matched_incomplete_bk2)
    total_behav_only = len(all_behav_only)
    total_fmri_only = len(all_fmri_only)
    total_all = total_matched_complete + total_matched_incomplete + total_behav_only + total_fmri_only
    logger.info(f"Total matched runs (complete): {total_matched_complete}")
    logger.info(f"Total matched runs (missing BK2s): {total_matched_incomplete}")
    logger.info(f"Total behavioral-only runs: {total_behav_only}")
    logger.info(f"Total fMRI-only runs: {total_fmri_only}")
    logger.info(f"Overall match rate: {total_matched_complete + total_matched_incomplete}/{total_all} ({(total_matched_complete + total_matched_incomplete)/total_all*100:.1f}%)")
    logger.info(f"\nRecommendation: Run 'datalad get' on {total_matched_complete + total_matched_incomplete} matched fMRI runs")
    if total_matched_incomplete > 0:
        logger.info(f"Note: {total_matched_incomplete} runs have incomplete BK2 files - check bk2_files_status.txt")


if __name__ == '__main__':
    log_alignment_status(
        subjects=range(1, 7),
        logs_path=PATHS['logs'],
        raw_data_path=PATHS['raw_data'],
        fmriprep_path=PATHS['fmriprep_data']
    )
