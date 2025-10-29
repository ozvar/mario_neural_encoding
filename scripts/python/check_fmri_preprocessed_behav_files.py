import pandas as pd
from pathlib import Path
from mario_encoding.config import PATHS
from mario_encoding.utils import setup_logger


def find_fmri_runs(subject, fmriprep_path=PATHS['fmriprep_data']):
    """Find all surface BOLD runs for a subject."""
    pattern = f'sub-{subject:02d}/ses-*/func/*_space-fsLR_den-91k_bold.dtseries.nii'
    return list(fmriprep_path.glob(pattern))


def extract_session_run_from_fmri(filepath):
    """Extract session and run from fMRI filename."""
    stem = filepath.stem
    parts = stem.split('_')
    
    session = None
    run = None
    for part in parts:
        if part.startswith('ses-'):
            session = int(part.split('-')[1])
        elif part.startswith('run-'):
            run = int(part.split('-')[1])
    
    return session, run


def find_merged_tsv(subject, session, run, 
                    framewise_path=PATHS['per_session_framewise_tsvs']):
    """Check if merged TSV exists for a run."""
    from mario_encoding.utils.file_utils import get_run_framewise_path
    tsv_path = get_run_framewise_path(subject, session, run, framewise_path)
    return tsv_path.exists(), tsv_path


def check_fmri_tsv_alignment(subjects, fmriprep_path=PATHS['fmriprep_data'],
                             framewise_path=PATHS['per_session_framewise_tsvs']):
    """Check alignment between fMRI files and merged TSVs."""
    results = []
    for subject in subjects:
        fmri_files = find_fmri_runs(subject, fmriprep_path)
        for fmri_file in fmri_files:
            session, run = extract_session_run_from_fmri(fmri_file)
            if session is None or run is None:
                continue
            tsv_exists, tsv_path = find_merged_tsv(subject, session, run, framewise_path)
            
            results.append({
                'subject': subject,
                'session': session,
                'run': run,
                'fmri_file': fmri_file,
                'tsv_exists': tsv_exists,
                'tsv_path': tsv_path if tsv_exists else None
            })
    
    return pd.DataFrame(results)


def log_fmri_tsv_alignment(subjects, logs_path=PATHS['logs'],
                           fmriprep_path=PATHS['fmriprep_data'],
                           framewise_path=PATHS['per_session_framewise_tsvs'],
                           log_name='fmri_tsv_alignment'):
    """Log alignment status between fMRI and merged TSVs."""
    logger = setup_logger(str(logs_path), log_name)
    df = check_fmri_tsv_alignment(subjects, fmriprep_path, framewise_path)
    
    logger.info("fMRI-TSV Alignment Status")
    logger.info("=" * 80)
    
    for subject in subjects:
        subj_df = df[df['subject'] == subject]
        
        if len(subj_df) == 0:
            logger.info(f"\nSubject {subject}: No fMRI data found")
            continue
        
        matched = subj_df[subj_df['tsv_exists'] == True]
        missing = subj_df[subj_df['tsv_exists'] == False]
        
        logger.info(f"\nSubject {subject}")
        logger.info("-" * 80)
        logger.info(f"  Total fMRI runs: {len(subj_df)}")
        logger.info(f"  Matched with TSV: {len(matched)}")
        logger.info(f"  Missing TSV: {len(missing)}")
        
        if len(missing) > 0:
            logger.info(f"  Missing TSVs:")
            for _, row in missing.iterrows():
                logger.info(f"    Session {row['session']:03d}, Run {row['run']:02d}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Overall Summary")
    logger.info("=" * 80)
    
    total_fmri = len(df)
    total_matched = len(df[df['tsv_exists'] == True])
    total_missing = len(df[df['tsv_exists'] == False])
    
    logger.info(f"Total fMRI runs: {total_fmri}")
    logger.info(f"Total matched: {total_matched}")
    logger.info(f"Total missing: {total_missing}")
    
    if total_fmri > 0:
        match_pct = (total_matched / total_fmri) * 100
        logger.info(f"Match rate: {match_pct:.1f}%")
    
    return df


if __name__ == '__main__':
    df = log_fmri_tsv_alignment(subjects=range(1, 7))
    print(f"\nSummary: {len(df[df['tsv_exists']])} / {len(df)} fMRI runs have merged TSVs")
