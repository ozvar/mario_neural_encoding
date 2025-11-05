"""
Identify practice phase runs and generate metadata for cross-validation.

This script examines BIDS events files to identify which runs are from the practice
phase (vs discovery phase), and generates metadata needed for leave-one-run-out
cross-validation including run_onsets arrays.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from mario_encoding.config import PATHS


def get_annotated_events_path(subject, session, run, events_path=PATHS['bids_annotated_tsvs']):
    """Get annotated events TSV file path."""
    base = events_path / f'sub-{subject:02d}' / f'ses-{session:03d}' / 'func'
    return base / f'sub-{subject:02d}_ses-{session:03d}_task-mario_run-{run:02d}_desc-annotated_events.tsv'


def find_annotated_events(subject=None, session=None, run=None, events_path=PATHS['bids_annotated_tsvs']):
    """Find all annotated events TSV files matching criteria (None = wildcard)."""
    sub_pattern = f'sub-{subject:02d}' if subject else 'sub-*'
    ses_pattern = f'ses-{session:03d}' if session else 'ses-*'
    run_pattern = f'run-{run:02d}' if run else 'run-*'
    pattern = f'{sub_pattern}/{ses_pattern}/func/*{run_pattern}_desc-annotated_events.tsv'
    return list(events_path.glob(pattern))


def extract_run_from_events_filename(filepath):
    """Extract run number from events filename."""
    filename = filepath.stem
    if '_run-' in filename:
        try:
            run_part = filename.split('_run-')[1].split('_')[0]
            return int(run_part)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not extract run number from: {filename}") from e
    else:
        raise ValueError(f"No '_run-' pattern found in filename: {filename}")


def identify_practice_runs_from_bids(subject, session, events_path=PATHS['bids_annotated_tsvs']):
    """
    Check BIDS events files for phase column to identify practice runs.
    
    A run is considered practice phase if:
    - It has a 'phase' column
    - All non-null values in 'phase' column are 'practice'
    
    Parameters:
    -----------
    subject : int
        Subject number
    session : int
        Session number
    events_path : Path
        Path to BIDS annotated events directory
        
    Returns:
    --------
    practice_runs : list of int
        Sorted list of run numbers in practice phase
    run_phases : dict
        Dictionary mapping run number to phase ('practice', 'discovery', or 'mixed')
    """
    events_files = find_annotated_events(subject=subject, session=session, events_path=events_path)
    if not events_files:
        raise ValueError(f"No events files found for sub-{subject:02d}_ses-{session:03d}")
    
    practice_runs = []
    run_phases = {}
    
    for filepath in sorted(events_files):
        run = extract_run_from_events_filename(filepath)
        events = pd.read_csv(filepath, sep='\t')
        # Check if phase column exists
        if 'phase' not in events.columns:
            print(f"  Warning: Run {run:02d} has no 'phase' column, skipping")
            run_phases[run] = 'unknown'
            continue
        # Get unique phases (excluding NaN)
        phases = events['phase'].dropna().unique()
        if len(phases) == 0:
            print(f"  Warning: Run {run:02d} has no phase information, skipping")
            run_phases[run] = 'unknown'
        elif len(phases) == 1:
            phase = phases[0]
            run_phases[run] = phase
            if phase == 'practice':
                practice_runs.append(run)
        else:
            print(f"  Warning: Run {run:02d} has mixed phases: {phases}, skipping")
            run_phases[run] = 'mixed'
    
    return sorted(practice_runs), run_phases


def get_run_downsampled_path(subject, session, run, per_run_downsampled_path=PATHS['per_run_downsampled_to_TR']):
    """Get path for downsampled TSV."""
    base = per_run_downsampled_path / f'sub-{subject:02d}' / f'ses-{session:03d}'
    return base / f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_desc-downsampled.tsv'


def compute_run_onsets(subject, session, practice_runs, per_run_downsampled_path=PATHS['per_run_downsampled_to_TR']):
    """
    Compute run_onsets array from downsampled TSV files.
    
    run_onsets is an array of cumulative TR counts used for CV splitting
    and within-run z-scoring. Format: [0, n_TRs_run1, n_TRs_run1+run2, ...]
    
    Parameters:
    -----------
    subject : int
        Subject number
    session : int
        Session number
    practice_runs : list of int
        Run numbers to include
    per_run_downsampled_path : Path
        Path to downsampled data directory
        
    Returns:
    --------
    run_onsets : array of int
        Cumulative TR counts at start of each run
    n_samples : int
        Total TRs across all practice runs
    n_trs_per_run : list of int
        Number of TRs in each run
    """
    onsets = [0]
    n_trs_per_run = []
    total_trs = 0
    
    for run in practice_runs:
        filepath = get_run_downsampled_path(subject, session, run, per_run_downsampled_path)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Downsampled file not found: {filepath}")
        
        df = pd.read_csv(filepath, sep='\t')
        n_trs = len(df)
        n_trs_per_run.append(n_trs)
        total_trs += n_trs
        onsets.append(total_trs)
    
    # Return onsets without the final cumsum (VEM convention)
    return np.array(onsets[:-1]), total_trs, n_trs_per_run


def compute_level_onsets(subject, session, practice_runs, 
                         events_path=PATHS['bids_annotated_tsvs'],
                         per_run_downsampled_path=PATHS['per_run_downsampled_to_TR']):
    """
    Compute level_onsets array from BIDS events TSV files.
    
    level_onsets is an array of cumulative TR counts where each level starts,
    used for within-level z-scoring of behavioral features.
    
    Parameters:
    -----------
    subject : int
        Subject number
    session : int
        Session number
    practice_runs : list of int
        Run numbers to include
    events_path : Path
        Path to BIDS annotated events directory
    per_run_downsampled_path : Path
        Path to downsampled data directory
        
    Returns:
    --------
    level_onsets : array of int
        Cumulative TR indices where each level starts
    n_levels : int
        Total number of level repetitions across all practice runs
    """
    level_onsets = [0]
    current_tr = 0
    n_levels = 0
    
    for run in practice_runs:
        # Load events file
        events_filepath = get_annotated_events_path(subject, session, run, events_path)
        if not events_filepath.exists():
            raise FileNotFoundError(f"Events file not found: {events_filepath}")
        events = pd.read_csv(events_filepath, sep='\t')
        # Load downsampled file to get TRs per level
        downsampled_filepath = get_run_downsampled_path(subject, session, run, per_run_downsampled_path)
        if not downsampled_filepath.exists():
            raise FileNotFoundError(f"Downsampled file not found: {downsampled_filepath}")
        downsampled = pd.read_csv(downsampled_filepath, sep='\t')
        # Check if level column exists
        if 'level' not in downsampled.columns:
            raise ValueError(f"'level' column not found in {downsampled_filepath}")
        # Find level changes in downsampled data
        level_changes = downsampled['level'].ne(downsampled['level'].shift())
        level_change_indices = np.where(level_changes)[0]
        # Convert to cumulative TR indices
        for idx in level_change_indices[1:]:  # Skip first (already at 0)
            level_onsets.append(current_tr + idx)
            n_levels += 1
        # Update current_tr for next run
        current_tr += len(downsampled)
        n_levels += 1  # Count the last level in this run
    
    return np.array(level_onsets), n_levels


def get_practice_metadata_path(subject, session, practice_metadata_path=PATHS['practice_phase_metadata']):
    """Get path for practice metadata JSON."""
    practice_metadata_path.mkdir(parents=True, exist_ok=True)
    return practice_metadata_path / f'sub-{subject:02d}_ses-{session:03d}_practice_metadata.json'


def save_practice_metadata(subject, session, practice_runs, run_onsets, n_samples, 
                           n_trs_per_run, level_onsets, n_levels, run_phases, 
                           practice_metadata_path=PATHS['practice_phase_metadata']):
    """
    Save practice phase metadata as JSON.
    
    Parameters:
    -----------
    subject : int
        Subject number
    session : int
        Session number
    practice_runs : list of int
        Run numbers in practice phase
    run_onsets : array of int
        Cumulative TR counts
    n_samples : int
        Total TRs
    n_trs_per_run : list of int
        TRs per run
    level_onsets : array of int
        Cumulative TR counts where each level starts
    n_levels : int
        Total number of level repetitions
    run_phases : dict
        Phase information for all runs
    practice_metadata_path : Path
        Output directory
        
    Returns:
    --------
    filepath : Path
        Path to saved JSON file
    """
    metadata = {
        'subject': subject,
        'session': session,
        'practice_runs': practice_runs,
        'run_onsets': run_onsets.tolist(),
        'n_samples': n_samples,
        'n_runs': len(practice_runs),
        'n_trs_per_run': n_trs_per_run,
        'level_onsets': level_onsets.tolist(),
        'n_levels': n_levels,
        'run_phases': run_phases
    }
    
    filepath = get_practice_metadata_path(subject, session, practice_metadata_path)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filepath


def extract_subject_session_from_events_path(filepath):
    """Extract subject and session from events file path."""
    parts = filepath.parts
    subject_dir = [p for p in parts if p.startswith('sub-')][0]
    session_dir = [p for p in parts if p.startswith('ses-')][0]
    subject = int(subject_dir.split('-')[1])
    session = int(session_dir.split('-')[1])
    return subject, session


def find_all_sessions_with_events(events_path=PATHS['bids_annotated_tsvs']):
    """Find all unique subject/session combinations with events files."""
    pattern = 'sub-*/ses-*/func/*_desc-annotated_events.tsv'
    events_files = list(events_path.glob(pattern))
    
    sessions = set()
    for filepath in events_files:
        subject, session = extract_subject_session_from_events_path(filepath)
        sessions.add((subject, session))
    
    return sorted(sessions)


def process_session_practice_metadata(subject, session, 
                                      events_path=PATHS['bids_annotated_tsvs'],
                                      per_run_downsampled_path=PATHS['per_run_downsampled_to_TR'],
                                      practice_metadata_path=PATHS['practice_phase_metadata']):
    """
    Process one session to generate practice phase metadata.
    
    Parameters:
    -----------
    subject : int
        Subject number
    session : int
        Session number
    events_path : Path
        Path to BIDS events directory
    per_run_downsampled_path : Path
        Path to downsampled data
    practice_metadata_path : Path
        Path to save metadata
        
    Returns:
    --------
    dict with processing results
    """
    print(f"  Processing session {session:03d}...")
    
    try:
        # Identify practice runs
        practice_runs, run_phases = identify_practice_runs_from_bids(subject, session, events_path)
        
        if not practice_runs:
            return {
                'subject': subject,
                'session': session,
                'n_practice_runs': 0,
                'status': 'no practice runs found'
            }
        print(f"    Found {len(practice_runs)} practice runs: {practice_runs}")
        # Compute run onsets
        run_onsets, n_samples, n_trs_per_run = compute_run_onsets(
            subject, session, practice_runs, per_run_downsampled_path
        )
        print(f"    Total TRs: {n_samples}")
        # Compute level onsets
        level_onsets, n_levels = compute_level_onsets(
            subject, session, practice_runs, events_path, per_run_downsampled_path
        )
        print(f"    Total levels: {n_levels}")
        # Save metadata
        filepath = save_practice_metadata(
            subject, session, practice_runs, run_onsets, n_samples,
            n_trs_per_run, level_onsets, n_levels, run_phases, practice_metadata_path
        )

        return {
            'subject': subject,
            'session': session,
            'n_practice_runs': len(practice_runs),
            'n_samples': n_samples,
            'n_levels': n_levels,
            'filepath': filepath,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'subject': subject,
            'session': session,
            'n_practice_runs': 0,
            'n_samples': 0,
            'filepath': None,
            'status': f'failed: {str(e)}'
        }


def process_all_practice_metadata(events_path=PATHS['bids_annotated_tsvs'],
                                  per_run_downsampled_path=PATHS['per_run_downsampled_to_TR'],
                                  practice_metadata_path=PATHS['practice_phase_metadata']):
    """
    Process all sessions to generate practice phase metadata.
    
    Returns:
    --------
    pd.DataFrame with processing results
    """
    sessions = find_all_sessions_with_events(events_path)
    if not sessions:
        raise ValueError(f"No sessions with events files found in {events_path}")
    print(f"Found {len(sessions)} sessions to process")
    print()
    results = []
    current_subject = None
    for subject, session in sessions:
        if subject != current_subject:
            current_subject = subject
            print(f"Processing sub-{subject:02d}...")
        result = process_session_practice_metadata(
            subject, session, events_path, per_run_downsampled_path, practice_metadata_path
        )
        results.append(result)
        if result['status'] == 'success':
            print(f"    Saved metadata: {result['n_practice_runs']} runs, {result['n_samples']} TRs")
        else:
            print(f"    {result['status']}")
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Generating practice phase metadata for cross-validation...")
    print(f"Events path: {PATHS['bids_annotated_tsvs']}")
    print(f"Downsampled data path: {PATHS['per_run_downsampled_to_TR']}")
    print(f"Output path: {PATHS['practice_phase_metadata']}")
    print()
    
    results = process_all_practice_metadata()
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Total sessions processed: {len(results)}")
    print(f"Successful: {(results['status'] == 'success').sum()}")
    print(f"No practice runs: {(results['status'] == 'no practice runs found').sum()}")
    print(f"Failed: {(results['status'].str.startswith('failed')).sum()}")
    
    if (results['status'] == 'success').any():
        successful = results[results['status'] == 'success']
        print(f"\nPractice phase statistics:")
        print(f"  Average runs per session: {successful['n_practice_runs'].mean():.1f}")
        print(f"  Average TRs per session: {successful['n_samples'].mean():.0f}")
        print(f"  Average levels per session: {successful['n_levels'].mean():.1f}")
        print(f"  Total practice runs across all sessions: {successful['n_practice_runs'].sum()}")
    
    if (results['status'] != 'success').any():
        print("\nSessions without practice runs or failed:")
        failed = results[results['status'] != 'success']
        print(failed[['subject', 'session', 'status']])
