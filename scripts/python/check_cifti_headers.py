"""
Check CIFTI header TR and start time across runs for a given subject/session.
"""
import argparse
import nibabel as nib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from mario_encoding.config import PATHS, PARAMETERS
from mario_encoding.utils.data_loading import get_fmri_filepath, load_practice_metadata


def main():
    parser = argparse.ArgumentParser(description='Check CIFTI headers for a subject/session')
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--session', type=int, required=True)
    args = parser.parse_args()

    pipeline  = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']
    meta      = load_practice_metadata(args.subject, args.session, PATHS['practice_phase_metadata'])

    print(f"sub-{args.subject:02d} ses-{args.session:03d} -- CIFTI headers ({pipeline})")
    print(f"{'Run':<8} {'Shape':<20} {'TR (s)':<10} {'Start (s)':<12}")
    print('-' * 52)

    for run in meta['practice_runs']:
        path = get_fmri_filepath(args.subject, args.session, run, fmri_path, pipeline)
        if not path.exists():
            print(f"run-{run:02d}   MISSING: {path}")
            continue
        img  = nib.load(str(path))
        ax   = img.header.matrix.get_axis(0)
        print(f"run-{run:02d}   {str(img.shape):<20} {ax.step:<10.4f} {ax.start:<12.4f}")


if __name__ == '__main__':
    main()
