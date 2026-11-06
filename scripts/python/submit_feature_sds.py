"""
Submit per-feature SD jobs for all subjects in a group config.

Mirrors submit_vp_jobs.py: reads session configs and experiment IDs from
group_configs.py and submits one sbatch job per subject. Environment variables
are set on the subprocess to avoid --export quoting issues with space-separated
session lists.

Usage:
    # Dry run (print commands, do not submit)
    python submit_feature_sds.py --group GROUP_N5_PRIMARY

    # Submit
    python submit_feature_sds.py --group GROUP_N5_PRIMARY --submit
"""
import argparse
import os
import subprocess

from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}

SBATCH_SDS = 'scripts/bash/run_feature_sds.sbatch'


def build_env(subject_cfg: dict) -> dict:
    env = os.environ.copy()
    env['SUBJECT']       = str(subject_cfg['subject'])
    env['SESSIONS']      = ' '.join(str(s) for s in subject_cfg['train_sessions'])
    env['TEST_SESSIONS'] = ' '.join(str(s) for s in subject_cfg['test_sessions'])
    env['EXPERIMENT_ID'] = str(subject_cfg['experiment_id'])
    return env


def submit_sds(subject_cfg: dict, dry_run: bool) -> None:
    sub = subject_cfg['subject']
    env = build_env(subject_cfg)
    cmd = ['sbatch', SBATCH_SDS]

    if dry_run:
        print(
            f"SUBJECT={sub} "
            f"SESSIONS=\"{env['SESSIONS']}\" "
            f"TEST_SESSIONS=\"{env['TEST_SESSIONS']}\" "
            f"EXPERIMENT_ID={env['EXPERIMENT_ID']} "
            f"sbatch {SBATCH_SDS}"
        )
    else:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(f"sub-{sub:02d}: {result.stdout.strip() or result.stderr.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Submit per-feature SD jobs for all subjects in a group config'
    )
    parser.add_argument('--group', required=True, choices=list(GROUP_REGISTRY.keys()))
    parser.add_argument('--submit', action='store_true',
                        help='Actually submit jobs (default: dry run)')
    args = parser.parse_args()

    dry_run = not args.submit
    group_cfg = GROUP_REGISTRY[args.group]
    subjects = group_cfg['subjects']

    if dry_run:
        print(f"DRY RUN -- commands that would be submitted for {args.group}:\n")

    for subject_cfg in subjects:
        submit_sds(subject_cfg, dry_run)

    if dry_run:
        print(f"\nAdd --submit to execute.")


if __name__ == '__main__':
    main()
