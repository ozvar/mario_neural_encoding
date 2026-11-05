"""
Submit per-subject block-permutation significance jobs for a group config.

Mirrors submit_vp_jobs.py / submit_feature_sds.py: submits one CPU sbatch per
subject so the (independent) subjects run concurrently instead of sequentially
in a single job. Each job writes that subject's significance.npz; run
compute_group_significance.py --group <GROUP> afterwards for the group map and
four-way classification.

Usage:
    # Dry run (print commands, do not submit)
    python submit_permutation_jobs.py --group GROUP_N5_PRIMARY

    # Submit
    python submit_permutation_jobs.py --group GROUP_N5_PRIMARY --submit

    # Override permutation count (otherwise the configured default is used)
    python submit_permutation_jobs.py --group GROUP_N5_PRIMARY --n-perm 10000 --submit
"""
import argparse
import os
import subprocess

from mario_encoding.group_configs import GROUP_N3_PRIMARY, GROUP_N5_PRIMARY

GROUP_REGISTRY = {
    'GROUP_N3_PRIMARY': GROUP_N3_PRIMARY,
    'GROUP_N5_PRIMARY': GROUP_N5_PRIMARY,
}

SBATCH_PERM = 'scripts/bash/run_permutation_subject.sbatch'


def build_env(subject_cfg: dict, n_perm: int | None) -> dict:
    env = os.environ.copy()
    env['SUBJECT']       = str(subject_cfg['subject'])
    env['SESSIONS']      = ' '.join(str(s) for s in subject_cfg['train_sessions'])
    env['TEST_SESSIONS'] = ' '.join(str(s) for s in subject_cfg['test_sessions'])
    env['EXPERIMENT_ID'] = str(subject_cfg['experiment_id'])
    if n_perm is not None:
        env['N_PERM'] = str(n_perm)
    return env


def submit_perm(subject_cfg: dict, n_perm: int | None, dry_run: bool) -> None:
    sub = subject_cfg['subject']
    env = build_env(subject_cfg, n_perm)
    cmd = ['sbatch', SBATCH_PERM]

    if dry_run:
        nperm_str = f"N_PERM={env['N_PERM']} " if 'N_PERM' in env and n_perm is not None else ""
        print(
            f"SUBJECT={sub} "
            f"SESSIONS=\"{env['SESSIONS']}\" "
            f"TEST_SESSIONS=\"{env['TEST_SESSIONS']}\" "
            f"EXPERIMENT_ID={env['EXPERIMENT_ID']} "
            f"{nperm_str}"
            f"sbatch {SBATCH_PERM}"
        )
    else:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(f"sub-{sub:02d}: {result.stdout.strip() or result.stderr.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Submit per-subject block-permutation significance jobs for a group config'
    )
    parser.add_argument('--group', required=True, choices=list(GROUP_REGISTRY.keys()))
    parser.add_argument('--n-perm', type=int, default=None,
                        help='Permutation count (default: use the configured value in the script)')
    parser.add_argument('--submit', action='store_true',
                        help='Actually submit jobs (default: dry run)')
    args = parser.parse_args()

    dry_run = not args.submit
    group_cfg = GROUP_REGISTRY[args.group]
    subjects = group_cfg['subjects']

    if dry_run:
        print(f"DRY RUN -- commands that would be submitted for {args.group}:\n")

    for subject_cfg in subjects:
        submit_perm(subject_cfg, args.n_perm, dry_run)

    if dry_run:
        print(f"\nAdd --submit to execute.")


if __name__ == '__main__':
    main()
