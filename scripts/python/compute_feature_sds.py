"""
Compute per-feature training standard deviations for the tuning-profile analysis.

The tuning analysis (docs/tuning_profiles_design.md) rescales each encoding
weight by the standard deviation of its feature, turning raw weights into
standardized coefficients that are comparable across features on different
natural scales. That SD is the std of each feature over the active training TRs.
It is invariant to the fit-time mean-centering and to FIR delaying, so it is
recomputed here directly from the feature time series without needing the model
or the fMRI responses.

Zero-variance filtering during the fit drops a small, subject-specific set of
rare scene features, so the surviving feature set is not the full config list
and differs across subjects. To guarantee row-alignment with the saved weights,
this script loads the fit's own valid_features_mask.npy rather than recomputing
the mask, and emits per-space SD arrays in config order with dropped features
omitted (matching *_weights_avg). Each space's surviving feature names are saved
alongside so downstream code can align by name.
"""
import argparse
import logging

import numpy as np

from mario_encoding.config import PATHS, PARAMETERS, FEATURE_SPACES
from mario_encoding.utils.data_loading import (
    load_session_data_with_names,
    select_and_validate_features,
    compute_baseline_mask,
)


def setup_logging(subject: int, session_range: tuple, log_dir) -> logging.Logger:
    log_file = log_dir / f'sub-{subject:02d}_ses-{session_range[0]:03d}-{session_range[1]:03d}_feature_sds.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def get_output_directory(subject: int, train_sessions: list, test_sessions: list,
                         experiment_id: str, base_path):
    """Mirror fit_variance_partitioning.get_output_directory so SDs sit beside weights."""
    train_str = f"{min(train_sessions):03d}-{max(train_sessions):03d}"
    test_str = f"{min(test_sessions):03d}-{max(test_sessions):03d}"
    output_dir = base_path / f'sub-{subject:02d}_train-ses-{train_str}_test-ses-{test_str}' / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_training_features(subject: int, sessions: list, fmri_path, pipeline: str,
                           logger: logging.Logger):
    """Concatenate training feature time series across sessions, discarding fMRI.

    Replicates the run-onset accumulation of concatenate_sessions_with_names but
    drops each session's Y immediately to keep memory low.
    """
    X_list = []
    run_onsets_list = [0]
    cumulative_samples = 0
    feature_names = None

    for session in sessions:
        X_sess, Y_sess, metadata, feat_names = load_session_data_with_names(
            subject, session, PATHS['practice_phase_metadata'],
            PATHS['per_run_downsampled_to_TR'], fmri_path, pipeline, logger
        )
        del Y_sess  # fMRI responses are not needed for feature SDs

        if feature_names is None:
            feature_names = feat_names
        elif feat_names != feature_names:
            raise ValueError(
                f"Feature mismatch in session {session}: feature order differs from first session"
            )

        X_list.append(X_sess)
        run_onsets_sess = np.array(metadata['run_onsets']) + cumulative_samples
        if cumulative_samples == 0:
            run_onsets_list.extend(run_onsets_sess[1:].tolist())
        else:
            run_onsets_list.extend(run_onsets_sess.tolist())
        cumulative_samples += metadata['n_samples']

    X = np.vstack(X_list)
    run_onsets = np.array(run_onsets_list)
    logger.info(f"Concatenated {len(sessions)} sessions: {X.shape[0]} TRs, {len(run_onsets)} runs")
    return X, run_onsets, feature_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute per-feature training SDs for the tuning-profile analysis'
    )
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--sessions', type=int, nargs='+', required=True,
                        help='Training session numbers (must match the fit)')
    parser.add_argument('--test-sessions', type=int, nargs='+', required=True,
                        help='Held-out test session(s); used only to locate the fit output directory')
    parser.add_argument('--experiment-id', type=str, required=True,
                        help='Experiment ID of the corresponding fit, to colocate feature_sds.npz')
    args = parser.parse_args()

    train_sessions = args.sessions
    session_range = (min(train_sessions), max(train_sessions))
    logger = setup_logging(args.subject, session_range, PATHS['logs'])

    pipeline = PARAMETERS['preprocessing_pipeline']
    fmri_path = PATHS['hcp_data'] if pipeline == 'hcp' else PATHS['fmriprep_data']

    logger.info("=" * 80)
    logger.info("COMPUTE PER-FEATURE TRAINING SDs")
    logger.info("=" * 80)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Training sessions: {train_sessions}")
    logger.info(f"Pipeline: {pipeline}")

    output_dir = get_output_directory(
        args.subject, train_sessions, args.test_sessions, args.experiment_id,
        PATHS['variance_partitioning']
    )
    logger.info(f"Output directory: {output_dir}")

    mask_file = output_dir / 'valid_features_mask.npy'
    if not mask_file.exists():
        raise FileNotFoundError(
            f"valid_features_mask.npy not found in {output_dir}. Check that --experiment-id "
            f"matches a completed fit for this subject (group_configs.py)."
        )

    # Load features (fMRI discarded), select configured features, mark active TRs.
    X, run_onsets, feature_names = load_training_features(
        args.subject, train_sessions, fmri_path, pipeline, logger
    )
    X, feature_names, _ = select_and_validate_features(X, feature_names, FEATURE_SPACES, logger)
    active_mask = compute_baseline_mask(X, run_onsets, logger=logger)

    # Use the fit's own zero-variance mask so surviving features (and their order)
    # match *_weights_avg exactly. The mask is over the selected (config-subset) features.
    valid_features_mask = np.load(mask_file)
    if valid_features_mask.shape[0] != len(feature_names):
        raise ValueError(
            f"valid_features_mask length {valid_features_mask.shape[0]} != number of selected "
            f"features {len(feature_names)}; selection is inconsistent with the fit."
        )
    X = X[:, valid_features_mask]
    feature_names = [f for f, keep in zip(feature_names, valid_features_mask) if keep]
    np.nan_to_num(X, copy=False)

    # SD over active training TRs (ddof=0 matches sklearn StandardScaler).
    sds = X[active_mask].std(axis=0, ddof=0)
    sd_by_name = dict(zip(feature_names, sds))

    # Per-space arrays in config order with dropped features omitted, row-aligned to
    # *_weights_avg. Surviving names saved per space so downstream can align by name.
    save_dict = {'feature_names': np.array(feature_names)}
    for space, names in FEATURE_SPACES.items():
        surviving = [f for f in names if f in sd_by_name]
        dropped = [f for f in names if f not in sd_by_name]
        save_dict[space] = np.array([sd_by_name[f] for f in surviving])
        save_dict[f'{space}_features'] = np.array(surviving)
        logger.info(f"  {space}: {len(surviving)}/{len(names)} features, "
                    f"SD range [{save_dict[space].min():.4f}, {save_dict[space].max():.4f}]"
                    + (f"; dropped (zero-variance in fit): {dropped}" if dropped else ""))

    out_file = output_dir / 'feature_sds.npz'
    np.savez(out_file, **save_dict)
    logger.info(f"Saved feature SDs to: {out_file}")


if __name__ == '__main__':
    main()
