"""Screen/Gameplay annotation handling for the refit mask.

Expand the new ``Screen/*`` event rows into framewise columns, aggregate them
per TR, build the gameplay mask used by the variance-partitioning fit, and
carry annotation provenance.

The module is pure numpy/pandas plus json/subprocess so it can be tested
without fMRI data. The new annotation files use an inclusive ``frame_stop``
(``n_frames = stop - start + 1``), unlike the legacy annotation source and the
existing merge code, which treat it as exclusive.
"""
import importlib.metadata
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GAMEPLAY_LABEL = 'Gameplay'
SCREEN_LABELS = ('Gameplay', 'LevelIntro', 'Death', 'LevelEnd', 'Transition')
SCREEN_PREFIX = 'Screen/'
REPETITION_TRIAL_TYPE = 'gym-retro_game'

SCREEN_FRAME_COLUMNS = ('screen_gameplay', 'screen_label')
SCREEN_FRACTION_COLUMNS = tuple(f'screen_frac_{label}' for label in SCREEN_LABELS)
SCREEN_METADATA_COLUMNS = (
    ('screen_gameplay',)
    + SCREEN_FRACTION_COLUMNS
    + ('screen_n_frames',)
)

BASE_METADATA_COLUMNS = ('TR_index', 'TR_time', 'level', 'score', 'time', 'coins')
ALL_METADATA_COLUMNS = BASE_METADATA_COLUMNS + SCREEN_METADATA_COLUMNS

PROVENANCE_FILENAME = 'annotation_provenance.json'

MASK_RULES = {
    'greater': np.greater,
    'greater_equal': np.greater_equal,
    'equal': np.equal,
    'less': np.less,
    'less_equal': np.less_equal,
}


class ScreenAnnotationError(RuntimeError):
    """Raised when Screen/Gameplay annotations are missing or inconsistent."""


def _as_int(value, name):
    """Return a finite integer-valued scalar, or raise."""
    if pd.isna(value):
        raise ScreenAnnotationError(f'{name} is missing')
    number = float(value)
    if not np.isfinite(number) or number != int(number):
        raise ScreenAnnotationError(f'{name} is not an integer: {value!r}')
    return int(number)


def _parse_screen_containers(annotations_df):
    """Group annotation rows by repetition container and fill Screen labels.

    Returns a dict mapping stim_file to a dict with frame_start, frame_stop,
    onset, labels and n_frames. Raises ScreenAnnotationError on missing or
    overlapping Screen rows.
    """
    required = {'trial_type', 'frame_start', 'frame_stop'}
    missing = required - set(annotations_df.columns)
    if missing:
        raise ScreenAnnotationError(
            f'annotation file missing columns: {sorted(missing)}')

    containers = {}
    current = None
    rows = annotations_df.reset_index(drop=True)

    for idx, row in rows.iterrows():
        trial_type = row['trial_type']
        if pd.isna(trial_type):
            raise ScreenAnnotationError(f'row {idx}: missing trial_type')
        trial_type = str(trial_type)

        if trial_type == REPETITION_TRIAL_TYPE:
            stim_file = row.get('stim_file')
            if not isinstance(stim_file, str) or not stim_file.strip():
                raise ScreenAnnotationError(
                    f'row {idx}: repetition container has no stim_file')
            if stim_file in containers:
                raise ScreenAnnotationError(
                    f'duplicate repetition container for {stim_file}')
            start = _as_int(row['frame_start'], f'row {idx} frame_start')
            stop = _as_int(row['frame_stop'], f'row {idx} frame_stop')
            if start != 0:
                raise ScreenAnnotationError(
                    f'{stim_file}: container starts at frame {start}, expected 0')
            if stop < start:
                raise ScreenAnnotationError(
                    f'{stim_file}: container stops before it starts')
            onset = row.get('onset')
            onset = None if pd.isna(onset) else float(onset)
            current = {
                'stim_file': stim_file,
                'frame_start': start,
                'frame_stop': stop,
                'onset': onset,
                'n_frames': stop - start + 1,
                'labels': np.full(stop - start + 1, None, dtype=object),
            }
            containers[stim_file] = current
            continue

        if current is None:
            raise ScreenAnnotationError(
                f'row {idx}: event outside a repetition container')

        if not trial_type.startswith(SCREEN_PREFIX):
            continue

        label = trial_type[len(SCREEN_PREFIX):]
        if label not in SCREEN_LABELS:
            raise ScreenAnnotationError(
                f'{current["stim_file"]}: unknown Screen label {label!r}')
        start = _as_int(row['frame_start'], f'row {idx} frame_start')
        stop = _as_int(row['frame_stop'], f'row {idx} frame_stop')
        if stop < start:
            raise ScreenAnnotationError(
                f'{current["stim_file"]}: {trial_type} stops before it starts')
        if start < current['frame_start'] or stop > current['frame_stop']:
            raise ScreenAnnotationError(
                f'{current["stim_file"]}: {trial_type} [{start}, {stop}] '
                f'outside container [{current["frame_start"]}, '
                f'{current["frame_stop"]}]')
        local = current['labels'][start:stop + 1]
        if np.any(local != None):  # noqa: E711  (object array, None sentinel)
            raise ScreenAnnotationError(
                f'{current["stim_file"]}: overlapping Screen rows at '
                f'frames {start}-{stop}')
        current['labels'][start:stop + 1] = label

    for stim_file, container in containers.items():
        labels = container['labels']
        uncovered = np.flatnonzero(labels == None)  # noqa: E711
        container['n_terminal_frames_filled'] = 0
        if not uncovered.size:
            continue

        covered = np.flatnonzero(labels != None)  # noqa: E711
        if not covered.size:
            raise ScreenAnnotationError(f'{stim_file}: no Screen rows')
        last_covered = int(covered[-1])
        interior = uncovered[uncovered <= last_covered]
        if interior.size:
            raise ScreenAnnotationError(
                f'{stim_file}: Screen rows leave {interior.size} interior '
                f'frames uncovered (first {int(interior[0])})')
        trailing = uncovered[uncovered > last_covered]
        if trailing.size != 1:
            raise ScreenAnnotationError(
                f'{stim_file}: Screen rows leave {trailing.size} terminal '
                f'frames uncovered; expected the single terminal frame')
        # The new generator includes one final done frame in the repetition
        # container but not in the Screen label array. It belongs to the last
        # screen state, so extend that label rather than zero-filling it.
        labels[trailing] = labels[last_covered]
        container['n_terminal_frames_filled'] = int(trailing.size)

    return containers


def create_screen_columns(variables_df, annotations_df, frame_rate=None):
    """Expand Screen event rows into framewise columns on the variables table.

    The input variables must contain one row per emulator frame of each
    repetition, with ``relative_path`` matching the annotation ``stim_file``
    and integer ``frame_index``. The annotation file must use the new
    inclusive ``frame_stop`` convention. The generator includes one terminal
    done frame in the container that has no Screen row; that frame inherits
    the last Screen label. Any other gap or an overlap raises.

    Returns
    -------
    out_df : pandas.DataFrame
        A copy of variables_df with ``screen_gameplay`` (0/1) and
        ``screen_label`` added in the original row order.
    coverage : dict
        Per-run coverage summary for the review manifest.
    """
    variables_required = {'relative_path', 'frame_index'}
    missing = variables_required - set(variables_df.columns)
    if missing:
        raise ScreenAnnotationError(
            f'variables table missing columns: {sorted(missing)}')

    containers = _parse_screen_containers(annotations_df)

    paths = variables_df['relative_path'].astype(str).to_numpy()
    frames = variables_df['frame_index'].to_numpy()
    frame_ints = np.asarray([_as_int(v, 'frame_index') for v in frames], dtype=int)

    variables_paths = set(paths)
    container_paths = set(containers)
    if variables_paths != container_paths:
        missing_containers = sorted(variables_paths - container_paths)
        extra_containers = sorted(container_paths - variables_paths)
        raise ScreenAnnotationError(
            'repetition mismatch between variables and annotations: '
            f'{len(missing_containers)} missing containers, '
            f'{len(extra_containers)} extra containers; '
            f'missing sample {missing_containers[:3]}, '
            f'extra sample {extra_containers[:3]}')

    label = np.empty(len(variables_df), dtype=object)
    for path in variables_paths:
        mask = paths == path
        container = containers[path]
        order = np.argsort(frame_ints[mask])
        frames_here = frame_ints[mask][order]
        if len(np.unique(frames_here)) != len(frames_here):
            raise ScreenAnnotationError(f'{path}: duplicate frame_index rows')
        full = np.arange(container['n_frames'])
        # The replay variables may omit the same single terminal done frame
        # that the Screen rows omit. Any other gap is an error.
        full_or_terminal_omitted = full[:-1] if len(full) > 1 else full[:0]
        if not (np.array_equal(frames_here, full)
                or np.array_equal(frames_here, full_or_terminal_omitted)):
            missing_frames = np.setdiff1d(full, frames_here)
            extra_frames = np.setdiff1d(frames_here, full)
            raise ScreenAnnotationError(
                f'{path}: frame_index does not cover the repetition; '
                f'{len(missing_frames)} missing, {len(extra_frames)} extra '
                f'(only the single terminal frame may be absent)')
        labels_here = container['labels'][frames_here]
        label[np.flatnonzero(mask)[order]] = labels_here

    if np.any(label == None):  # noqa: E711
        raise ScreenAnnotationError('some variables rows have no Screen label')

    out_df = variables_df.copy()
    out_df['screen_label'] = label
    out_df['screen_gameplay'] = (label == GAMEPLAY_LABEL).astype(int)

    frame_counts = pd.Series(label).value_counts().to_dict()
    max_onset_delta = None
    n_onset_over_one_frame = None
    if 'rep_onset' in variables_df.columns:
        deltas = []
        for path in variables_paths:
            container = containers[path]
            if container['onset'] is None:
                continue
            rep_onset = variables_df.loc[
                variables_df['relative_path'].astype(str) == path, 'rep_onset'
            ].iloc[0]
            if pd.isna(rep_onset):
                continue
            deltas.append(abs(float(rep_onset) - container['onset']))
        if deltas:
            max_onset_delta = float(np.max(deltas))
            if frame_rate is not None:
                tolerance = 1.0 / float(frame_rate)
                n_onset_over_one_frame = int(sum(d > tolerance for d in deltas))

    coverage = {
        'n_repetitions': len(containers),
        'n_frames_total': int(sum(c['n_frames'] for c in containers.values())),
        'n_variables_frames': int(len(variables_df)),
        'n_uncovered': 0,
        'n_double_covered': 0,
        'n_terminal_frames_filled': int(sum(
            c['n_terminal_frames_filled'] for c in containers.values())),
        'labels_seen': {str(k): int(v) for k, v in frame_counts.items()},
        'max_onset_delta_s': max_onset_delta,
        'n_onset_over_one_frame': n_onset_over_one_frame,
    }
    return out_df, coverage


def aggregate_screen_fractions(group):
    """Aggregate framewise Screen labels within one TR into metadata columns."""
    if 'screen_label' not in group.columns:
        raise ScreenAnnotationError('TR group has no screen_label column')
    n_frames = len(group)
    if n_frames == 0:
        values = {col: 0.0 for col in SCREEN_METADATA_COLUMNS}
        return pd.Series(values, index=SCREEN_METADATA_COLUMNS)

    labels = group['screen_label'].astype(str)
    unknown = set(labels) - set(SCREEN_LABELS)
    if unknown:
        raise ScreenAnnotationError(f'unknown Screen labels in TR: {sorted(unknown)}')
    counts = labels.value_counts()

    values = {
        f'screen_frac_{label}': counts.get(label, 0) / n_frames
        for label in SCREEN_LABELS
    }
    values['screen_gameplay'] = values[f'screen_frac_{GAMEPLAY_LABEL}']
    values['screen_n_frames'] = float(n_frames)

    if 'screen_gameplay' in group.columns:
        frame_p = float(np.mean(group['screen_gameplay'].to_numpy(dtype=float)))
        if not np.isclose(frame_p, values['screen_gameplay'], atol=1e-12):
            raise ScreenAnnotationError(
                'framewise screen_gameplay disagrees with screen_label')

    return pd.Series(values, index=SCREEN_METADATA_COLUMNS)


def validate_screen_metadata(df, tolerance=1e-6):
    """Validate the TR-level screen metadata columns on a downsampled table."""
    missing = set(SCREEN_METADATA_COLUMNS) - set(df.columns)
    if missing:
        raise ScreenAnnotationError(
            f'screen metadata columns missing: {sorted(missing)}')

    p = df['screen_gameplay'].to_numpy(dtype=float)
    if not np.all(np.isfinite(p)):
        raise ScreenAnnotationError('screen_gameplay contains non-finite values')
    if np.any((p < 0) | (p > 1)):
        raise ScreenAnnotationError('screen_gameplay outside [0, 1]')

    n_frames = df['screen_n_frames'].to_numpy(dtype=float)
    if not np.all(np.isfinite(n_frames)):
        raise ScreenAnnotationError('screen_n_frames contains non-finite values')
    if np.any(n_frames < 0) or np.any(n_frames != np.round(n_frames)):
        raise ScreenAnnotationError('screen_n_frames must be non-negative integers')

    fractions = df[list(SCREEN_FRACTION_COLUMNS)].to_numpy(dtype=float)
    if not np.all(np.isfinite(fractions)):
        raise ScreenAnnotationError('screen fractions contain non-finite values')
    if np.any((fractions < 0) | (fractions > 1)):
        raise ScreenAnnotationError('screen fractions outside [0, 1]')

    fraction_sum = fractions.sum(axis=1)
    has_frames = n_frames > 0
    if np.any(np.abs(fraction_sum[has_frames] - 1.0) > tolerance):
        raise ScreenAnnotationError('screen fractions do not sum to 1 on TRs with frames')
    if np.any(np.abs(fraction_sum[~has_frames]) > tolerance):
        raise ScreenAnnotationError('screen fractions are non-zero on TRs without frames')

    if np.any(np.abs(df['screen_gameplay'].to_numpy(dtype=float)
                     - df[f'screen_frac_{GAMEPLAY_LABEL}'].to_numpy(dtype=float)) > tolerance):
        raise ScreenAnnotationError('screen_gameplay differs from screen_frac_Gameplay')


def compute_screen_gameplay_mask(p, rule='greater', threshold=0.0, logger=None):
    """Return the boolean TR mask selected by a Screen/Gameplay rule.

    The primary rule is ``greater`` at 0.0 (``p > 0``). A later strict
    sensitivity rule is ``equal`` at 1.0 (``p == 1``).
    """
    if rule not in MASK_RULES:
        raise ValueError(f'unknown mask rule {rule!r}; expected {sorted(MASK_RULES)}')
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError('screen_gameplay must be one-dimensional')
    if not np.all(np.isfinite(p)):
        raise ValueError('screen_gameplay contains non-finite values')
    if np.any((p < 0) | (p > 1)):
        raise ValueError('screen_gameplay outside [0, 1]')
    if not np.isfinite(float(threshold)):
        raise ValueError('mask threshold must be finite')

    mask = MASK_RULES[rule](p, float(threshold))
    if logger is not None:
        n_kept = int(np.sum(mask))
        logger.info(
            f'Screen/Gameplay mask ({rule} {threshold}): '
            f'{n_kept}/{len(mask)} TRs kept ({n_kept / len(mask) * 100:.1f}%)')
    return mask


def gameplay_segments(mask, run_onsets):
    """Return contiguous gameplay segment ids and bounds, never crossing runs.

    Returns
    -------
    segment_ids : (n_samples,) int
        Segment id per pre-drop sample, -1 where the mask is False.
    bounds : (n_segments, 4) int
        Rows of (segment_id, run_index, start, end_exclusive) in pre-drop
        sample indices.
    """
    mask = np.asarray(mask, dtype=bool)
    run_onsets = np.asarray(run_onsets, dtype=int)
    if mask.ndim != 1:
        raise ValueError('mask must be one-dimensional')
    if run_onsets.ndim != 1 or run_onsets.size == 0:
        raise ValueError('run_onsets must be a non-empty 1D array')
    if run_onsets[0] != 0 or np.any(np.diff(run_onsets) <= 0):
        raise ValueError('run_onsets must start at 0 and increase')
    if run_onsets[-1] >= len(mask):
        raise ValueError('run_onsets exceed the mask length')

    run_ends = list(run_onsets[1:]) + [len(mask)]
    segment_ids = np.full(len(mask), -1, dtype=int)
    bounds = []
    next_id = 0

    for run_index, (start, end) in enumerate(zip(run_onsets, run_ends)):
        i = int(start)
        while i < end:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < end and mask[j]:
                j += 1
            segment_ids[i:j] = next_id
            bounds.append((next_id, run_index, i, j))
            next_id += 1
            i = j

    bounds = np.asarray(bounds, dtype=int) if bounds else np.empty((0, 4), dtype=int)
    return segment_ids, bounds


def _run_git(root, *args):
    """Run git in root; return stdout or None if unavailable."""
    try:
        result = subprocess.run(
            ['git', '-C', str(root), *args],
            capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _videogames_utils_commit():
    """Return the installed videogames_utils VCS commit when available."""
    try:
        dist = importlib.metadata.distribution('videogames_utils')
        direct_url = dist.read_text('direct_url.json')
        if direct_url:
            info = json.loads(direct_url).get('vcs_info', {})
            commit = info.get('commit_id')
            if commit:
                return commit
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass

    try:
        spec = importlib.util.find_spec('videogames_utils')
        if spec is not None and spec.origin:
            package_dir = Path(spec.origin).resolve().parent
            commit = _run_git(package_dir, 'rev-parse', 'HEAD')
            if commit:
                return commit
    except (ImportError, OSError):
        pass
    return None


def collect_annotation_provenance(annotation_root, generation_command=None):
    """Collect provenance from the annotation checkout and installed package."""
    root = Path(annotation_root).resolve()
    commit = _run_git(root, 'rev-parse', 'HEAD')
    status = _run_git(root, 'status', '--porcelain')
    try:
        version = importlib.metadata.version('videogames_utils')
    except importlib.metadata.PackageNotFoundError:
        version = None

    return {
        'annotation_root': str(root),
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': {
            'repository': 'https://github.com/courtois-neuromod/mario',
            'commit': commit,
            'dirty': bool(status) if status is not None else None,
            'videogames_utils_version': version,
            'videogames_utils_commit': _videogames_utils_commit(),
        },
        'generation_command': generation_command,
    }


def write_annotation_provenance(annotation_root, provenance):
    """Write annotation_provenance.json at the annotation root."""
    path = Path(annotation_root) / PROVENANCE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(provenance, handle, indent=2)
        handle.write('\n')
    return path


def load_annotation_provenance(annotation_root):
    """Load annotation_provenance.json, raising if it is missing."""
    path = Path(annotation_root) / PROVENANCE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f'Annotation provenance not found: {path}. '
            'Regenerate the annotation inputs first.')
    with open(path) as handle:
        return json.load(handle)


def load_or_collect_annotation_provenance(annotation_root, generation_command=None,
                                           logger=None):
    """Load provenance, or collect and write it when the file is absent."""
    path = Path(annotation_root) / PROVENANCE_FILENAME
    if path.exists():
        return load_annotation_provenance(annotation_root)
    provenance = collect_annotation_provenance(
        annotation_root, generation_command=generation_command)
    write_annotation_provenance(annotation_root, provenance)
    if logger is not None:
        logger.warning(
            f'Annotation provenance missing at {path}; collected what could '
            'be determined and wrote it. Check the commit and version fields.')
    return provenance


def carry_provenance(source_root, target_dir, step, command, extra=None,
                     logger=None):
    """Copy provenance into a downstream output directory and record the step."""
    provenance = load_annotation_provenance(source_root)
    carried = provenance.setdefault('carried_forward', [])
    carried.append({
        'step': step,
        'command': command,
        'at_utc': datetime.now(timezone.utc).isoformat(),
        'extra': extra or {},
    })
    path = write_annotation_provenance(target_dir, provenance)
    if logger is not None:
        logger.info(f'Carried annotation provenance to {path}')
    return path
