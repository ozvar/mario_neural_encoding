"""Shared helpers for the PMe audit scripts."""
import logging
import re
from pathlib import Path

import numpy as np


def setup_logging(output_dir, log_name='audit.log'):
    """Create the output directory and configure file and console logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(output_dir / log_name),
                  logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def safe_median(series):
    """Median that returns NaN for an all-missing series without warnings."""
    values = series.dropna()
    return float(values.median()) if len(values) else np.nan


def frame_to_markdown(df):
    """Render a DataFrame as a simple Markdown table without tabulate."""
    columns = [str(col) for col in df.columns]
    lines = ['| ' + ' | '.join(columns) + ' |',
             '|' + '|'.join(['---'] * len(columns)) + '|']
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float) and np.isfinite(value):
                values.append(f'{value:.3f}')
            elif isinstance(value, float):
                values.append('')
            else:
                values.append(str(value).replace('|', '/'))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def find_run_path(directory, subject, session, run, suffix):
    """Return the path of a per-run TSV from a project inputs directory."""
    return (Path(directory) / f'sub-{subject:02d}' / f'ses-{session:03d}' /
            f'sub-{subject:02d}_ses-{session:03d}_run-{run:02d}_{suffix}.tsv')


def level_sort_key(level):
    """Sort levels such as w2l1 by world and level number."""
    match = re.match(r'w(\d+)l(\d+)', str(level))
    if match:
        return int(match.group(1)), int(match.group(2)), str(level)
    return 999, 999, str(level)
