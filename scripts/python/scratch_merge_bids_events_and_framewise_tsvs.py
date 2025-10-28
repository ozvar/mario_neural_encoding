import json
import pandas as pd
from mario_encoding.config import PATHS
from mario_encoding.utils import process_all_run_merges


if __name__ == '__main__':
    results = process_all_run_merges(
            framewise_path=PATHS['per_session_framewise_tsvs'],
            events_path=PATHS['bids_annotated_tsvs']
            )
