import json
import pandas as pd
from mario_encoding.config import PATHS
from mario_encoding.utils import process_all_replay_sessions, process_all_run_merges


if __name__ == '__main__':
    replay_results = process_all_replay_sessions(
            replay_variables_path=PATHS['replay_variables_jsons'],
            framewise_path=PATHS['per_session_framewise_tsvs']
            )
    merge_results = process_all_run_merges(
            framewise_path=PATHS['per_session_framewise_tsvs'],
            events_path=PATHS['bids_annotated_tsvs']
            )
