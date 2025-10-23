import json
import pandas as pd
from mario_encoding.config import PATHS
from mario_replays.load_data import load_replay_data

if __name__ == '__main__':
    filepath = '/home/ozvar/Git/cneuromod/mario.replays/outputdata_test/replays/sub-01/ses-001/beh/variables/sub-01_ses-001_task-mario_level-w1l1_rep-000.json'
    with open(filepath, 'r') as file:
        data = json.load(file)
    # List type and length (if applicable)
    for key, value in data.items():
        if isinstance(value, list):
            print(f"{key}: {type(value).__name__} (length: {len(value)})")
        else:
            print(f"{key}: {type(value).__name__}")

    exclude_keys = [
            'filename',
            'level',
            'subject',
            'session',
            'actions'
            ]

    def json_to_dataframe(data, exclude_keys=None):
        if exclude_keys is None:
            exclude_keys = []
        # Filter out unwanted keys
        filtered_data = {k: v for k, v in data.items() if k not in exclude_keys}
        
        # Convert to DataFrame (pandas handles mixed string/list values)
        df = pd.DataFrame(filtered_data)
        
        return df

    df = json_to_dataframe(data, exclude_keys) 

    repetitions_df = load_replay_data(PATHS['preprocessed_sidecars'] / 'replays', type='metadata')
    repetitions_variables = load_replay_data(PATHS['preprocessed_sidecars'] / 'replays', type='variables')

    first = repetitions_variables.iloc[0]
    df = first.to_frame().T

    list_columns = [col for col in df.columns if isinstance(df[col].iloc[0], list)]
    explode_cols = [col for col in df.columns if col not in exclude_keys]

    # Identify non-list columns
    non_list_columns = [col for col in df.columns if not isinstance(df[col].iloc[0], list)]

    # Explode the list columns. As of pandas 1.3, you can explode multiple columns at once.
    # All exploded columns must have lists of the same length.
    final_df = df.explode(explode_cols, ignore_index=True)

    # Print the resulting DataFrame
    print(final_df)
