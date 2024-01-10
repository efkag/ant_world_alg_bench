import pandas as pd
from ast import literal_eval

def read_results(path: str):
    data = pd.read_csv(path, index_col=False)
    # if columns have been left with Nan Values replace then with False
    data.fillna('False', inplace=True)
    # Coloumns that we want to have
    keys = ['errors', 'dist_diff', 'abs_index_diff', 'tx', 'ty', 'th', 
            'matched_index', 'min_dist_index', 'window_log', 'best_sims', 
            'tfc_idxs']
    # Convert list of strings to actual list of lists
    for k in keys:
        if k in data.columns:
            data[k] = data[k].apply(literal_eval)
    return data


def filter_results(df: pd.DataFrame, **filters):
    for fil in filters:
        if fil not in df:
            print(fil, ' is not a valid column name')
            return
        df = df.loc[df[fil] == filters[fil]]
    return df

