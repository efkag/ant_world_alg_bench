import pandas as pd

def filter_results(df: pd.DataFrame, **filters):
    for fil in filters:
        if fil not in df:
            print(fil, ' is not a valid column name')
            return
        df = df.loc[df[fil] == filters[fil]]
    return df