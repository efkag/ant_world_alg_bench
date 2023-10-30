import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())


import pandas as pd
from source.utils import check_for_dir_and_create

directory = '2023-06-23'
results_path = os.path.join('Results', 'ftl', directory)
df1 = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)

# w=15
# df1.loc[df1['window'] == w, 'nav-name'] = f'SMW({w})'
# w=20
# df1.loc[df1['window'] == w, 'nav-name'] = f'SMW({w})'
# w=25
# df1.loc[df1['window'] == w, 'nav-name'] = f'SMW({w})'
# w=-15
# df1.loc[df1['window'] == w, 'nav-name'] = f'A-SMW({abs(w)})'

# df1.to_csv(os.path.join(results_path, 'results.csv'))


directory = '2023-09-08'
results_path = os.path.join('Results', 'ftl', directory)
df2 = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)


combined_df = pd.concat([df1, df2])
save_path = os.path.join(fwd, 'combined')
check_for_dir_and_create(save_path)
save_path = os.path.join(save_path, 'results.csv')

combined_df.to_csv(save_path, index=False)