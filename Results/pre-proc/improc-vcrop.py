import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)

directory = 'antworld2/2023-09-20'
results_path = os.path.join('Results', 'pre-proc', directory)
fig_save_path = os.path.join('Results', 'pre-proc', directory)
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# data['trial_fail_count'] = data['trial_fail_count'].apply(eval)
data['errors'] = data['errors'].apply(literal_eval)

matcher = 'mae'
data = data.loc[data['matcher'] == matcher]


metric = 'errors'
grouping_func = sum
grouping_factors = ['gauss_loc_norm', 'res',  'vcrop']
grouped = data.groupby(grouping_factors)[metric].apply(grouping_func).to_frame(metric).reset_index()
#print(grouped)
#alternate method
#grouped = data.groupby(grouping_factors, as_index=False).agg({metric:grouping_func, 'vcrop':'first'} )

grouped['errors'] = grouped['errors'].apply(np.median)

#grouped['combined'] = grouped[['gauss_loc_norm', 'res']].apply(lambda row: '\n'.join(row.values.astype(str)), axis=1) 

grouped['combined'] = grouped['res'] + '\n' + grouped['gauss_loc_norm']


heat = grouped.pivot('combined', 'vcrop', 'errors')


figsize = (6, 4)
fig, ax = plt.subplots(figsize=figsize)
sns.heatmap(heat, annot=True, fmt=".2f", cbar_kws={'label': metric}, ax=ax)

plt.tight_layout
plt.show()