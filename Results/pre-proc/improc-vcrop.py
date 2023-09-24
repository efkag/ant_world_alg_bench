import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from source.utils import check_for_dir_and_create
from ast import literal_eval
sns.set_context("paper", font_scale=0.9)

directory = 'ftl/2023-09-22'
results_path = os.path.join('Results', 'pre-proc', directory)
fig_save_path = os.path.join('Results', 'pre-proc', directory)
fig_save_path = os.path.join(results_path, 'analysis')
check_for_dir_and_create(fig_save_path)
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# data['trial_fail_count'] = data['trial_fail_count'].apply(eval)
data['errors'] = data['errors'].apply(literal_eval)

matcher = 'corr'
data = data.loc[data['matcher'] == matcher]


metric = 'errors'
grouping_func = sum
grouping_factors = ['gauss_loc_norm', 'res', 'edge', 'vcrop']
grouped = data.groupby(grouping_factors)[metric].apply(grouping_func).to_frame(metric).reset_index()
#print(grouped)
#alternate method
#grouped = data.groupby(grouping_factors, as_index=False).agg({metric:grouping_func, 'vcrop':'first'} )

grouped['errors'] = grouped['errors'].apply(np.median)

#grouped['combined'] = grouped[['gauss_loc_norm', 'res']].apply(lambda row: '\n'.join(row.values.astype(str)), axis=1) 


grouped.loc[grouped["edge"] == "(180, 200)", "edge"] = 'edges\n'
grouped.loc[grouped["edge"] == "False", "edge"] = ''

grouped.loc[grouped["gauss_loc_norm"] == "{'sig1': 2, 'sig2': 20}", "gauss_loc_norm"] = 'glocn\n'
grouped.loc[grouped["gauss_loc_norm"] == "False", "gauss_loc_norm"] = ''

grouped.loc[grouped["vcrop"] == 1.0, "vcrop"] = 0.0
# Here I can add more in the combined column.
grouped['pre processing'] = grouped['res'] + '\n' + grouped['edge'] + '\n' + grouped['gauss_loc_norm']


heat = grouped.pivot('pre processing', 'vcrop', 'errors')
print(np.min(heat))

figsize = (6, 5)
fig, ax = plt.subplots(figsize=figsize)
sns.heatmap(heat, annot=True, fmt=".2f", cbar_kws={'label': 'median AAE'}, ax=ax)
ax.set_xlabel('vertical cropping %')

plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, f'pre-proc-m({matcher}).png'))
plt.show()




############ show a violin plot

metric = 'errors'
grouping_func = sum
grouping_factors = ['gauss_loc_norm', 'res', 'edge', 'vcrop']
grouped = data.groupby(grouping_factors)[metric].apply(grouping_func).to_frame(metric).reset_index()

grouped['pre processing'] = grouped['res'] + '\n' + grouped['edge'] + '\n' + grouped['gauss_loc_norm']
grouped = grouped.explode('errors')
grouped['errors'] = pd.to_numeric(grouped['errors'])
grouped.reset_index()
figsize = (6, 5)
fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(grouped, x='pre processing', y='errors', cut=0, ax=ax)
ax.set_xlabel('vertical cropping %')

plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, f'violin-pre-proc-m({matcher}).png'))
plt.show()

