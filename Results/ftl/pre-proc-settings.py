import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale=1)

directory = 'ftl/2023-06-23'
results_path = os.path.join('Results', directory)
fig_save_path = os.path.join('Results', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# data['trial_fail_count'] = data['trial_fail_count'].apply(eval)
data['errors'] = data['errors'].apply(eval)

#select window size
window = 0
data = data.loc[data['window'] == window]
grouping_func = sum
grouping_factors = ['blur', 'gauss_loc_norm', 'matcher']
metric = 'errors'
grouped = data.groupby(grouping_factors)[metric].apply(grouping_func).to_frame(metric).reset_index()


#combine colusmk mfor plotting
grouped['combined'] = grouped[grouping_factors].apply(lambda row: '\n'.join(row.values.astype(str)), axis=1) 
grouped = grouped.explode('errors')

figsize = (20, 10)
fig, ax = plt.subplots(figsize=figsize)
sns.barplot(x="combined", y=metric, data=grouped, ax=ax, estimator=np.mean, capsize=.2)
#ax.bar(x=grouped['combined'], y=)
#ax.tick_params(axis='x', labelrotation=90)
path = os.path.join(fig_save_path, f'w={window}.png')
fig.savefig(path)
plt.show() 