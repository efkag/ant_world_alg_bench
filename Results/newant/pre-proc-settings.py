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

directory = '2022-07-26_mid_update'
fig_save_path = os.path.join('Results', 'newant', directory)
data = pd.read_csv(os.path.join(fig_save_path, 'results.csv'), index_col=False)
# data['trial_fail_count'] = data['trial_fail_count'].apply(eval)

#select window size
window = -15
data = data.loc[data['window'] == window]
grouping_func = sum
grouping_factors = ['blur', 'edge', 'gauss_loc_norm', 'loc_norm', 'matcher']
grouped = data.groupby(grouping_factors)["trial_fail_count"].apply(grouping_func).to_frame("trial_fail_count").reset_index()
print(grouped)

#combine colusmk mfor plotting
grouped['combined'] = grouped[grouping_factors].apply(lambda row: '\n'.join(row.values.astype(str)), axis=1) 

figsize = (20, 10)
fig, ax = plt.subplots(figsize=figsize)
sns.barplot(x="combined", y="trial_fail_count", data=grouped, ax=ax, estimator=sum, capsize=.2)
#ax.bar(x=grouped['combined'], y=)
#ax.tick_params(axis='x', labelrotation=90)
path = os.path.join(fig_save_path, f'w={window}.png')
fig.savefig(path)
plt.show() 