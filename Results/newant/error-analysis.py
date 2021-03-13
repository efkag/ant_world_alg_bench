import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)


def to_array(x):
    return np.fromstring(x[1:-1], dtype=np.int, sep=' ').tolist()

fig_save_path = 'violins.png'
data = pd.read_csv('exp3.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)


route_id = 5
matcher = 'mae'
preproc = '(220, 240)'
figsize = (4, 3)
'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=figsize)
plt.title(matcher + ', route:' + str(route_id))
route1 = data.loc[(data['matcher'] == matcher) & (data['route_id'] == route_id) & (data['route_id'] == route_id)]
# Group then back to dataframe
route1 = route1.groupby(['window'])['errors'].apply(sum).to_frame('errors').reset_index()
v_data = route1['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=v_data, cut=0, ax=ax)
labels = route1['window'].tolist()
ax.set_xticklabels(labels)
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
plt.tight_layout(pad=0)
# fig_save_path = 'Figures/correlation and high-res edges.png'
# # fig_save_path = 'Figures/rmse and low-res blur.png'
# fig.savefig(fig_save_path)
plt.show()


'''
Plot for one specific matcher with one specific pre-proc
'''
missmatch_metric = 'dist_diff'
# missmatch_metric = 'abs_index_diff'

fig, ax = plt.subplots(figsize=figsize)
plt.title(matcher + ', route:' + str(route_id))
route1 = data.loc[(data['matcher'] == matcher) & (data['route_id'] == route_id) & (data['edge'] == preproc)]
# Group then back to dataframe
route1 = route1.groupby(['window'])[missmatch_metric].apply(sum).to_frame(missmatch_metric).reset_index()
v_data = route1[missmatch_metric].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=v_data, cut=0, ax=ax)
labels = route1['window'].tolist()
ax.set_xticklabels(labels)
ax.set_ylabel(missmatch_metric)
ax.set_xlabel('Window size')
plt.tight_layout(pad=0)
# fig_save_path = 'Figures/correlation and high-res edges.png'
# # fig_save_path = 'Figures/rmse and low-res blur.png'
# fig.savefig(fig_save_path)
plt.show()

