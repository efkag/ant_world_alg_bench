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
data = pd.read_csv('combined-results.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)

route_id = 5
matcher = 'corr'
# for the lack of edges I have to use .isna() function
edge = '(220, 240)'
figsize = (4, 3)
res = '(180, 50)'
route = data.loc[(data['matcher'] == matcher) & (data['route_id'] == route_id)
                 & (data['edge'] == edge) & (data['res'] == res)]

'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=figsize)
plt.title(matcher + ', route:' + str(route_id))
# Group then back to dataframe
df = route.groupby(['window'])['errors'].apply(sum).to_frame('errors').reset_index()
v_data = df['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=v_data, cut=0, ax=ax)
labels = df['window'].tolist()
ax.set_xticklabels(labels)
ax.set_ylabel('Angular error')
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
# Group then back to dataframe
df = route.groupby(['window'])[missmatch_metric].apply(sum).to_frame(missmatch_metric).reset_index()
v_data = df[missmatch_metric].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=v_data, cut=0, ax=ax)
labels = df['window'].tolist()
ax.set_xticklabels(labels)
ax.set_ylabel(missmatch_metric)
ax.set_xlabel('Window size')
plt.tight_layout(pad=0)
# fig_save_path = 'Figures/correlation and high-res edges.png'
# # fig_save_path = 'Figures/rmse and low-res blur.png'
# fig.savefig(fig_save_path)
plt.show()

