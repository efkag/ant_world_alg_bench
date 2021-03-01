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
data = pd.read_csv('exp1.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(to_array)


'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=(6,2))
plt.title('mae, route 3')
route1 = data.loc[(data['matcher'] == 'mae') & (data['route_id'] == 3)]
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
fig, ax = plt.subplots()
plt.title('mae, route 3, miss-match')
route1 = data.loc[(data['matcher'] == 'mae') & (data['route_id'] == 3)]
# Group then back to dataframe
route1 = route1.groupby(['window'])['abs_index_diff'].apply(sum).to_frame('abs_index_diff').reset_index()
v_data = route1['abs_index_diff'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=v_data, cut=0, ax=ax)
labels = route1['window'].tolist()
ax.set_xticklabels(labels)
ax.set_ylabel('Index missmatch')
ax.set_xlabel('Window size')
plt.tight_layout(pad=0)
# fig_save_path = 'Figures/correlation and high-res edges.png'
# # fig_save_path = 'Figures/rmse and low-res blur.png'
# fig.savefig(fig_save_path)
plt.show()

