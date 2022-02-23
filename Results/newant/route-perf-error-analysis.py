import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
from source.utils import check_for_dir_and_create
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)



fig_save_path = 'Results/newant/test2'
data = pd.read_csv('Results/newant/test2/results.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)

title = 'B'
route_id = 3
check_for_dir_and_create(fig_save_path)
matcher = 'corr'
edge = 'False'  # 'False'
blur = True
figsize = (4, 2)
res = '(180, 50)'
route = data.loc[(data['matcher'] == matcher) & (data['route_id'] == route_id)
                 & (data['edge'] == edge) & (data['res'] == res)]
# window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']

'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=figsize)
plt.title(title, loc="left")
# Group then back to dataframe
df = route.groupby(['window'])['errors'].apply(sum).to_frame('errors').reset_index()
v_data = df['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=v_data, cut=0, ax=ax)
window_labels = df['window'].tolist()
ax.set_xticklabels(window_labels)
ax.set_ylabel('Angular error')
ax.set_xlabel('Window size')
plt.tight_layout(pad=0)

# fig.savefig(fig_save_path + '/{}.route{}.png'.format(matcher, route_id))
fig_path = os.path.join(fig_save_path, 'route[{}].m{}.res{}.b{}.e{}.png'.format(route_id, matcher, res, blur, edge))
fig.savefig(fig_path)
plt.show()


'''
Plot for one specific matcher with one specific pre-proc
'''
missmatch_metric = 'dist_diff'
# missmatch_metric = 'abs_index_diff'

fig, ax = plt.subplots(figsize=figsize)
plt.title(title, loc="left")
# Group then back to dataframe
df = route.groupby(['window'])[missmatch_metric].apply(sum).to_frame(missmatch_metric).reset_index()
v_data = df[missmatch_metric].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=v_data, cut=0, ax=ax)
window_labels = df['window'].tolist()
ax.set_xticklabels(window_labels)
ax.set_ylabel('Euclidean distance (m)')
ax.set_xlabel('Window size')
plt.tight_layout(pad=0)
# fig.savefig(fig_save_path + '/{}.{}.route{}.png'.format(missmatch_metric, matcher, route_id))
fig_path = os.path.join(fig_save_path, 'route[{}].{}.m{}.res{}.b{}.e{}.png'.format(route_id, missmatch_metric, matcher, res, blur, edge))
fig.savefig(fig_path)
plt.show()

