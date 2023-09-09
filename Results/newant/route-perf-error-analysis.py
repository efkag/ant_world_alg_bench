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



directory = '2023-04-26/combined'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)

########
data.drop(data[data['nav-name'] == 'InfoMax'].index, inplace=True)

# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)

title = ''
route_id = None
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
matcher = 'corr'
edge = 'False'  # 'False'
blur = True
res = '(180, 80)'
g_loc_norm = "False"
loc_norm = 'False'
route = data.loc[(data['matcher'] == matcher) 
                 #& (data['edge'] == edge) 
                 & (data['res'] == res) 
                 & (data['blur'] == blur) 
                 & (data['gauss_loc_norm'] == g_loc_norm) 
                 #& (data['loc_norm'] == loc_norm) 
                 & (data['route_id'] == route_id)]
# print(route)                 
# window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']

repeat_no = 0
route = route.loc[route['num_of_repeat'] == repeat_no]
'''
Plot for one specific matcher with one specific pre-proc
'''

df = route.groupby(['window', 'nav-name'])['errors'].apply(sum).to_frame('errors').reset_index()
df = df.explode('errors')
df['errors'] = pd.to_numeric(df['errors'])

figsize = (6, 3)
fig, ax = plt.subplots(figsize=figsize)
plt.title(title, loc="left")
# Group then back to dataframe

#v_data = df['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
#sns.violinplot(data=v_data, cut=0, ax=ax)
sns.violinplot(data=df, x='nav-name', y='errors', ax=ax, cut=0)
#window_labels = df['window'].tolist()
#ax.set_xticklabels(window_labels)
ax.set_ylabel('AAE')
ax.set_xlabel('navigation algorithm')
plt.tight_layout()

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

