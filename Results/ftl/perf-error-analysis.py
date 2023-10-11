import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from source.utils import check_for_dir_and_create
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)


directory = 'ftl/2023-10-06'
results_path = os.path.join('Results', directory)
fig_save_path = os.path.join('Results', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
# data['dist_diff'] = data['dist_diff'].apply(literal_eval)
# data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)


# imax_df = data.loc[data['nav-name'] == 'InfoMax']
# data.drop(data[data['nav-name'] == 'InfoMax'].index, inplace=True)

check_for_dir_and_create(fig_save_path)
route_id = 3
matcher = 'mae'
edge = 'False'
blur = True
res = '(180, 50)'
g_loc_norm = "{'sig1': 2, 'sig2': 20}"
g_loc_norm = "False"
# loc_norm = 'False'
data = data.loc[(data['matcher'] == matcher) 
                #& (data['edge'] == edge) 
                #& (data['res'] == res) 
                #& (data['blur'] == blur) 
                #& (data['gauss_loc_norm'] == g_loc_norm) 
                #& (data['route_id'] == route_id)
                #& (data['loc_norm'] == loc_norm)]
                ]
# window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']
#data = pd.concat([data, imax_df])
thresh = 0

'''
Plot errors vs window sizes for a combo of parameters
'''
figsize = (7, 3)
fig, ax = plt.subplots(figsize=figsize)
#plt.title('m{}.res{}.b{}.e{}.gloc{}.png'.format(matcher, res, blur, edge, g_loc_norm))
# Group then back to dataframe
df = data.groupby(['nav-name', 'res'])['errors'].apply(sum).to_frame('errors').reset_index()
df = df.explode('errors')
df['errors']=df['errors'].astype('float64')
#temporary meause to abs the values
if thresh:
    df = df.loc[df['errors'] >= thresh]
sns.violinplot(data=df, x='nav-name', hue='res', y='errors', cut=0, ax=ax)

# window_labels = ['Adaptive SMW', 'PM', 'Fixed 15', 'Fixed 25', 'Fixed 25']
# ax.set_xticklabels(window_labels)
ax.set_ylabel('AAE')
ax.set_xlabel('navigation algorithm')
plt.tight_layout()

save_path = os.path.join(fig_save_path, 'm{}.res{}.b{}.e{}.gloc{}.png'.format(matcher, res, blur, edge, g_loc_norm))
fig.savefig(save_path)
plt.show()



'''
PLot count of AAE > x
'''

thresh = 45

df = data.groupby(['nav-name'])['errors'].apply(sum).to_frame('errors').reset_index()
df = df.explode('errors')
if thresh:
    df = df.loc[df['errors'] >= thresh]

dkeys = pd.unique(df['nav-name'])
counts = []
for k in dkeys:
    counts.append(df[(df['nav-name'] == k) & (df['errors'] >= thresh)].count()[0])

df = pd.DataFrame.from_dict({'nav-name':dkeys.tolist(), 'count':counts})

figsize = (7, 3)
fig, ax = plt.subplots(figsize=figsize)

sns.barplot(data=df, x='nav-name', y='count', ax=ax)

# window_labels = ['Adaptive SMW', 'PM', 'Fixed 15', 'Fixed 25', 'Fixed 25']
# ax.set_xticklabels(window_labels)
ax.set_ylabel(f'count of AAE > {thresh}')
ax.set_xlabel('navigation algorithm')
plt.tight_layout()

save_path = os.path.join(fig_save_path, f'AAE>{thresh}.m{matcher}.res{res}.b{blur}.e{edge}.gloc{g_loc_norm}.png')
fig.savefig(save_path)
plt.show()

# '''
# Plot scatter of different algos for a pre-proc setting
# '''

# figsize = (7, 3)
# fig, ax = plt.subplots(figsize=figsize)
# df = data.groupby(['window'])['errors'].apply(sum).to_frame('errors').reset_index()
# df = df.explode('errors')
# x = df.loc[df['window'] == 0]['errors'].to_numpy()
# y = df.loc[df['window'] == -15]['errors'].to_numpy()
# ax.scatter(x, y)
# ax.set_ylabel('ASMW AAE')
# ax.set_xlabel('PM AAE')
# plt.tight_layout()
# save_path = os.path.join(fig_save_path, f'scatter-m{matcher}.res{res}.b{blur}.e{edge}.gloc{g_loc_norm}.png')
# fig.savefig(save_path)
# plt.show()
