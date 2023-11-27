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



directory = '2023-11-22/combined'
results_path = os.path.join('Results',  'newant', directory)
fig_save_path = os.path.join('Results', 'newant',  directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)

#####
#data.drop(data[data['nav-name'] == 'InfoMax'].index, inplace=True)

# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)


check_for_dir_and_create(fig_save_path)
matcher = 'corr'
edge = 'False'
blur = True
res = '(180, 40)'
g_loc_norm = "False"
# loc_norm = 'False'
data = data.loc[(data['matcher'] == matcher) 
                #& (data['edge'] == edge) 
                & (data['res'] == res) 
                & (data['blur'] == blur) 
                #& (data['gauss_loc_norm'] == g_loc_norm) 
                #& (data['loc_norm'] == loc_norm)]
                ]


'''
Plot errors vs window sizes for a combo of parameters
'''
df = data.groupby(['window', 'nav-name'])['errors'].apply(sum).to_frame('errors').reset_index()
df = df.explode('errors')
df['errors'] = pd.to_numeric(df['errors'])

figsize = (6, 3)
fig, ax = plt.subplots(figsize=figsize)
# plt.title('m{}.res{}.b{}.e{}.gloc{}.png'.format(matcher, res, blur, edge, g_loc_norm))
# Group then back to dataframe

sns.violinplot(data=df, x='nav-name', y='errors', ax=ax, cut=0)
#sns.boxplot(data=df, x='nav-name', y='errors', ax=ax,)
#window_labels = ['Adaptive SMW', 'PM', 'Fixed 15', 'Fixed 25']
# ax.set_xticklabels(window_labels)
ax.set_ylim([-1, 180])
ax.set_ylabel('AAE')
ax.set_xlabel('navigation algorithm')
plt.tight_layout()

fig_path = os.path.join(fig_save_path, 'm{}.res{}.b{}.e{}.gloc{}.png'.format(matcher, res, blur, edge, g_loc_norm))
fig.savefig(fig_path)
plt.show()



'''
Plot aliasing metric vs window sizes for a combo of parameters
'''
#missmatch_metric = 'dist_diff'
missmatch_metric = 'abs_index_diff'
df = data.groupby(['window', 'nav-name'])[missmatch_metric].apply(sum).to_frame(missmatch_metric).reset_index()
df = df.explode(missmatch_metric)
df[missmatch_metric] = pd.to_numeric(df[missmatch_metric])

figsize = (6, 3)
fig, ax = plt.subplots(figsize=figsize)
# plt.title(title, loc="left")
sns.violinplot(data=df, x='nav-name', y=missmatch_metric, ax=ax, cut=0)
#ax.set_ylim([0, 5])
ax.set_ylabel('index difference')
ax.set_xlabel('navigation algorithm')
plt.tight_layout()
# fig.savefig(fig_save_path + '/{}.{}.route{}.png'.format(missmatch_metric, matcher, route_id))
fig_path = os.path.join(fig_save_path, f'aliasing[{missmatch_metric}].m{matcher}.res{res}.b{blur}.e{edge}.png')
fig.savefig(fig_path)
plt.show()