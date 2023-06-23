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


directory = 'ftl/2023-06-22'
results_path = os.path.join('Results', directory)
fig_save_path = os.path.join('Results', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)


check_for_dir_and_create(fig_save_path)
matcher = 'corr'
edge = 'False'
blur = True
res = '(180, 80)'
g_loc_norm = "{'sig1': 2, 'sig2': 20}"
#g_loc_norm = "False"
# loc_norm = 'False'
data = data.loc[(data['matcher'] == matcher) 
                & (data['edge'] == edge) 
                & (data['res'] == res) 
                & (data['blur'] == blur) 
                & (data['gauss_loc_norm'] == g_loc_norm) 
                #& (data['loc_norm'] == loc_norm)]
                ]
# window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']


'''
Plot errors vs window sizes for a combo of parameters
'''
figsize = (7, 3)
fig, ax = plt.subplots(figsize=figsize)
plt.title('m{}.res{}.b{}.e{}.gloc{}.png'.format(matcher, res, blur, edge, g_loc_norm))
# Group then back to dataframe
df = data.groupby(['window'])['errors'].apply(sum).to_frame('errors').reset_index()
df = df.explode('errors')
df['errors']=df['errors'].astype('float64')
#temporary meause to abs the values
df['errors']=df['errors'].apply(abs)
sns.violinplot(data=df, x='window', y='errors', cut=0, ax=ax)

#window_labels = ['Adaptive SMW', 'PM', 'Fixed 15', 'Fixed 25']
#ax.set_xticklabels(window_labels)
ax.set_ylabel('Angular error')
ax.set_xlabel('Window size')
plt.tight_layout()

fig_save_path = os.path.join(fig_save_path, 'm{}.res{}.b{}.e{}.gloc{}.png'.format(matcher, res, blur, edge, g_loc_norm))
fig.savefig(fig_save_path)
plt.show()
