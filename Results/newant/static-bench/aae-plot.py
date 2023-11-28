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


directory = 'static-bench/2021-04-06'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join(results_path, 'analysis')
check_for_dir_and_create(fig_save_path)

data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)


check_for_dir_and_create(fig_save_path)
matcher = 'corr'
edge = '(220, 240)'  # 'False'
blur = False
g_loc_norm = "False"
# loc_norm = 'False'

figsize = (6, 3)
res = '(180, 50)'
route = data.loc[(data['matcher'] == matcher) 
                 & (data['edge'] == edge) 
                 & (data['res'] == res) 
                 & (data['blur'] == blur)]
window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']

'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=figsize)
#plt.title(matcher + ', route:' + str(route_id))
# Group then back to dataframe
df = route.groupby(['nav-name'])['errors'].apply(sum).to_frame('errors').reset_index()
df = df.explode('errors')
df['errors'] = pd.to_numeric(df['errors'])

sns.violinplot(data=df, x='nav-name', y='errors', cut=0, ax=ax)

ax.set_ylim([-1, 180])
ax.set_ylabel('AAE')
ax.set_xlabel('navigation algorithm')
plt.tight_layout()
fig_path = os.path.join(fig_save_path, f'm{matcher}.res{res}.b{blur}.e{edge}.png')
fig.savefig(fig_path)
fig_path = os.path.join(fig_save_path, f'm{matcher}.res{res}.b{blur}.e{edge}.pdf')
fig.savefig(fig_path)
plt.show()