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



directory = 'static'
results_path = os.path.join('Results',  'newant', directory)
fig_save_path = os.path.join('Results', 'newant',  directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'combined-results2.csv'), index_col=False)


# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)

# remove PM data
data.drop(data[data['window'] == 0].index, inplace=True)
data['window_log'] = data['window_log'].apply(eval)

check_for_dir_and_create(fig_save_path)
route_id = 2
matcher = 'corr'
edge = 'False'
blur = True
res = '(180, 50)'
g_loc_norm = "False"
# loc_norm = 'False'
data = data.loc[(data['route_id'] == route_id)
                &(data['matcher'] == matcher) 
                #& (data['edge'] == edge) 
                & (data['res'] == res) 
                & (data['blur'] == blur) 
                #& (data['gauss_loc_norm'] == g_loc_norm) 
                #& (data['loc_norm'] == loc_norm)]
                ]

data.describe()

w_size = lambda x: np.diff(x, axis=1).squeeze().tolist()

data['window_log'] = data['window_log'].apply(w_size)

df = data.groupby(['window'])['window_log'].apply(sum).to_frame('window_log').reset_index()
df = df.explode('window_log')

df['window_log'] = pd.to_numeric(df['window_log'])

figsize = (6.5, 3)
fig, ax = plt.subplots(figsize=figsize)
sns.violinplot(data=df, x='window', y='window_log', ax=ax, cut=0)
#window_labels = ['Adaptive SMW', 'PM', 'Fixed 15', 'Fixed 25']
# ax.set_xticklabels(window_labels)
ax.set_ylabel('Window size')
ax.set_xlabel('navigation algorithm')
plt.tight_layout()

fig_path = os.path.join(fig_save_path, f'm{matcher}.res{res}.b{blur}.e{edge}.png')
#fig.savefig(fig_path)
plt.show()

