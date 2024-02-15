import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
sns.set_context("paper", font_scale=1)

directory = '2024-01-2/combined'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
check_for_dir_and_create(fig_save_path)
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
#data['trial_fail_count'] = data['trial_fail_count'].apply(literal_eval)



# Choose a specific pre. processing
# route_id = 7
matcher = 'mae'
blur = True
# edge = 'False' 
res = '(180, 40)'
g_loc_norm = 'False'#"{'sig1': 2, 'sig2': 20}"
# loc_norm = 'False'
title = None

data = data.loc[data['nav-name'] != 'InfoMax']
# imax_df = data.loc[data['nav-name'] == 'InfoMax']
# data = pd.concat([data, imax_df])

data = data.loc[(data['matcher'] == matcher) 
                & (data['res'] == res) 
                & (data['blur'] == blur) 
                #& (data['edge'] == edge) 
                #& (data['gauss_loc_norm'] == g_loc_norm)
                #& (data['loc_norm'] == loc_norm)
                #& (data['route_id'] == route_id ) 
                # & (data['num_of_repeat'] == 0) 
             #& (data['route_id'] <= 4 ) #& (data['route_id'] < 10 )
                
]


#################
# in case of repeats
method = np.mean
#data = data.groupby(['window', 'route_id'])["trial_fail_count"].apply(method).to_frame("trial_fail_count").reset_index()
##### if the dataset had nav-names
data = data.groupby(['nav-name', 'route_id'])["trial_fail_count"].apply(method).to_frame("trial_fail_count").reset_index()

figsize = (6., 3)
fig, ax = plt.subplots(figsize=figsize)
#ax.set_ylim(0, 20)
#sns.barplot(x="window", y="trial_fail_count", data=data, ax=ax, estimator=method, capsize=.2, ci=None)
sns.boxplot(data=data, x="nav-name", y="trial_fail_count",  ax=ax)
# window_labels = ['Adaptive SMW', 'PM', 'Fixed 15', 'Fixed 25']
# ax.set_xticklabels(window_labels)
ax.set_xlabel('Navigation Algorithm')
ax.set_ylabel('Mean TFC')
plt.tight_layout()
# path = os.path.join(fig_save_path, f'route[{route_id}]-failed trials.png')
temp_save_path = os.path.join(fig_save_path, 'failed-trials.png')
fig.savefig(temp_save_path)
temp_save_path = os.path.join(fig_save_path, 'failed-trials.pdf')
fig.savefig(temp_save_path)
#plt.show()


################# joint plot

# fig, axs = plt.subplots(2, 1, figsize=figsize)

# ax = axs[0]
# cols = ['steelblue', 'orange', 'green', 'red', 'purple', 'grey']
# ax.set_title('All Navigation Algorithms')
# sns.barplot(x="nav-name", y="trial_fail_count", data=data, ax=ax, 
#             estimator=method, capsize=.2, ci=None, palette=cols)
# ax.set_xlabel('navigation algorithm')
# ax.set_ylabel('mean TFC')

# ax = axs[1]
# ax.set_title('Temporal Algorithms')
# cols = ['steelblue', 'green', 'red', 'purple', 'grey']
# data = data.drop(data[data['nav-name'] == 'InfoMax'].index)
# sns.barplot(x="nav-name", y="trial_fail_count", data=data, ax=ax, 
#             estimator=method, capsize=.2, ci=None, palette=cols)
# ax.set_xlabel('navigation algorithm')
# ax.set_ylabel('mean TFC')
# plt.tight_layout()
# temp_save_path = os.path.join(fig_save_path, 'failed-trials-joinplot.png')
# fig.savefig(temp_save_path)
# plt.show()
