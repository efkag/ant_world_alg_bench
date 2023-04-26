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

directory = '2023-04-20_test'
fig_save_path = os.path.join('Results', 'newant', directory)
data = pd.read_csv(os.path.join(fig_save_path, 'results.csv'), index_col=False)
# data['trial_fail_count'] = data['trial_fail_count'].apply(literal_eval)


# Choose a specific pre. processing
route_id = 7
matcher = 'corr'
blur = True
# edge = 'False' 
res = '(180, 80)'
g_loc_norm = "{'sig1': 2, 'sig2': 20}"
# loc_norm = 'False'
title = 'D'

data = data.loc[(data['matcher'] == matcher) & (data['res'] == res) 
                & (data['blur'] == blur) 
                #& (data['edge'] == edge) 
                & (data['gauss_loc_norm'] == g_loc_norm)
                #& (data['loc_norm'] == loc_norm)
                & (data['route_id'] == route_id )
                # & (data['route_id'] >= 14 ) #& (data['route_id'] < 14 )
                & (data['num_of_repeat'] == 0) 
]
#################
# in case of repeats
method = sum
data = data.groupby(['window', 'route_id'])["trial_fail_count"].apply(method).to_frame("trial_fail_count").reset_index()

figsize = (5, 3)
fig, ax = plt.subplots(figsize=figsize)
sns.barplot(x="window", y="trial_fail_count", data=data, ax=ax, estimator=sum, capsize=.2, ci=None)
# window_labels = ['Adaptive SMW', 'PM', 'Fixed 15', 'Fixed 25']
# ax.set_xticklabels(window_labels)
plt.tight_layout(pad=0)
path = os.path.join(fig_save_path, f'route[{route_id}]-failed trials.png')
# path = os.path.join(fig_save_path, 'curve4-failed trials.png')
fig.savefig(path)
plt.show()

