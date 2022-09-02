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
sns.set_context("paper", font_scale=1)

directory = '2022-07-26_mid_update'
fig_save_path = os.path.join('Results', 'newant', directory)
data = pd.read_csv(os.path.join(fig_save_path, 'results.csv'), index_col=False)

# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(eval)
data['dist_diff'] = data['dist_diff'].apply(eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(eval)
#data["trial_fail_count"] = data["trial_fail_count"].apply(eval)

# choose a specific pre-processing
# Choose a specific pre. processing
matcher = 'corr'
edge = 'False'
blur =  True 
res = '(180, 80)'
g_loc_norm = "{'sig1': 2, 'sig2': 20}"
loc_norm = 'False'
title = 'D'

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['blur'] == blur) &
                (data['edge'] == edge) & (data['gauss_loc_norm'] == g_loc_norm)
                & (data['loc_norm'] == loc_norm)]



grouped = traj.groupby(['window', 'route_id'])["trial_fail_count"].apply(sum).to_frame("trial_fail_count").reset_index()

# pm_data = grouped.loc[grouped['window'] == 0]
# plt.plot(pm_data['route_id'], pm_data['mean_error'], label='PM')

#w_size = pd.unique(data['window'])
w_size = [0, -15, 20, 25]
for w in w_size:
    smw_data = grouped.loc[grouped['window'] == w]
    plt.plot(smw_data['route_id'], smw_data["trial_fail_count"], label=f'w={w}')

plt.xlabel('routes in increasing curvature')
plt.ylabel('mean trial fails')
plt.legend()
plt.show()
