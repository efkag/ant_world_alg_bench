from cProfile import label
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

grouped = data.groupby(['window', 'route_id'])['mean_error'].apply(np.mean).to_frame('mean_error').reset_index()

# pm_data = grouped.loc[grouped['window'] == 0]
# plt.plot(pm_data['route_id'], pm_data['mean_error'], label='PM')

#w_size = pd.unique(data['window'])
w_size = [0, -15, 25]
for w in w_size:
    smw_data = grouped.loc[grouped['window'] == w]
    plt.plot(smw_data['route_id'], smw_data['mean_error'], label=f'w={w}')

plt.xlabel('routes in increasing curvature')
plt.ylabel('mean angular error (degrees)')
plt.legend()
plt.show()
