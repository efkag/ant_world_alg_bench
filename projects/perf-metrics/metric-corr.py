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
from ast import literal_eval
sns.set_context("paper", font_scale=1)


directory = '2022-09-20_mid_update'
fig_save_path = os.path.join('Results', 'newant', directory)
data = pd.read_csv(os.path.join(fig_save_path, 'results.csv'), index_col=False)
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['divergence'] = data['divergence'].apply(literal_eval)
data['mdivergence'] = data['divergence'].apply(np.mean)

#### all data
aae = data['mean_error']
tfc = data['trial_fail_count']
mdiv = data['mdivergence']

fig, ax = plt.subplots()
ax.scatter(aae, tfc, label='aae-tfc')
ax.set_xlabel('mean aae')
ax.set_ylabel('tfc')
ax.legend(loc=2)
ax1 = ax.twinx()
ax1.scatter(aae, mdiv, label='aae-mdiv', color='g')
ax1.set_ylabel('mean divergence')
ax1.legend(loc=1)
plt.show()

#### filter data
# window = 0
matcher = 'corr'
edge = 'False'
blur = True
res = '(180, 80)'
g_loc_norm = "{'sig1': 2, 'sig2': 20}"
loc_norm = 'False'
data = data.loc[(data['matcher'] == matcher) & (data['edge'] == edge) &
                 (data['res'] == res) & (data['blur'] == blur) &
                 (data['gauss_loc_norm'] == g_loc_norm) & 
                 (data['loc_norm'] == loc_norm)]

aae = data['mean_error']
tfc = data['trial_fail_count']
mdiv = data['mdivergence']

fig, ax = plt.subplots()
ax.scatter(aae, tfc, label='aae-tfc')
ax.set_xlabel('mean aae')
ax.set_ylabel('tfc')
ax.legend(loc=2)
ax1 = ax.twinx()
ax1.scatter(aae, mdiv, label='aae-mdiv', color='g')
ax1.set_ylabel('mean divergence')
ax1.legend(loc=1)
plt.show()

#### Route specific
route_id = 1
# window = 0
data = data.loc[data['route_id'] == route_id]
aae = data['mean_error']
tfc = data['trial_fail_count']
mdiv = data['mdivergence']

fig, ax = plt.subplots()
ax.scatter(aae, tfc, label='aae-tfc')
ax.set_xlabel('mean aae')
ax.set_ylabel('tfc')
ax.legend(loc=2)
ax1 = ax.twinx()
ax1.scatter(aae, mdiv, label='aae-mdiv', color='g')
ax1.set_ylabel('mean divergence')
ax1.legend(loc=1)
plt.show()