from cProfile import label
import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from source.utils import check_for_dir_and_create
from source.routedatabase import Route
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
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)

#### all data
aae = data['mean_error']
tfc = data['trial_fail_count']
mdiv = data['mdivergence']

fig, ax = plt.subplots(figsize=(8, 6))
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

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(aae, tfc, label='aae-tfc')
ax.set_xlabel('mean aae')
ax.set_ylabel('tfc')
ax.legend(loc=2)
ax1 = ax.twinx()
ax1.scatter(aae, mdiv, label='aae-mdiv', color='g')
ax1.set_ylabel('mean divergence')
ax1.legend(loc=1)
plt.show()

#### route with specific performance metric values
# data = data.loc[data['mean_error'] > 60]
# row = data.iloc[0].to_dict()
# route_id = row['route_id']
# with open(os.path.join(fig_save_path, 'params.yml')) as fp:
#     params = yaml.load(fp)
# routes_path = params['routes_path']
# routepath = os.path.join(routes_path, f"route{route_id}")
# route = Route(routepath, route_id=route_id)
# route = route.get_route_dict()
# traj = {'x': np.array(row['tx']),
#         'y': np.array(row['ty']),
#         'heading': np.array(row['th'])}

#### Route specific
route_id = 3
# window = 0
data = data.loc[data['route_id'] == route_id]
aae = data['mean_error']
tfc = data['trial_fail_count']
mdiv = data['mdivergence']

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(aae, tfc, label='aae-tfc')
ax.set_xlabel('mean aae')
ax.set_ylabel('tfc')
ax.legend(loc=2)
ax1 = ax.twinx()
ax1.scatter(aae, mdiv, label='aae-mdiv', color='g')
ax1.set_ylabel('mean divergence')
ax1.legend(loc=1)
plt.show()