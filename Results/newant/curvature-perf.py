import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from source.utils import check_for_dir_and_create, meancurv2d
from source.routedatabase import load_routes
import seaborn as sns
import yaml
sns.set_context("paper", font_scale=1)

directory = '2023-01-25_mid_update'
fig_save_path = os.path.join('Results', 'newant', directory)
data = pd.read_csv(os.path.join(fig_save_path, 'results.csv'), index_col=False)
with open(os.path.join(fig_save_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']
route_ids = params['route_ids']
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
# loc_norm = 'False'
title = 'D'

traj = data.loc[(data['matcher'] == matcher) 
                & (data['res'] == res) & (data['blur'] == blur) 
                #& (data['edge'] == edge) 
                & (data['gauss_loc_norm'] == g_loc_norm) ]
                #& (data['loc_norm'] == loc_norm)]


method = np.mean
grouped = traj.groupby(['window', 'route_id'])["trial_fail_count"].apply(method).to_frame("trial_fail_count").reset_index()

route_curvatures = []
routes = load_routes(routes_path, route_ids)
for route in routes:
    xy = route.get_xycoords()
    k = meancurv2d(xy['x'], xy['y'])
    route_curvatures.append(k)
# then i need to order by k and plot.

# pm_data = grouped.loc[grouped['window'] == 0]
# plt.plot(pm_data['route_id'], pm_data['mean_error'], label='PM')

# Get the curvatures here
route_ids = pd.unique(grouped['route_id'])
curvatures = []
routes = load_routes(routes_path, route_ids, read_imgs=False)
for route in routes:
    route_dict = route.get_route_dict()
    k = meancurv2d(route_dict['x'], route_dict['y'])
    curvatures.append(k)
curvatures = np.array(curvatures)

ind = np.argsort(curvatures)
curvatures = curvatures[ind]

# Plot a line of the median tfc across the repeats for each window size
w_size = pd.unique(data['window'])
fig , ax = plt.subplots(figsize=(8, 5))
for w in w_size:
    w_data = grouped.loc[grouped['window'] == w]
    tfc_sorted = w_data["trial_fail_count"].to_numpy()[ind]
    ax.plot(route_ids, tfc_sorted, label=f'w={w}')
ax.set_xlabel('routes in increasing curvature')
ax.set_ylabel('mean trial fails')
ax.legend()
plt.show()



