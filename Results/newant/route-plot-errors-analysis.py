import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
from source.utils import load_route_naw, animated_window, check_for_dir_and_create
from source.tools.display import plot_route
from source.routedatabase import Route
from source.tools.results import filter_results, read_results
import yaml
sns.set_context("paper", font_scale=1)

directory = '2024-03-07'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
data = read_results(os.path.join(results_path, 'results.csv'))
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']

data.drop(data[data['nav-name'] == 'InfoMax'].index, inplace=True)


# Plot a specific route
route_id = 8
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
path = os.path.join(routes_path, f"route{route_id}")

# parameters
threshold = 0
repeat_no = 0

filters = {'nav-name':'PM',
           'route_id':route_id, 'res':'(180, 40)','blur':True, 
           'matcher':'mae', 'edge':False,
           'num_of_repeat': repeat_no, 
           }
traj = filter_results(data, **filters)
print(traj.shape[0], ' rows')
traj = traj.to_dict(orient='records')[0]

figsize = (10, 10)
title = None



errors = traj['aae']

#errors = np.array(errors[0])
traj['x'] = np.array(traj['tx'])
traj['y'] = np.array(traj['ty'])
traj['heading'] = np.array(traj['th'])
traj['min_dist_index'] = np.array(traj['min_dist_index'])

route = Route(path, route_id=route_id)
route = route.get_route_dict()
if threshold:
    index = np.argwhere(errors >= threshold).ravel()
    traj['x'] = traj['x'][index]
    traj['y'] = traj['y'][index]
    traj['heading'] = traj['heading'][index]
    traj['min_dist_index'] = traj['min_dist_index'][index]

print(traj.keys())
temp_save_path = os.path.join(fig_save_path, f'route{route_id}_thre({threshold})_{traj["nav-name"]}.png')

print(temp_save_path)

fig, ax = plt.subplots(figsize=(4, 4))

zoom = (np.mean(route['x']).item(), np.mean(route['y']).item())

plot_route(route, traj, scale=100, size=figsize, save=False, 
           path=temp_save_path, title=title, ax=ax, zoom=None, zoom_factor=7)
fig.tight_layout()
fig.savefig(temp_save_path)
plt.show()

# if traj['window_log']:
#     temp_path = os.path.join(fig_save_path,f'window-plots-{traj["nav-name"]}')
#     animated_window(route, traj=traj, path=temp_path, size=figsize, title=None)
