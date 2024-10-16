import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
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
route_id = 16
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
path = os.path.join(routes_path, f"route{route_id}")

# parameters
threshold = 0
repeat_no = 0

filters = {'nav-name':'SMW(300)',
           'route_id':route_id, 'res':'(180, 40)','blur':True, 
           'matcher':'mae', 'edge':False,
           'num_of_repeat': repeat_no, 
           }
traj = filter_results(data, **filters)
print(traj.shape[0], ' rows')
traj = traj.to_dict(orient='records')[0]

figsize = (10, 10)
title = None



errors = np.asarray(traj['aae'])

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
scale=None
fig, ax = plt.subplots(figsize=(4, 4))

zoom = (np.mean(route['x']).item(), np.mean(route['y']).item())

plot_route(route, traj, qwidth=0.05, size=figsize, save=False, 
           path=temp_save_path, title=title, ax=ax, zoom=zoom, zoom_factor=7, step=2)
ax.set_ylabel('X[m]')
ax.set_xlabel('Y[m]')

axins = zoomed_inset_axes(ax, zoom=2, loc=1)
plot_route(route, traj, qwidth=0.05, size=figsize, 
           ax=axins, zoom=zoom, zoom_factor=7, step=2)
# sub region of the original plot/axes
xz, yz = -2.20, -3.20 
x1, x2, y1, y2 = xz-1, xz+1, yz-1, yz+1
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

#axins.set_axis_off()
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

#fig.tight_layout()
fig.savefig(temp_save_path)
fig.savefig(os.path.join(fig_save_path, f'route{route_id}_thre({threshold})_{traj["nav-name"]}.pdf'))
plt.show()

# if traj['window_log']:
#     temp_path = os.path.join(fig_save_path,f'window-plots-{traj["nav-name"]}')
#     animated_window(route, traj=traj, path=temp_path, size=figsize, title=None)
