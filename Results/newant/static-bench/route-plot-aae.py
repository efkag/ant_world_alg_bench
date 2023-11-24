import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
from source.routedatabase import Route
sns.set_context("paper", font_scale=1)

routes_path = '/its/home/sk526/ant_world_alg_bench/new-antworld/exp1'
fig_save_path = os.path.join(fwd, 'analysis')
check_for_dir_and_create(fig_save_path)
results_path = os.path.join(fwd, 'combined-results2.csv')
data = pd.read_csv(results_path)
# data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
route_id = 5
fig_save_path = os.path.join(fig_save_path, f'route{route_id}')
check_for_dir_and_create(fig_save_path)

route_path = os.path.join(routes_path, f'route{route_id}')
window = 0
matcher = 'corr'
edge = '(220, 240)'
res = '(180, 50)'
threshold = 20
figsize = (4, 4)
title = None

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]
if window:
    w_log = literal_eval(traj['window_log'])
errors = np.array(traj['errors'])
traj = {'x': np.array(traj['tx']), 'y': np.array(traj['ty']), 'heading': np.array(traj['th'])}

#route = load_route_naw(route_path, route_id=route_id)
route = Route(route_path, route_id=route_id, read_imgs=False)
route = route.get_route_dict()
index = np.argwhere(errors > threshold)
thres = {}
thres['x'] = traj['x'][index]
thres['y'] = traj['y'][index]
thres['heading'] = traj['heading'][index]
fig_save_path = fig_save_path + '/route{}.w{}.m{}.res{}.edge{}.thres{}.png'\
    .format(route_id, window, matcher, res, edge, threshold)

fig, ax = plt.subplots(figsize=figsize)
plot_route(route, thres, title=title, ax=ax)

################################################
route_id = 5
window = 30
matcher = 'corr'
edge = '(220, 240)'
res = '(180, 50)'
threshold = 20
figsize = (4, 4)
title = None

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]
nav_name = traj["nav-name"]
if window:
    w_log = literal_eval(traj['window_log'])
errors = np.array(traj['errors'])
traj = {'x': np.array(traj['tx']), 'y': np.array(traj['ty']), 'heading': np.array(traj['th'])}

ax.scatter(traj['x'], traj['y'], label=f'{nav_name}')
###############################################


plt.legend()
fig.tight_layout()
fig.savefig(fig_save_path)
plt.show()
plt.close(fig)

# if window:
#     path = '/home/efkag/Desktop/route' + str(route_id) + '/window-plots/'
#     animated_window(route, w_log, traj=traj, path=path, size=figsize, title='D')