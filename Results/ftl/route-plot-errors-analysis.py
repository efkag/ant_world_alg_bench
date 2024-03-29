import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from source.utils import animated_window, check_for_dir_and_create
from source.display import plot_ftl_route
from source.routedatabase import Route, BoBRoute
import yaml
sns.set_context("paper", font_scale=1)

directory = 'ftl/2023-09-08'
results_path = os.path.join('Results', directory)
fig_save_path = os.path.join('Results', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
#routes_path = params['routes_path']
#when plotting localy
routes_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
# data['dist_diff'] = data['dist_diff'].apply(literal_eval)
# data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
route_id = 1
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
# reference route is always route repeat 1
route_path = os.path.join(routes_path, f"route{route_id}", 'N-1')
window = None
matcher = 'corr'
edge = 'False'
res = '(180, 80)'
blur = True
g_loc_norm = "{'sig1': 2, 'sig2': 20}"
loc_norm = 'False'
threshold = 0
# # rep_id is the repeat route id used for the testing
rep_id = 2

figsize = (10, 10)
title = None


traj = data.loc[(data['matcher'] == matcher) 
                & (data['res'] == res) 
                #& (data['edge'] == edge) 
                & (data['blur'] == blur) 
                & (data['window'] == window) 
                & (data['gauss_loc_norm'] == g_loc_norm) 
                # & (data['loc_norm'] == loc_norm) 
                & (data['route_id'] == route_id)]

## for repeats
if not window:  
    traj = data.loc[data['rep_id'] == rep_id]
else:
    traj = traj.loc[data['rep_id'] == rep_id]

#traj = data.to_dict(orient='records')[0]


errors = traj['errors'].tolist()
errors = np.array(errors[0])
traj = {'x': np.array(traj['tx'].tolist()[0]),
        'y': np.array(traj['ty'].tolist()[0]),
        'heading': np.array(traj['th'].tolist()[0])}

route = BoBRoute(route_path, route_id=route_id)
route = route.get_route_dict()
if threshold:
    index = np.argwhere(errors >= threshold)[0]
    thres = {}
    thres['x'] = traj['x'][index]
    thres['y'] = traj['y'][index]
    thres['heading'] = traj['heading'][index]

# temp_save_path = os.path.join(fig_save_path, 'route{}.w{}.m{}.res{}.edge{}.glocnorm{}.thres{}.png'\
#     .format(route_id, window, matcher, res, edge, g_loc_norm, threshold))
temp_save_path = os.path.join(fig_save_path, 'imax.png')

plot_ftl_route(route, traj, scale=80, size=figsize, save=True, 
               path=temp_save_path, title=title)

if window:
    w_log = literal_eval(traj['window_log'].to_list()[0])

# if window:
#     temp_path = os.path.join(fig_save_path,'window-plots')
#     animated_window(route, w_log, traj=traj, path=temp_path, size=figsize, title=None)
