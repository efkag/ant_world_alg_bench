import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from source.analysis import log_error_points
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import yaml
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
from source.routedatabase import Route
from source.antworld2 import Agent
sns.set_context("paper", font_scale=1)


directory = 'static-bench/2023-11-23/2023-11-23_pm'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
#routes_path = params['routes_path']


data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)
data['matched_index'] = data['matched_index'].apply(literal_eval)

# Set up params 
# for a specific route
route_id = 5
routes_path = '/its/home/sk526/ant_world_alg_bench/datasets/new-antworld/exp1'
antworld_agent = None
if not antworld_agent:
    grid_path = '/its/home/sk526/ant_world_alg_bench/datasets/new-antworld/grid70'
else:
    grid_path = None

r_path = os.path.join(routes_path ,f'route{route_id}')

# Bench params
window = 0
blur =  True
matcher = 'corr'
edge = '(190, 240)'
loc_norm = 'False' # {'kernel_shape':(5, 5)}
#gauss_loc_norm = "{'sig1': 2, 'sig2': 20}"
res = '(180, 40)'
threshold = 30
#repeat_no = 0
figsize = (6, 6)
title = None

fig_save_path = os.path.join(fig_save_path, f"route{route_id}",  f'w={window}route{route_id}')
check_for_dir_and_create(fig_save_path)

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) 
                & (data['edge'] == edge) 
                & (data['window'] == window) 
                & (data['blur'] == blur)
                #& (data['loc_norm'] == loc_norm) 
                #& (data['gauss_loc_norm'] == gauss_loc_norm)
                & (data['route_id'] == route_id)
                ##### with repeats
                #& (data['num_of_repeat'] == repeat_no)
                ]

traj = traj.to_dict(orient='records')[0]

traj['x'] = traj.pop('tx')
traj['y'] = traj.pop('ty')
traj['heading'] = np.array(traj.pop('th'))
traj['rmfs'] = np.load(os.path.join(results_path, traj['rmfs_file']+'.npy'), allow_pickle=True)
if window == 0:
    traj['window_log'] = None
else:
    traj['window_log'] = eval(traj['window_log'])


#route = load_route_naw(path, imgs=True, route_id=route_id)
route = Route(r_path, route_id=route_id, grid_path=grid_path).get_route_dict()
plot_route(route, traj, scale=70, size=figsize, save=True, path=fig_save_path, title=title)

log_error_points(route, traj, thresh=threshold, target_path=fig_save_path, aw_agent=antworld_agent)

