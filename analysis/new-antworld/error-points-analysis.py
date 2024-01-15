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
from source.tools.results import filter_results, read_results
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
from source.routedatabase import Route
#from source.antworld2 import Agent
sns.set_context("paper", font_scale=1)


directory = 'static-bench/2023-11-23/2023-11-23_asmw'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
#routes_path = params['routes_path']


data = read_results(os.path.join(results_path, 'results.csv'))


# Set up params 
# for a specific route
route_id = 5
routes_path = 'datasets/new-antworld/exp1'
antworld_agent = None
if not antworld_agent:
    grid_path = 'datasets/new-antworld/grid70'
else:
    grid_path = None

r_path = os.path.join(routes_path ,f'route{route_id}')

# Bench params

threshold = 0
#repeat_no = 0
figsize = (5, 5)
title = None


filters = {'route_id':route_id, 'res':'(180, 40)','blur':True, 
           'window':-15, 'matcher':'mae', 'edge':'False'}
traj = filter_results(data, **filters)

traj = traj.to_dict(orient='records')[0]

traj['x'] = traj.pop('tx')
traj['y'] = traj.pop('ty')
traj['heading'] = np.array(traj.pop('th'))
traj['rmfs'] = np.load(os.path.join(results_path, traj['rmfs_file']+'.npy'), allow_pickle=True)

fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
#route = load_route_naw(path, imgs=True, route_id=route_id)
route = Route(r_path, route_id=route_id, grid_path=grid_path).get_route_dict()
plot_route(route, traj, scale=70, size=figsize, save=True, path=fig_save_path, title=title)

log_error_points(route, traj, thresh=threshold, target_path=fig_save_path, aw_agent=antworld_agent)

