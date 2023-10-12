import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
from source.display import plot_ftl_route
from source.routedatabase import BoBRoute
import yaml
sns.set_context("paper", font_scale=1)

directory = 'preliminary/asmw2023-10-11'
results_path = os.path.join('Results', 'ftl', directory)
fig_save_path = os.path.join('Results', 'ftl', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']
# after runnnig on perceptron i have to use the local route path
routes_path = '/its/home/sk526/ftl-trial-repeats/asmw-trials'

#data.drop(data[data['nav-name'] == 'InfoMax'].index, inplace=True)

# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
# data['dist_diff'] = data['dist_diff'].apply(literal_eval)
# data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
route_id = 3
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
route_path = os.path.join(routes_path, f"route{route_id}")
window = None
matcher = None
edge = 'False' 
res = '(180, 50)'
#blur = True
#g_loc_norm = "{'sig1': 2, 'sig2': 20}"
#loc_norm = 'False'
threshold = 30
rep_id = 2

angle_correction = 90

figsize = (5,5)
title = None


traj = data.loc[(data['matcher'] == matcher) 
                & (data['res'] == res) 
                #& (data['edge'] == edge) 
                #& (data['blur'] == blur) 
                & (data['window'] == window) 
                #& (data['gauss_loc_norm'] == g_loc_norm) 
                # & (data['loc_norm'] == loc_norm) 
                & (data['route_id'] == route_id)]

### for repeats
traj = traj.loc[traj['rep_id'] == rep_id]
traj = data.to_dict(orient='records')[0]
ref_id = traj['ref_route']




errors = traj['errors']
errors = np.array(errors)
traj = {'x': np.array(traj['tx']),
        'y': np.array(traj['ty']),
        'heading': np.array(traj['th']) + angle_correction}




route_path = os.path.join(routes_path, f"route{route_id}", f'N-{ref_id}')
route = BoBRoute(route_path, route_id=route_id, read_imgs=False)
route = route.get_route_dict()
route['yaw'] = route['yaw'] + angle_correction
if threshold:
    indices = np.argwhere(errors >= threshold).ravel()
    print(f'error indices {indices.tolist()}')
    traj['x'] = traj['x'][indices]
    traj['y'] = traj['y'][indices]
    traj['heading'] = traj['heading'][indices]

temp_save_path = os.path.join(fig_save_path, f'route{route_id}.w{window}.m{matcher}.res{res}.thres{threshold}.png')

plot_ftl_route(route, traj, scale=None, size=figsize, save=True, path=temp_save_path, title=title)
