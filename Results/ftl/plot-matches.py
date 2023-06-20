import sys
import os
path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from source.utils import check_for_dir_and_create
from source.display import plot_matches
from source.routedatabase import Route, BoBRoute
import yaml
from ast import literal_eval
import seaborn as sns
sns.set_context("paper", font_scale=1)


directory = 'ftl/2023-05-26'
results_path = os.path.join('Results', directory)
fig_save_path = os.path.join('Results', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
route_path = params.get('routes_path')
# Convert list of strings to actual list of lists
data['matched_index'] = data['matched_index'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)



route_id = 1
route_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
route_path =  os.path.join(route_path, 'route{}'.format(route_id))
suffix = 'N-'
if suffix:
        route_path = os.path.join(route_path, suffix)
# the referencee route is always 0, i.e the first route recorded
route_path = route_path + str(route_id)
route = BoBRoute(route_path, read_imgs=False,  route_id=route_id)
route = route.get_route_dict()

check_for_dir_and_create(fig_save_path)
window = 0
matcher = 'corr'
edge = 'False'
blur = True
res = '(180, 80)'
g_loc_norm = "{'sig1': 2, 'sig2': 20}"

traj = data.loc[(data['window'] == window)
                & (data['route_id'] == route_id)
                & (data['res'] == res)
                & (data['blur'] == blur) 
                & (data['matcher'] == matcher) 
                & (data['edge'] == edge)
                & (data['gauss_loc_norm'] == g_loc_norm) 
                ]
traj =  traj.to_dict('list')
matched_index = traj['matched_index'][0]
traj = {'x': np.array(traj['tx'][0]),
        'y': np.array(traj['ty'][0]),
        'heading': np.array(traj['th'][0])}

figsize = None
plot_matches(route, traj, matched_index)