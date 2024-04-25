import sys
import os
path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from source.utils import check_for_dir_and_create
from source.display import plot_matches
from source.routedatabase import Route
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)

fig_save_path = 'Results/newant/2022-01-27'
data = pd.read_csv('Results/newant/2022-01-27/results.csv')
# Convert list of strings to actual list of lists
data['matched_index'] = data['matched_index'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)



route_id = 1
path = 'new-antworld/exp1/route' + str(route_id) + '/'
route = Route(path, route_id=route_id)
route = route.get_route_dict()

check_for_dir_and_create(fig_save_path)
window = 0
matcher = 'corr'
edge = 'False'  # 'False'
blur = True
figsize = None
res = '(180, 50)'
traj = data.loc[(data['window'] == window) & (data['matcher'] == matcher) & (data['route_id'] == route_id)
                 & (data['edge'] == edge) & (data['res'] == res)]
traj =  traj.to_dict('list')
matched_index = traj['matched_index'][0]
traj = {'x': np.array(traj['tx'][0]),
        'y': np.array(traj['ty'][0]),
        'heading': np.array(traj['th'][0])}


plot_matches(route, traj, matched_index)