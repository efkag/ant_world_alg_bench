import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, animated_window
sns.set_context("paper", font_scale=1)

directory = '2022-06-13'
fig_save_path = os.path.join('Results','newant', directory, 'window-plots')
results_path = os.path.join('Results','newant', directory)
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)
# data['window_log'] = data['window_log'].apply(to_list)


# Plot a specific route
route_id = 1
fig_save_path = os.path.join(fig_save_path, str(route_id))
path = 'new-antworld/exp1/route' + str(route_id) + '/'
window = 20
matcher = 'corr'
edge = '(180, 200)'
blur = False
res = '(180, 80)'
figsize = (4, 4)

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id) & (data['blur'] == blur)]

traj = traj.to_dict(orient='records')[0]
errors = np.array(traj['errors'])
window_log = literal_eval(traj['window_log'])
traj = {'x': np.array(traj['tx']), 'y': np.array(traj['ty']), 'heading': np.array(traj['th'])}

route = load_route_naw(path, route_id=route_id)
route['qx'] = traj['x']
route['qy'] = traj['y']

animated_window(route, window_log, path=fig_save_path)

