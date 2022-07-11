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
sns.set_context("paper", font_scale=1)

directory = '2022-06-13'
fig_save_path = os.path.join('Results', 'newant', directory)
data = pd.read_csv(os.path.join(fig_save_path, 'results.csv'), index_col=False)
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
route_id = 1
fig_save_path = os.path.join(fig_save_path, str(route_id))
check_for_dir_and_create(fig_save_path)
path = 'new-antworld/exp1/route' + str(route_id) + '/'
window = -15
matcher = 'corr'
edge = 'False'
res = '(180, 80)'
threshold = 0
figsize = (10, 10)
title = 'D'

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]
# traj = data.to_dict(orient='records')[0]
if window:
    w_log = literal_eval(traj['window_log'].to_list()[0])

errors = traj['errors'].tolist()
errors = np.array(errors[0])
traj = {'x': np.array(traj['tx'].tolist()[0]),
        'y': np.array(traj['ty'].tolist()[0]),
        'heading': np.array(traj['th'].tolist()[0])}

route = load_route_naw(path, route_id=route_id)
if threshold:
    index = np.argwhere(errors > threshold)[0]
    thres = {}
    thres['x'] = traj['x'][index]
    thres['y'] = traj['y'][index]
    thres['heading'] = traj['heading'][index]

temp_save_path = os.path.join(fig_save_path, 'route{}.w{}.m{}.res{}.edge{}.thres{}.png'\
    .format(route_id, window, matcher, res, edge, threshold))

plot_route(route, traj, scale=70, size=figsize, save=True, path=temp_save_path, title=title)


# if window:
#     temp_path = os.path.join(fig_save_path,'window-plots')
#     animated_window(route, w_log, traj=traj, path=temp_path, size=figsize, title='D')
