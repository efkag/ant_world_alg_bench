import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from source.analysis import log_error_points
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
sns.set_context("paper", font_scale=1)


fig_save_path = 'Results/newant/2022-02-14'
data = pd.read_csv('Results/newant/2022-02-14/results.csv')
# data = pd.read_csv('exp4.csv')
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
window = -20
matcher = 'corr'
edge = '(180, 200)'
blur =  False
res = '(180, 50)'
threshold = 0
figsize = (10, 10)
title = 'D'

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id) & (data['blur'] == blur)]
traj = traj.to_dict(orient='records')[0]

traj['x'] = traj.pop('tx')
traj['y'] = traj.pop('ty')
traj['heading'] = np.array(traj.pop('th'))

route = load_route_naw(path, route_id=route_id)
plot_route(route, traj, scale=70, size=figsize, save=False, path=fig_save_path, title=title)

log_error_points(route, traj, thresh=threshold)

