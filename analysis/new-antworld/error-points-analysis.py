import pandas as pd
import numpy as np
from source.analysis import log_error_points
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
sns.set_context("paper", font_scale=1)


fig_save_path = '/home/efkag/Desktop/route'
data = pd.read_csv('../../Results/newant/exp5.csv')
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
fig_save_path = fig_save_path + str(route_id)
check_for_dir_and_create(fig_save_path)
path = '../../new-antworld/exp1/route' + str(route_id) + '/'
window = -20
matcher = 'corr'
edge = '(220, 240)'
res = '(180, 50)'
threshold = 0
figsize = (10, 10)
title = 'D'

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]

traj['x'] = traj['tx']
traj['y'] = traj['ty']
traj['heading'] = traj['th']

route = load_route_naw(path, route_id=route_id)
plot_route(route, traj, scale=70, size=figsize, save=False, path=fig_save_path, title=title)

log_error_points(route, traj)

