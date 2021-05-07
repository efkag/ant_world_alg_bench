import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
sns.set_context("paper", font_scale=1)


fig_save_path = '/home/efkag/Desktop/route'
data = pd.read_csv('test2.csv')
# data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
route_id = 6
fig_save_path = fig_save_path + str(route_id)
check_for_dir_and_create(fig_save_path)
path = '../../new-antworld/exp1/route' + str(route_id) + '/'
window = 30
matcher = 'mae'
edge = 'False'
res = '(90, 25)'
threshold = 0
figsize = (4, 4)
title = 'B'

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]
traj = data.loc[(data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]
if window:
    w_log = literal_eval(traj['window_log'])
errors = np.array(traj['errors'])
traj = {'x': np.array(traj['tx']), 'y': np.array(traj['ty']), 'heading': np.array(traj['th'])}

route = load_route_naw(path, route_id=route_id)
index = np.argwhere(errors > threshold)
thres = {}
thres['x'] = traj['x'][index]
thres['y'] = traj['y'][index]
thres['heading'] = traj['heading'][index]
fig_save_path = fig_save_path + '/route{}.w{}.m{}.res{}.edge{}.thres{}.png'\
    .format(route_id, window, matcher, res, edge, threshold)
plot_route(route, thres, size=figsize, save=False, path=fig_save_path, title=title)

# if window:
#     path = '/home/efkag/Desktop/route' + str(route_id) + '/window-plots/'
#     animated_window(route, w_log, traj=traj, path=path, size=figsize, title='D')
