import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window
sns.set_context("paper", font_scale=1)


fig_save_path = '/home/efkag/Desktop/route'
# data = pd.read_csv('combined-results.csv')
data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
route_id = 4
fig_save_path = fig_save_path + str(route_id)
path = '../../new-antworld/exp1/route' + str(route_id) + '/'
window = -20
matcher = 'corr'
edge = '(220, 240)'
res = '(180, 50)'
threshold = 30
figsize = (4, 4)

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]
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
plot_route(route, thres, size=(6, 6), save=False, path=fig_save_path)


path = '/home/efkag/Desktop/route' + str(route_id) + '/window-plots/'
animated_window(route, w_log, traj=traj, path=path)
