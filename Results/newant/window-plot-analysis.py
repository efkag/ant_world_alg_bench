import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from source2 import load_route_naw, animated_window
sns.set_context("paper", font_scale=1)


def to_list(x):
    return literal_eval(x)

fig_save_path = 'window-plots/'
data = pd.read_csv('combined-results.csv')
# data = pd.read_csv('test.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)
# data['window_log'] = data['window_log'].apply(to_list)


# Plot a specific route
route_id = 4
fig_save_path = fig_save_path + str(route_id)
path = '../../new-antworld/exp1/route' + str(route_id) + '/'
window = -20
matcher = 'corr'
edge = '(220, 240)'
res = '(180, 50)'
figsize = (4, 4)

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & # (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]

traj = traj.to_dict(orient='records')[0]
errors = np.array(traj['errors'])
window_log = literal_eval(traj['window_log'])
traj = {'x': np.array(traj['tx']), 'y': np.array(traj['ty']), 'heading': np.array(traj['th'])}

route = load_route_naw(path, route_id=route_id)
route['qx'] = traj['x']
route['qy'] = traj['y']

animated_window(route, window_log, path=fig_save_path)

