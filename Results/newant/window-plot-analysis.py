import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, animated_window
sns.set_context("paper", font_scale=1)


def to_list(x):
    return literal_eval(x)

fig_save_path = 'violins.png'
data = pd.read_csv('exp1.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)
# data['window_log'] = data['window_log'].apply(to_list)


# Plot a specific route
route_id = 3
path = '../../new-antworld/exp1/route' + str(route_id) + '/'
window = 25
threshold = 30

traj = data.loc[(data['matcher'] == 'mae') & (data['res'] == '(180, 50)')]
traj = traj.loc[(data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]
errors = np.array(traj['errors'])
window_log = literal_eval(traj['window_log'])
traj = {'x': np.array(traj['tx']), 'y': np.array(traj['ty']), 'heading': np.array(traj['th'])}

route = load_route_naw(path, route_id=route_id)
route['qx'] = traj['x']
route['qy'] = traj['y']

path = 'window_plots/'
animated_window(route, window_log, path=path)

