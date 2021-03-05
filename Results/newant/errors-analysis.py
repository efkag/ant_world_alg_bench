import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route
sns.set_context("paper", font_scale=1)



def to_array(x):
    return np.fromstring(x[1:-1], dtype=np.int, sep=' ').tolist()

fig_save_path = 'violins.png'
data = pd.read_csv('exp1.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
route_id = 3
path = '../../new-antworld/exp1/route' + str(route_id) + '/'
window = 25
threshold = 30

traj = data.loc[(data['matcher'] == 'mae') & (data['res'] == '(180, 50)')]
traj = traj.loc[(data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]
errors = np.array(traj['errors'])
traj = {'x': np.array(traj['tx']), 'y': np.array(traj['ty']), 'heading': np.array(traj['th'])}

route = load_route_naw(path, route_id)
index = np.argwhere(errors > threshold)
traj['x'] = traj['x'][index]
traj['y'] = traj['y'][index]
traj['heading'] = traj['heading'][index]
plot_route(route, traj, size=(6, 6))
