import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, pre_process
from source.seqnav import SequentialPerfectMemory
import numpy as np

sns.set_context("paper", font_scale=1)


fig_save_path = '/home/efkag/Desktop/route'
data = pd.read_csv('combined-results2.csv')
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
path = '../../new-antworld/exp1/route' + str(route_id) + '/'
window = -20
matcher = 'corr'
edge = '(220, 240)'
res = '(180, 50)'
figsize = (4, 4)

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]
traj['window_log'] = literal_eval(traj['window_log'])

route = load_route_naw(path, route_id=route_id, imgs=True, query=True, max_dist=0.2)
# plot_route(route, traj, size=(6, 6), save=False, path=fig_save_path)

route['imgs'] = pre_process(route['imgs'], sets={'edge_range': (220, 240), 'shape': (180, 50)})
nav = SequentialPerfectMemory(route['imgs'], matching=matcher, window=window)
route['qimgs'] = pre_process(route['qimgs'], sets={'edge_range': (220, 240), 'shape': (180, 50)})
recovered_heading, window_log = nav.navigate(route['qimgs'])


w_size = np.diff(traj['window_log'], axis=1)
plt.plot(range(len(traj['abs_index_diff'])), traj['abs_index_diff'], label='index missmatch')
plt.plot(range(len(w_size)), w_size, label='window size')
plt.legend()
# plt.plot(range(len(sims)), sims)
plt.show()
