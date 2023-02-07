import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from source.utils import check_for_dir_and_create
from ast import literal_eval
from source.routedatabase import Route
from source.utils import load_route_naw, plot_route, pre_process
from source.seqnav import SequentialPerfectMemory
import numpy as np

sns.set_context("paper", font_scale=1)


directory = '2023-01-20_mid_update'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)

# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
title = None
route_id = 5
fig_save_path = os.path.join(fig_save_path, f'route{route_id}')
check_for_dir_and_create(fig_save_path)
window = -15
matcher = 'corr'
blur = True
edge = None
loc_norm = 'False' # {'kernel_shape':(5, 5)}
gauss_loc_norm = "{'sig1': 2, 'sig2': 20}"
res = '(180, 80)'
figsize = (5, 2)

traj = data.loc[(data['matcher'] == matcher)
                & (data['blur'] == blur) 
                & (data['res'] == res) 
                #& (data['edge'] == edge)
                #& (data['loc_norm'] == loc_norm) 
                & (data['gauss_loc_norm'] == gauss_loc_norm)
                & (data['window'] == window) 
                & (data['route_id'] == route_id) 
                ]

traj = traj.to_dict(orient='records')[0]
traj['window_log'] = literal_eval(traj['window_log'])
traj['best_sims'] = literal_eval(traj['best_sims'])

# path = 'new-antworld/exp1/route' + str(route_id) + '/'
# route = Route(path, route_id=route_id)
# route = route.get_route_dict()
# plot_route(route, traj, size=(6, 6), save=False, path=fig_save_path)

w_size = np.diff(traj['window_log'], axis=1)

fig, ax1 = plt.subplots(figsize=figsize)
plt.title(title, loc="left")
ax1.plot(range(len(traj['abs_index_diff'])), traj['abs_index_diff'], label='index missmatch')
ax1.set_ylim([0, 260])
ax1.plot(range(len(w_size)), w_size, label='window size')
ax1.set_ylabel('route index scale')

ax2 = ax1.twinx()
ax2.plot(range(len(traj['best_sims'])), traj['best_sims'], label='best sim', color='g')
ax2.set_ylim([0.0, 1.0])
ax2.set_ylabel(f'{matcher} image distance')
ax1.legend(loc=2)
ax2.legend(loc=0)

fig_save_path = os.path.join(fig_save_path, 'aliasing-route{}.w{}.m{}.res{}.edge{}.png'\
    .format(route_id, window, matcher, res, edge))
fig.savefig(fig_save_path)
plt.show()
