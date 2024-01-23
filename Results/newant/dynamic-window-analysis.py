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
from source.tools.results import filter_results, read_results
from source.seqnav import SequentialPerfectMemory
import numpy as np

sns.set_context("paper", font_scale=1)


directory = '2024-01-22'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
data = read_results(os.path.join(results_path, 'results.csv'))



# Plot a specific route
title = None
figsize = (10, 6)
route_id = 0
repeat_no = 0
fig_save_path = os.path.join(fig_save_path, f'route{route_id}')
check_for_dir_and_create(fig_save_path)

filters = {'route_id':route_id, 'res':'(180, 40)','blur':True, 
           'window':15, 'matcher':'mae', 'edge':'False', 'num_of_repeat': repeat_no}
traj = filter_results(data, **filters)
print(traj.shape[0], ' rows')

traj = traj.to_dict(orient='records')[0]

# path = 'new-antworld/exp1/route' + str(route_id) + '/'
# route = Route(path, route_id=route_id)
# route = route.get_route_dict()
# plot_route(route, traj, size=(6, 6), save=False, path=fig_save_path)

w_size = np.diff(traj['window_log'], axis=1)

fig, ax1 = plt.subplots(figsize=figsize)
plt.title(title, loc="left")
ax1.plot(range(len(traj['index_diff'])), traj['index_diff'], label='index missmatch')

ax1.plot(range(len(w_size)), w_size, label='window size')
ax1.scatter(range(len(w_size)), w_size)

ymin, ymax = ax1.get_ylim()

ax1.vlines(traj.get('tfc_idxs'), ymin=ymin, ymax=ymax, linestyles='dashed', colors='r', label='fail points')
#ax1.set_ylim([0, 260])
ax1.set_ylabel('route index scale')
ax1.set_xlabel('test points')

ax2 = ax1.twinx()
ax2.plot(range(len(traj['best_sims'])), traj['best_sims'], label='image diff.', color='g')
ax2.scatter(range(len(traj['best_sims'])), traj['best_sims'], color='g')

#ax2.set_ylim([0.0, 1.0])
ax2.set_ylabel(f'{filters["matcher"]} image distance')
ax1.legend(loc=2)
ax2.legend(loc=0)

fig_save_path = os.path.join(fig_save_path, f'aliasing-route{route_id}.{traj["nav-name"]}.png')
fig.savefig(fig_save_path)
plt.show()
