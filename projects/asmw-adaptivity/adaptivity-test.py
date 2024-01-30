import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import yaml
from source.utils import cor_dist, mae, check_for_dir_and_create, scale2_0_1
from source.analysis import flip_gauss_fit, eval_gauss_rmf_fit, d2i_rmfs_eval
from collections import deque
from source.tools.results import read_results, filter_results
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
           'window':-15, 'matcher':'mae', 'edge':'False', 'num_of_repeat': repeat_no}
traj = filter_results(data, **filters)
print(traj.shape[0], ' rows')
traj = traj.to_dict(orient='records')[0]


best_sims = traj['best_sims']

perc_chng = np.diff(best_sims) / best_sims[1:]
perc_chng = perc_chng[np.abs(perc_chng) < 1]

# plt.plot(perc_chng)

fig, ax  = plt.subplots()
ax.hist(perc_chng, bins=100)
ax.set_ylabel('density')
ax.set_xlabel('best sims percentage change')
fig_save_path = os.path.join(fwd, f'sims_dist.{traj["nav-name"]}.png')
fig.savefig(fig_save_path)
plt.show()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#some common window params
min_window = 10
window = None
w_thresh = 0.05

def thresh_log_update(prev_sim, curr_sim, window, thresh=0.1):
    # threshold against the percentage change of match quality
    perc_cng = (curr_sim - prev_sim)/prev_sim
    if perc_cng > thresh or window <= min_window:
        window += round(min_window/np.log(window))
    else:
        window -= round(np.log(window))
    return window


def sma_update(sma_val, curr_sim, window):
    if curr_sim > sma_val or window <= min_window:
        window += round(min_window/np.log(window))
    else:
        window -= round(np.log(window))
    return window


# for an SMA update
best_sims = traj['best_sims']
sma_sims = moving_average(best_sims, 3)
sma_sims = np.insert(sma_sims, 0, np.mean(best_sims[:2]))
sma_sims = np.append(sma_sims, np.mean(best_sims[-2:]))


# for a new update
new_window_log = []
window = traj['window_log'][0][1] - traj['window_log'][0][0]
for i in range(len(traj['best_sims'])-1):
    curr_sim = traj['best_sims'][i]
    prev_sim = traj['best_sims'][i-1]
    #latest sma val
    sma_val = sma_sims[i]
    # add here a new criterion for window update
    window = thresh_log_update(prev_sim, curr_sim, window, thresh=w_thresh)
    #window = sma_update(sma_val, curr_sim, window)
    new_window_log.append(window)


w_size = np.diff(traj['window_log'], axis=1)
#PLot the results


fig, ax1 = plt.subplots(figsize=figsize)
plt.title(title, loc="left")
ax1.plot(range(len(traj['index_diff'])), traj['index_diff'], label='index missmatch')

ax1.plot(range(len(w_size)), w_size, label='window size')
#ax1.scatter(range(len(w_size)), w_size)
ax1.plot(range(len(new_window_log)), new_window_log, label=f'thres = {w_thresh}')
#ax1.plot(range(len(new_window_log)), new_window_log, label=f'sma update')

ymin, ymax = ax1.get_ylim()

ax1.vlines(traj.get('tfc_idxs'), ymin=ymin, ymax=ymax, linestyles='dashed', colors='r', label='fail points')
#ax1.set_ylim([0, 260])
ax1.set_ylabel('route index scale')
ax1.set_xlabel('test points')

ax2 = ax1.twinx()
ax2.plot(range(len(traj['best_sims'])), traj['best_sims'], label='image diff.', color='g')
#ax2.scatter(range(len(traj['best_sims'])), traj['best_sims'], color='g')

#ax2.set_ylim([0.0, 1.0])
ax2.set_ylabel(f'{filters["matcher"]} image distance')
ax1.legend(loc=2)
ax2.legend(loc=0)

fig_save_path = os.path.join(fwd, f'aliasing-route{route_id}.{traj["nav-name"]}.png')
fig.savefig(fig_save_path)
plt.show()