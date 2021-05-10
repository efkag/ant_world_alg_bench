import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from source.utils import check_for_dir_and_create
from ast import literal_eval
from source.utils import load_route_naw, plot_route, pre_process
from source.seqnav import SequentialPerfectMemory
import numpy as np

sns.set_context("paper", font_scale=1)


fig_save_path = '/home/efkag/Desktop/dynamic-window'
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
title = 'B'
route_id = 4
fig_save_path = fig_save_path + '/route{}'.format(route_id)
check_for_dir_and_create(fig_save_path)
path = '../../new-antworld/exp1/route' + str(route_id) + '/'
window = -20
matcher = 'corr'
edge = '(220, 240)'
res = '(180, 50)'
figsize = (5, 2)

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) & (data['edge'] == edge) &
                (data['window'] == window) & (data['route_id'] == route_id)]
traj = traj.to_dict(orient='records')[0]
traj['window_log'] = literal_eval(traj['window_log'])
traj['best_sims'] = literal_eval(traj['best_sims'])

route = load_route_naw(path, route_id=route_id, imgs=True, query=True, max_dist=0.2)
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
ax2.set_ylim([0.3, 1.0])
ax2.set_ylabel('cc image distance')
ax1.legend(loc=2)
ax2.legend(loc=0)

fig_save_path = fig_save_path + '/route{}.w{}.m{}.res{}.edge{}.png'\
    .format(route_id, window, matcher, res, edge)
fig.savefig(fig_save_path)
plt.show()
