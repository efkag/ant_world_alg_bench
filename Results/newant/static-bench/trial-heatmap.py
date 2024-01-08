import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
from source.routedatabase import Route
from source.tools.results import filter_results
import yaml
sns.set_context("paper", font_scale=1)


directory = '2023-11-23/2023-11-23_asmw'
results_path = os.path.join('Results', 'newant', 'static-bench',  directory)
fig_save_path = os.path.join(results_path, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# with open(os.path.join(results_path, 'params.yml')) as fp:
#     params = yaml.load(fp)
# routes_path = params['routes_path']
routes_path = '/its/home/sk526/ant_world_alg_bench/datasets/new-antworld/exp1'
grid_path = '/its/home/sk526/ant_world_alg_bench/datasets/new-antworld/grid70'

# Plot a specific route
route_id = 5
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
route_path = os.path.join(routes_path, f'route{route_id}')
route = Route(route_path, route_id=route_id, read_imgs=True, grid_path=grid_path, max_dist=0.2)

g_loc_norm = 'False'#"{'sig1': 2, 'sig2': 20}"
loc_norm = 'False'

figsize = (4, 4)
title = None


filters = {'route_id':route_id, 'res':'(180, 40)','blur':True, 
           'window':-15, 'matcher':'mae', 'edge':'False'}
asmw_traj = filter_results(data, **filters)
asmw_traj = asmw_traj.to_dict(orient='records')[0]
print(asmw_traj.keys())


# pm data 
directory = '2023-11-23/2023-11-23_pm'
results_path = os.path.join('Results', 'newant', 'static-bench',  directory)
fig_save_path = os.path.join(results_path, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)

filters = {'route_id':route_id, 'res':'(180, 40)','blur':True, 
           'window':0, 'matcher':'mae', 'edge':'False'}
pm_traj = filter_results(data, **filters)
pm_traj = pm_traj.to_dict(orient='records')[0]
print(pm_traj.keys())


#Read the pickled file
print(pm_traj['rmfs_file'])
rmfs_path = os.path.join(results_path, f"{pm_traj['rmfs_file']}.npy")
print(rmfs_path)
rmfs = np.load(rmfs_path, allow_pickle=True)
print(rmfs[0].shape)




# fig_size = (4, 3)
# fig, ax = plt.subplots(figsize=fig_size)
# sns.heatmap(heatmap, ax=ax)
# #ax.imshow(heatmap)
# ax.plot(matched_i, range(len(matched_i)), label='ASMW match')
# if pm_best_match:
#     ax.plot(pm_matched_i, range(len(pm_matched_i)), c='k', label='PM match')
# ax.plot(ws, range(len(ws)), c='g', label='window limits')
# ax.plot(we, range(len(we)), c='g')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xlabel('route images')
# ax.set_ylabel('query images')

# plt.legend()
# plt.tight_layout()
# fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-trial({trial_name})-pmline({pm_best_match}).pdf'), dpi=200)
# fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-trial({trial_name})-pmline({pm_best_match}).png'), dpi=200)
# plt.show()
