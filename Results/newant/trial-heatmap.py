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
from source.tools.results import filter_results, read_results
import yaml
sns.set_context("paper", font_scale=1)

# general paths
directory = '2024-03-07'
results_path = os.path.join('Results', 'newant',  directory)
fig_save_path = os.path.join(results_path, 'analysis')

data = read_results(os.path.join(results_path, 'results.csv'))
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']

#routes_path = 'datasets/new-antworld/curve-bins'


# Plot a specific route
route_id = 16
repeat_no = 0
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
route_path = os.path.join(routes_path, f'route{route_id}')
route = Route(route_path, route_id=route_id, read_imgs=True)

figsize = (6, 6)
title = None


filters = {'route_id':route_id, 'num_of_repeat': repeat_no,
           'res':'(180, 40)','blur':True, 
           'window':300, 'matcher':'mae', 'edge':False,
           }
traj = filter_results(data, **filters)
print(traj.shape[0], ' rows')
traj = traj.to_dict(orient='records')[0]
print(traj.keys())
matched_i = traj['matched_index']
window_log = np.array(traj['window_log'])
min_dist_index = traj['min_dist_index']

#Read the pickled file
print(traj['rmfs_file'])
rmfs_path = os.path.join(results_path, 'metadata', f"{traj['rmfs_file']}.npy")
print(rmfs_path)
rmfs = np.load(rmfs_path, allow_pickle=True)
print(rmfs.shape)
print(f'tfc i:{traj["tfc_idxs"]}')


# test points, route points, theta (search angle)
tp = len(rmfs)
rp = window_log.max() - window_log.min()
theta = rmfs[0].shape[1]

# create the heatmap
max_heat_value = 100.
heatmap = np.full((tp, rp), max_heat_value)
# populate heatmap
for i in range(tp):
    #get the window values
    ridf_mins = np.min(rmfs[i], axis=1)
    heatmap[i,window_log[i, 0]:window_log[i, 1]] = ridf_mins


fig_size = figsize
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(heatmap, ax=ax)
#ax.imshow(heatmap)
ax.plot(matched_i, range(len(matched_i)), label='best match')
ax.plot(min_dist_index, range(len(min_dist_index)), label='optimal match')
# ax.plot(ws, range(len(ws)), c='g', label='window limits')
# ax.plot(we, range(len(we)), c='g')
# ax.set_xticks([])
# ax.set_yticks([])
xmin, xmax = ax.get_xlim()
ax.hlines(traj.get('tfc_idxs'), xmin=xmin, xmax=xmax, linestyles='dashed', colors='r', label='fail points')

ax.set_xlabel('route images')
ax.set_ylabel('query images')

plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}.pdf'), dpi=200)
fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}.png'), dpi=200)
plt.show()


# Zooom into individual points
ymargin = 50
xmargin = 50
tfcidxs = traj['tfc_idxs'].copy()
tfcidxs.append(49)
for tfci in tfcidxs:
    fig_size = fig_size
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(heatmap, ax=ax)
    #ax.imshow(heatmap)
    ax.plot(matched_i, range(len(matched_i)), label='best match')
    ax.plot(min_dist_index, range(len(min_dist_index)), label='optimal match')
    # ax.plot(ws, range(len(ws)), c='g', label='window limits')
    # ax.plot(we, range(len(we)), c='g')
    # ax.set_xticks([])
    # ax.set_yticks([])
    xmin, xmax = ax.get_xlim()
    ax.hlines(traj.get('tfc_idxs'), xmin=xmin, xmax=xmax, linestyles='dashed', colors='r', label='fail points')

    ax.set_xlabel('route images')
    ax.set_ylabel('query images')

    plt.legend()
    plt.tight_layout()
    
    xmin = max(0,min_dist_index[tfci] - xmargin)
    xmax = min(rp, min_dist_index[tfci] + xmargin)
    ymin = max(0, tfci - ymargin)
    ymax = min(tp, tfci + ymargin)
    print([xmin, xmax, ymin, ymax])
    plt.axis([xmin, xmax, ymax, ymin])

    plt.legend()
    plt.tight_layout()
    #fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}.pdf'), dpi=200)
    fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}-({tfci}).png'), dpi=200)
    #plt.show()
    plt.close(fig)