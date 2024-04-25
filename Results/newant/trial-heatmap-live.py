import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from source.utils import rmf, pick_im_matcher, check_for_dir_and_create
from source.antworld2 import Agent
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
from source.imgproc import Pipeline
from source.routedatabase import Route
from source.tools.results import filter_results, read_results, save_to_mat
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

routes_path = 'datasets/new-antworld/curve-bins'


# Plot a specific route
route_id = 12
repeat_no = 0
fig_save_path = os.path.join(fig_save_path, f"route{route_id}", 'full_heatmaps')
check_for_dir_and_create(fig_save_path)
route_path = os.path.join(routes_path, f'route{route_id}')
route = Route(route_path, route_id=route_id, read_imgs=True)
route_imgs = route.get_imgs()
figsize = (7, 7)
title = None


filters = {'route_id':route_id, 'res':'(180, 40)','blur':True, 
           'window':10, 'matcher':'mae', 'edge':False,
           'num_of_repeat': repeat_no}
traj = filter_results(data, **filters)
print(traj.shape[0], ' rows')
traj = traj.to_dict(orient='records')[0]
print(traj.keys())
matched_i = traj['matched_index']
window_log = np.array(traj['window_log'])
ws = window_log[:,0]
we = window_log[:,1]
min_dist_index = traj['min_dist_index']

#pre-proc images
traj['shape'] = eval(traj['res'])
pipe = Pipeline(**traj)
route_imgs = pipe.apply(route_imgs)

# test points, route points, theta (search angle)
tp = len(matched_i)
rp = len(route_imgs)
theta = abs(traj['deg_range'][0] - traj['deg_range'][1])


heatmap_io_path = os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}.npy')
mat_heatmap_io_path = os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}.mat')

if os.path.isfile(heatmap_io_path):
    heatmap = np.load(heatmap_io_path)
else:
    agent = Agent()
    matcher = pick_im_matcher(traj['matcher'])
    # create the heatmap
    heatmap = np.zeros((tp, rp))
    # populate heatmap
    for i in range(tp):
        q_img = agent.get_img((traj['tx'][i], traj['ty'][i]), traj['th'][i])
        q_img = pipe.apply(q_img)
        ridf = rmf(q_img, route_imgs, matcher=matcher)
        ridf_mins = np.min(ridf, axis=1)
        heatmap[i,:] = ridf_mins
    
    save_to_mat(mat_heatmap_io_path, heatmap)
    np.save(heatmap_io_path, heatmap)

fig_size = (10, 5)
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(heatmap, ax=ax)
#ax.imshow(heatmap)
ax.plot(matched_i, range(len(matched_i)), label='navigator match')
ax.plot(min_dist_index, range(len(min_dist_index)), label='optimal match')
ax.plot(ws, range(len(ws)), c='g', label='window limits')
ax.plot(we, range(len(we)), c='g')
# ax.set_xticks([])
# ax.set_yticks([])
xmin, xmax = ax.get_xlim()
ax.hlines(traj.get('tfc_idxs'), xmin=xmin, xmax=xmax, linestyles='dashed', colors='r', label='fail points')

ax.set_xlabel('route images')
ax.set_ylabel('query images')

plt.legend()
plt.tight_layout()
#fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}.pdf'), dpi=200)
fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}.png'), dpi=200)
# plt.show()


# second set fo filters in case you wanna zoop in 
###############################################################
filters2 = {'route_id':route_id, 'res':'(180, 40)','blur':True, 
           'window':10, 'matcher':'mae', 'edge':False,
           'num_of_repeat': repeat_no}
traj2 = filter_results(data, **filters2)
print(traj2.shape[0], ' rows')
traj2 = traj2.to_dict(orient='records')[0]
print(traj2.keys())
traj['tfc_idxs'].extend(traj2['tfc_idxs'])

# Zooom into individual points
ymargin = 50
xmargin = 50
for tfci in traj2['tfc_idxs']:
    fig_size = (10, 10)
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(heatmap, ax=ax)
    #ax.imshow(heatmap)
    ax.plot(matched_i, range(len(matched_i)), label='navigator match')
    ax.plot(min_dist_index, range(len(min_dist_index)), label='optimal match')
    ax.plot(ws, range(len(ws)), c='g', label='window limits')
    ax.plot(we, range(len(we)), c='g')
    # ax.set_xticks([])
    # ax.set_yticks([])
    xmin, xmax = ax.get_xlim()
    ax.hlines(traj.get('tfc_idxs'), xmin=xmin, xmax=xmax, linestyles='dashed', colors='r', label='fail points')

    ax.set_xlabel('route images')
    ax.set_ylabel('query images')

    plt.legend()
    plt.tight_layout()
    
    xmin = min_dist_index[max(0, tfci - xmargin)]
    xmax = min_dist_index[min(tp-1,tfci + xmargin)]
    ymin = max(0, tfci - ymargin)
    ymax = min(tp-1, tfci + ymargin)
    #print([xmin, xmax, ymin, ymax])
    plt.axis([xmin, xmax, ymax, ymin])

    #fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}.pdf'), dpi=200)
    fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-{traj["nav-name"]}-({tfci}).png'), dpi=200)
plt.show()