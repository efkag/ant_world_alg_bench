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
from source.imgproc import Pipeline
from source.utils import mae, rmf, rotate

import yaml
sns.set_context("paper", font_scale=1)

# general paths
directory = '2023-11-23'
results_path = os.path.join('Results', 'newant', 'static-bench',  directory)
fig_save_path = os.path.join(results_path, 'analysis')


routes_path = 'datasets/new-antworld/exp1'
grid_path = 'datasets/new-antworld/grid70'

# Plot a specific route
route_id = 5
grid_dist = 0.2
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
route_path = os.path.join(routes_path, f'route{route_id}')
route = Route(route_path, route_id=route_id, read_imgs=True, grid_path=grid_path, max_dist=grid_dist)


figsize = (4, 4)
title = None


# pm data 
directory = '2023-11-23/2023-11-23_pm'
results_path = os.path.join('Results', 'newant', 'static-bench',  directory)

data = read_results(os.path.join(results_path, 'results.csv'))
filters = {'route_id':route_id, 'res':'(180, 40)','blur':True, 
           'window':0, 'matcher':'mae', 'edge':'False'}
traj = filter_results(data, **filters)
print(traj.shape[0], ' rows')
traj = traj.to_dict(orient='records')[0]
print(traj.keys())



# This need to be the same as the filters
combo = {'shape':(180, 40), 'blur':True}
pipe = Pipeline(**combo)

#get images etc
ref_imgs = route.get_imgs()
ref_imgs = pipe.apply(ref_imgs)
trial_imgs = route.get_qimgs()
trial_imgs = pipe.apply(trial_imgs)

matcher = mae
deg_range = (-180, 180)
degrees = np.arange(*deg_range)


trial_i_of_interest = [15, 16, 17, 26, 28, 41, 67]



for ti in trial_i_of_interest:
    # find the bets match based on coordinates
    # qxy = [trial['x'][ti], trial['y'][ti]]
    # dist = np.squeeze(cdist([qxy], np.transpose(route_xy), 'euclidean'))
    # best_i = np.argmin(dist)
    best_i = traj['min_dist_index'][ti]
    best_ref_im = ref_imgs[best_i]
    q_im = trial_imgs[ti]
    trial_matched_i = traj['matched_index'][ti]
    matched_ref_im = ref_imgs[trial_matched_i]
    print(f'trial image index {ti},', f'best optimal ref image {best_i},', f'trial matched index{trial_matched_i}' )

    #get the RIDFS, their minima and headings
    q_best_ridf = rmf(q_im, best_ref_im, matcher=matcher, d_range=deg_range)
    q_best_ridf_i = np.argmin(q_best_ridf)
    q_best_ridf_h = degrees[q_best_ridf_i]
    #######################################################
    q_matched_ridf = rmf(q_im, matched_ref_im, matcher=matcher, d_range=deg_range)
    q_matched_ridf_i = np.argmin(q_matched_ridf)
    q_matched_ridf_h = degrees[q_matched_ridf_i]
    
    # Roatate the image to the minima
    q_im_best_rot = rotate(q_best_ridf_h, q_im)
    q_im_matched_rot = rotate(q_matched_ridf_h, q_im)



    fig = plt.figure(figsize=(7, 6))
    #plt.suptitle('')
    rows = 4
    cols = 2

    # query img
    ax = fig.add_subplot(rows, 1, 1)
    ax.set_title(f'query image ({ti})')
    ax.imshow(q_im, cmap='gray')
    ax.set_axis_off()

    # opt match img
    ax = fig.add_subplot(rows, cols, 3)
    ax.set_title(f'optimal route image (route index ({best_i})')
    ax.imshow(best_ref_im, cmap='gray')
    ax.set_axis_off()

    # opt residual image
    ax = fig.add_subplot(rows, cols, 4)
    ax.set_title(f'optimal residual image')
    res_img = np.abs(q_im_best_rot - best_ref_im)
    ax.imshow(res_img, cmap='hot')
    ax.set_axis_off()

    # matched img
    ax = fig.add_subplot(rows, cols, 5)
    ax.set_title(f'matched route image (route index ({trial_matched_i}))')
    ax.imshow(matched_ref_im, cmap='gray')
    ax.set_axis_off()

    # matched residual image    
    ax = fig.add_subplot(rows, cols, 6)
    ax.set_title(f'matched residual image')
    res_img = np.abs(q_im_matched_rot - matched_ref_im)
    ax.imshow(res_img, cmap='hot')
    ax.set_axis_off()

    ax = fig.add_subplot(rows, 1, 4)
    ax.plot(degrees, q_best_ridf, label='optimal match RIDF')
    ax.plot(degrees, q_matched_ridf, label='matched RIDF')
    ax.set_xlabel('Degrees')
    ax.set_ylabel('MAE')
    plt.tight_layout()
    plt.legend()
    fig.savefig(os.path.join(fig_save_path, f'aliasing-exp-trail_i({ti})-route({route_id}).png'))
    fig.savefig(os.path.join(fig_save_path, f'aliasing-exp-trail_i({ti})-route({route_id}).pdf'))
    #plt.show()