import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from ast import literal_eval
from source.utils import mae, rmf, cor_dist, save_image, rotate, check_for_dir_and_create
from source.display import plot_ftl_route
from source.routedatabase import BoBRoute
from source.imgproc import Pipeline
import yaml
sns.set_context("paper", font_scale=1)

directory = 'preliminary/asmw2023-10-11'
results_path = os.path.join('Results', 'ftl', directory)
fig_save_path = os.path.join('Results', 'ftl', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']
# after runnnig on perceptron i have to use the local route path
routes_path = '/its/home/sk526/ftl-trial-repeats/asmw-trials'

#data.drop(data[data['nav-name'] == 'InfoMax'].index, inplace=True)

# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
# data['dist_diff'] = data['dist_diff'].apply(literal_eval)
# data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)
data['matched_index'] = data['matched_index'].apply(literal_eval)


# Plot a specific route
route_id = 3
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
route_path = os.path.join(routes_path, f"route{route_id}")
window = -15
matcher = 'mae'
edge = 'False' 
res = '(180, 50)'
#blur = True
#g_loc_norm = "{'sig1': 2, 'sig2': 20}"
#loc_norm = 'False'
threshold = 30
rep_id = 2


figsize = (10,10)
title = None

combo = {'shape':(180, 50),'vcrop':0.5, 'histeq':True}
pipe = Pipeline(**combo)


traj = data.loc[(data['matcher'] == matcher) 
                & (data['res'] == res) 
                #& (data['edge'] == edge) 
                #& (data['blur'] == blur) 
                & (data['window'] == window) 
                #& (data['gauss_loc_norm'] == g_loc_norm) 
                # & (data['loc_norm'] == loc_norm) 
                & (data['route_id'] == route_id)]
matcher = mae
deg_range = (-90, 90)
degrees = np.arange(*deg_range)

### for repeats
traj = traj.loc[traj['rep_id'] == rep_id]

#trial data
traj = data.to_dict(orient='records')[0]
ref_id = traj['ref_route']
errors = traj['errors']
errors = np.array(errors)
trial = {'x': np.array(traj['tx']),
        'y': np.array(traj['ty']),
        'heading': np.array(traj['th']),
        'matched_index': np.array(traj['matched_index'])}

trial_path = os.path.join(routes_path, f"route{route_id}", f'N-{rep_id}')
traj_r = BoBRoute(trial_path, route_id=route_id, read_imgs=True)
trial['imgs'] = pipe.apply(traj_r.get_imgs())
trial_imgs = trial['imgs']



# route data
route_path = os.path.join(routes_path, f"route{route_id}", f'N-{ref_id}')
# route = BoBRoute(route_path, route_id=route_id, read_imgs=True)
# route_d = route.get_route_dict()
# route_d['yaw'] = route_d['yaw']
# # if threshold:
# #     indices = np.argwhere(errors >= threshold).ravel()
# #     print(f'error indices {indices.tolist()}')
# #     traj['x'] = traj['x'][indices]
# #     traj['y'] = traj['y'][indices]
# #     traj['heading'] = traj['heading'][indices]

route = BoBRoute(path=route_path, read_imgs=True)
ref_imgs = route.get_imgs()
route_len = len(ref_imgs)
route_xy = route.get_xycoords()
route_xy = np.array([route_xy['x'], route_xy['y']])
ref_imgs = pipe.apply(ref_imgs)




# here i can get the indices from the threshold filtered data.
trial_i_of_interest = [492, 493, 494, 496, 497, 498, 527, 528, 530, 531, 532, 533]



for ti in trial_i_of_interest:
    # find the bets match based on coordinates
    qxy = [trial['x'][ti], trial['y'][ti]]
    dist = np.squeeze(cdist([qxy], np.transpose(route_xy), 'euclidean'))
    best_i = np.argmin(dist)
    best_ref_im = ref_imgs[best_i]
    q_im = trial_imgs[ti]
    trial_matched_i = trial['matched_index'][ti]
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
    fig.savefig(os.path.join(fig_save_path, f'aliasing-exp-trail_i({ti})-route({route_id})-trial({rep_id}).png'))
    fig.savefig(os.path.join(fig_save_path, f'aliasing-exp-trail_i({ti})-route({route_id})-trial({rep_id}).pdf'))
    #plt.show()