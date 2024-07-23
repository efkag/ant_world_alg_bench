import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np
import pandas as pd
from source.routedatabase import BoBRoute
from source.unwraper import Unwraper
from source.imageproc.imgproc import Pipeline
from source.unwraper import Unwraper
from matplotlib import pyplot as plt
import seaborn as sns
from ast import literal_eval
import yaml
from source.display import plot_ftl_route
from source.utils import mae, rmf, cor_dist, check_for_dir_and_create


directory = 'ftl/2023-09-12'
results_path = os.path.join('Results', directory)
fig_save_path = os.path.join('Results', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
#routes_path = params['routes_path']
#when plotting localy
routes_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
# data['dist_diff'] = data['dist_diff'].apply(literal_eval)
# data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)



route_id = 1
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
# reference repeat route
ref_rep = 2
# rep_id is the repeat route id used for the testing
rep_id = 1
window_heatmap = False
pm_best_match = False

route_path = os.path.join(routes_path, f"route{route_id}", f'N-{ref_rep}')
window = -15
matcher = 'corr'
edge = 'False'
res = '(180, 80)'
blur = True
g_loc_norm = "{'sig1': 2, 'sig2': 20}"
loc_norm = 'False'

# degree range for the RMFs
degreee_range = (-180, 180)


traj = data.loc[(data['matcher'] == matcher) 
                & (data['res'] == res) 
                #& (data['edge'] == edge) 
                & (data['blur'] == blur) 
                & (data['window'] == window) 
                & (data['gauss_loc_norm'] == g_loc_norm) 
                # & (data['loc_norm'] == loc_norm) 
                & (data['route_id'] == route_id)]

# for repeats
traj = traj.loc[data['rep_id'] == rep_id]
#window information
w_log = literal_eval(traj['window_log'].to_list()[0])
w_log = np.array(w_log)

matched_i = literal_eval(traj['matched_index'].to_list()[0])
# first columns is window start
#second column is window end
ws = w_log[:, 0]
we = w_log[:, 1]


combo = {'shape':(180, 50),'vcrop':0.5, 'histeq':True}
pipe = Pipeline(**combo)

pm_logs = ['pm0', 'pm1', 'pm2', 'pm3', 'pm4'] 
asmw_logs = ['asmw0', 'asmw1', 'asmw2', 'asmw3', 'asmw4'] 


# route data
route = BoBRoute(route_path, route_id=route_id)
ref_imgs = route.get_imgs()
ref_imgs = pipe.apply(ref_imgs)

#trial data
rep_path = os.path.join(routes_path, f"route{route_id}", f'N-{rep_id}')
rep_route = BoBRoute(rep_path)
trial_imgs = rep_route.get_imgs()
trial_imgs = pipe.apply(trial_imgs)



heat_value = 0.0
heatmap = np.full((len(trial_imgs), len(ref_imgs)), heat_value)

if window_heatmap:
    #this populated the heatmap fro the windows only
    for i, (im, ws, we) in enumerate(zip(trial_imgs, ws, we)):
        w_imgs = ref_imgs[ws:we]
        #get the RDF field
        rdff = rmf(im, w_imgs, matcher=mae, d_range=(-90, 90))
        ridf_mins = np.min(rdff, axis=1)
        heatmap[i, ws:we] = ridf_mins
else:
    for i, im in enumerate(trial_imgs):
        #get the RDF field
        rdff = rmf(im, ref_imgs, matcher=mae, d_range=(-90, 90))
        ridf_mins = np.min(rdff, axis=1)
        heatmap[i,:] = ridf_mins


#maybe add PM data to this
if pm_best_match:
    window = 0
    matcher = 'corr'
    edge = 'False'
    res = '(180, 80)'
    blur = True
    g_loc_norm = "{'sig1': 2, 'sig2': 20}"
    loc_norm = 'False'


    traj = data.loc[(data['matcher'] == matcher) 
                    & (data['res'] == res) 
                    #& (data['edge'] == edge) 
                    & (data['blur'] == blur) 
                    & (data['window'] == window) 
                    & (data['gauss_loc_norm'] == g_loc_norm) 
                    # & (data['loc_norm'] == loc_norm) 
                    & (data['route_id'] == route_id)]
    traj = traj.loc[data['rep_id'] == rep_id]
    pm_matched_i = literal_eval(traj['matched_index'].to_list()[0])


fig_size = (7, 4)
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(heatmap, ax=ax)
#ax.imshow(heatmap)
ax.plot(matched_i, range(len(matched_i)), label='ASMW match')
if pm_best_match:
    ax.plot(pm_matched_i, range(len(pm_matched_i)), label='PM match')
ax.plot(ws, range(len(ws)), c='g', label='window limits')
ax.plot(we, range(len(we)), c='g')
ax.set_xlabel('route images')
ax.set_ylabel('query images')

plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, f'heatmap-ref({ref_rep})-rep({rep_id}).png'))
plt.show()
