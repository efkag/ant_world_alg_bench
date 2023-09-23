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
from source.imgproc import Pipeline
from matplotlib import pyplot as plt
import seaborn as sns
from source.display import plot_ftl_route
from source.utils import mae, rmf, cor_dist, check_for_dir_and_create


def load_testing_logs(route_path, dname):
    route_path = os.path.join(route_path, dname)
    data_path = os.path.join(route_path, 'database_entries.csv')
    dt = pd.read_csv(data_path, index_col=False)
    dt.rename(str.strip, axis='columns', inplace=True, errors="raise")
    route = dt.to_dict('list')
    route['x'] = np.array(route.pop('X [mm]'))
    route['y'] = np.array(route.pop('Y [mm]'))
    route['yaw'] = np.array(route.pop('Heading [degrees]'))
    route['filename'] = route.pop('Filename')
    route['matched_index'] = route.pop('Best snapshot index')
    if route.get('Window start'):
        route['ws'] = route.pop('Window start')
        route['we'] = route.pop('Window end')
    imgs = []
    for i in route['filename']:
        #careful the filenames contain a leading space
        im_path = os.path.join(route_path, i.strip())
        img = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
        imgs.append(img)
    unwraper = Unwraper(imgs[0])
    for i, im in enumerate(imgs):
        im = unwraper.unwarp(im)
        imgs[i] = im
    route['imgs'] = imgs
    return route

pm_logs = ['pm0', 'pm1', 'pm2', 'pm3', 'pm4'] 
asmw_logs = ['asmw0', 'asmw1', 'asmw2', 'asmw3', 'asmw4'] 

#Params
route_id=2
pm_best_match = True
#or
pm_simu_best_match = True
window_heatmap = False
trial_name = asmw_logs[1]
pm_trial_name = pm_logs[1]


combo = {'shape':(180, 50),'vcrop':0.5, 'histeq':True}
pipe = Pipeline(**combo)



route_path = os.path.join(fwd, '2023-09-11', f'route{route_id}')
fig_save_path = os.path.join(route_path, 'analysis')
check_for_dir_and_create(fig_save_path)

# route data
route = BoBRoute(path=route_path, read_imgs=True, unwraper=Unwraper)
ref_imgs = route.get_imgs()
ref_imgs = pipe.apply(ref_imgs)

#trial data
logs_path = os.path.join(route_path, 'testing')
trial = load_testing_logs(logs_path, trial_name )
trial_imgs = trial['imgs']
trial_imgs = pipe.apply(trial_imgs)


# pm trial
if pm_best_match and not pm_simu_best_match:
    #trial data
    logs_path = os.path.join(route_path, 'testing')
    pm_trial = load_testing_logs(logs_path, pm_trial_name )
    pm_matched_i = pm_trial['matched_index']

# # use this for the HEAT map using the PM trial images
# logs_path = os.path.join(route_path, 'testing')
# pm_trial = load_testing_logs(logs_path, pm_trial_name )
# trial_imgs = pm_trial['imgs']
# trial_imgs = pipe.apply(trial_imgs)

max_heat_value = 0
heatmap = np.full((len(trial_imgs), len(ref_imgs)), max_heat_value)



file_path = os.path.join(fig_save_path,f'heatmap-route({route_id})-trial({trial_name}).npy')
if os.path.isfile(file_path):
    heatmap = np.load(file_path)
else:
    if window_heatmap:
        #this populated the heatmap fro the windows only
        for i, (im, ws, we) in enumerate(zip(trial_imgs, trial['ws'], trial['we'])):
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
        np.save(file_path, heatmap)



if pm_best_match and pm_simu_best_match:
    pm_matched_i = np.argmin(heatmap, axis=1)


matched_i = trial['matched_index']
ws = trial['ws']
we = trial['we']

fig_size = (7, 4)
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(heatmap, ax=ax)
#ax.imshow(heatmap)
ax.plot(matched_i, range(len(matched_i)), label='ASMW match')
if pm_best_match:
    ax.plot(pm_matched_i, range(len(pm_matched_i)), c='k', label='PM match')
ax.plot(ws, range(len(ws)), c='g', label='window limits')
ax.plot(we, range(len(we)), c='g')
ax.set_xlabel('route images')
ax.set_ylabel('query images')

plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, f'heatmap-route({route_id})-trial({trial_name})-pmline({pm_best_match}).png'))
plt.show()
