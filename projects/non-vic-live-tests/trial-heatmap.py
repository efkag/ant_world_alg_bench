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
from source.imgproc import resize
from source.utils import check_for_dir_and_create
from source.tools.matchers import mae, rmf, cor_dist
from source.tools import torchmatchers
sns.set_context("paper", font_scale=1)

def read_img_gen(data_path, img_files):
    for i, fi in enumerate(img_files):
        im_path = os.path.join(data_path, fi.strip())
        img = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
        yield img

def load_testing_logs(data_path, dname='', pipe=None):
    data_path = os.path.join(data_path, dname)
    csv_path = os.path.join(data_path, 'database_entries.csv')
    dt = pd.read_csv(csv_path, index_col=False)
    dt.rename(str.strip, axis='columns', inplace=True, errors="raise")
    route = dt.to_dict('list')
    route['x'] = np.array(route.pop('X [mm]'))
    route['y'] = np.array(route.pop('Y [mm]'))
    route['yaw'] = np.array(route.pop('Heading [degrees]'))
    route['filename'] = route.pop('Filename')
    if route.get('Best snapshot index'):
        route['matched_index'] = route.pop('Best snapshot index')
    if route.get('Window start'):
        route['ws'] = route.pop('Window start')
        route['we'] = route.pop('Window end')
    imgs = []
    # imgs_gen = read_img_gen(data_path, route['filename'])
    im_path = os.path.join(data_path, route['filename'][0].strip())
    img0 = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
    unwraper = Unwraper(img0)
    resizer = resize((360, 90))
    for i in route['filename']:
        
        im_path = os.path.join(route_path, i.strip())
        img = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
        img = unwraper.unwarp(img)
        if pipe:
            img = pipe.apply(img)
        else:
            img = resizer(img)
        imgs.append(img)
    route['imgs'] = imgs
    return route

# pm_logs = ['pm0', 'pm1', 'pm2', 'pm3', 'pm4'] 
# asmw_logs = ['asmw0', 'asmw1', 'asmw2', 'asmw3', 'asmw4'] 

#Params
#route_id=2
#pm_best_match = True
#or
secondary_best_match_simu = True
window_heatmap = False
trial_name = '20160211_172745'


rmf_func = torchmatchers.rmf
rmf_matcher = torchmatchers.mae
combo = {'shape':(180, 50),'vcrop':0.5, 'histeq':True}
pipe = Pipeline(**combo)



directory = '2023-10-06'
route_path = os.path.join(fwd, directory)
fig_save_path = os.path.join(route_path, 'analysis')
check_for_dir_and_create(fig_save_path)

# route data
print(f'reading route from{route_path}')
route = load_testing_logs(route_path, dname='training' ,pipe=pipe)
ref_imgs = route['imgs']
ref_imgs = np.asarray(ref_imgs)



#trial data
logs_path = os.path.join(route_path, 'testing')
print(f'reading logs from{logs_path}')
trial = load_testing_logs(logs_path, dname=trial_name, pipe=pipe)
trial_imgs = trial['imgs']




# pm trial
# if pm_best_match and not secondary_best_match_simu:
#     #trial data
#     logs_path = os.path.join(route_path, 'testing')
#     pm_trial = load_testing_logs(logs_path, pm_trial_name )
#     secondary_matched = pm_trial['matched_index']

# # use this for the HEAT map using the PM trial images
# logs_path = os.path.join(route_path, 'testing')
# pm_trial = load_testing_logs(logs_path, pm_trial_name )
# trial_imgs = pm_trial['imgs']
# trial_imgs = pipe.apply(trial_imgs)





#file_path = os.path.join(fig_save_path,f'heatmap-route({route_id})-trial({trial_name}).npy')
file_path =  os.path.join(fig_save_path, 'heatmap.npy')
if os.path.isfile(file_path):
    heatmap = np.load(file_path)
else:
    fill_heat_value = 0.0
    heatmap = np.full((len(trial_imgs), len(ref_imgs)), fill_heat_value) 
    if window_heatmap:
        #this populated the heatmap fro the windows only
        for i, (im, ws, we) in enumerate(zip(trial_imgs, trial['ws'], trial['we'])):
            w_imgs = ref_imgs[ws:we]
            #get the RDF field
            rdff = rmf_func(im, w_imgs, matcher=rmf_matcher, d_range=(-90, 90))
            ridf_mins = np.min(rdff, axis=1)
            heatmap[i, ws:we] = ridf_mins
    else:
        for i, im in enumerate(trial_imgs):
            #get the RDF field
            rdff = rmf_func(im, ref_imgs, matcher=rmf_matcher, d_range=(-90, 90))
            ridf_mins = np.min(rdff, axis=1)
            heatmap[i,:] = ridf_mins
        np.save(file_path, heatmap)



if secondary_best_match_simu:
    secondary_matched = np.argmin(heatmap, axis=1)

matched_i = trial['matched_index']
ws = trial['ws']
we = trial['we']


fig_size = (4, 3)
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(heatmap, ax=ax)
#ax.imshow(heatmap, cmap='hot')
ax.plot(matched_i, range(len(matched_i)), label='ASMW match')
if secondary_best_match_simu:
    ax.plot(secondary_matched, range(len(secondary_matched)), c='k', label='PM match')
ax.plot(ws, range(len(ws)), c='g', label='window limits')
ax.plot(we, range(len(we)), c='g')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('route images')
ax.set_ylabel('query images')


plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, f'heatmap-route(-)-trial({trial_name})-pmline({secondary_best_match_simu}).png'), dpi=200)
fig.savefig(os.path.join(fig_save_path, f'heatmap-route(-)-trial({trial_name})-pmline({secondary_best_match_simu}).pdf'), dpi=200, rasterize=True)

plt.show()
