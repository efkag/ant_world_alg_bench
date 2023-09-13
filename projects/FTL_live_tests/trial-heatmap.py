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
    route['ws'] = route.pop('Window start')
    route['we'] = route.pop('Window end')
    imgs = []
    for i in route['filename']:
        #careful the filenames contain a leading space
        im_path = os.path.join(route_path, i.strip())
        img = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
        imgs.append(img)
    route['imgs'] = imgs
    return route

route_id=1

combo = {'shape':(180, 50),'vcrop':0.5, 'histeq':True}
pipe = Pipeline(**combo)

pm_logs = ['pm0', 'pm1', 'pm2', 'pm3', 'pm4'] 
asmw_logs = ['asmw0', 'asmw1', 'asmw2', 'asmw3', 'asmw4'] 

route_path = os.path.join(fwd, 'ftl-live-tests', f'r{route_id}')
fig_save_path = os.path.join(route_path, 'analysis')
check_for_dir_and_create(fig_save_path)

# route data
route = BoBRoute(path=route_path, read_imgs=True, unwraper=Unwraper)
ref_imgs = route.get_imgs()
ref_imgs = pipe.apply(ref_imgs)

#trial data
logs_path = os.path.join(route_path, 'testing')
trial = load_testing_logs(logs_path, asmw_logs[0])
trial_imgs = trial['imgs']
trial_imgs = pipe.apply(trial_imgs)

max_heat_value = 80
heatmap = np.full((len(trial_imgs), len(ref_imgs)), max_heat_value)


for i, (im, ws, we) in enumerate(zip(trial_imgs, trial['ws'], trial['we'])):
    w_imgs = ref_imgs[ws:we]
    #get the RDF field
    rdff = rmf(im, w_imgs, matcher=mae, d_range=(-90, 90))
    ridf_mins = np.min(rdff, axis=1)
    heatmap[i, ws:we] = ridf_mins

matched_i = trial['matched_index']
ws = trial['ws']
we = trial['we']

fig_size = (7, 4)
fig, ax = plt.subplots(figsize=fig_size)
sns.heatmap(heatmap, ax=ax)
#ax.imshow(heatmap)
ax.plot(matched_i, range(len(matched_i)))
ax.plot(ws, range(len(ws)), c='g')
ax.plot(we, range(len(we)), c='g')
ax.set_xlabel('route images')
ax.set_ylabel('query images')
plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, 'heatmap.png'))
plt.show()
