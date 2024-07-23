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
from matplotlib import pyplot as plt
import seaborn as sns
from source.display import plot_ftl_route
from source.utils import mae, rmf, cor_dist, check_for_dir_and_create
from source.imgproc import resize
from source.display import plot_multiline


def load_testing_logs(route_path, dname='', img_shape=(360, 180)):
    route_path = os.path.join(route_path, dname)
    data_path = os.path.join(route_path, 'database_entries.csv')
    dt = pd.read_csv(data_path, index_col=False)
    dt.rename(str.strip, axis='columns', inplace=True, errors="raise")
    route = dt.to_dict('list')
    route['x'] = np.array(route.pop('X [mm]'))
    route['y'] = np.array(route.pop('Y [mm]'))
    route['yaw'] = np.array(route.pop('Heading [degrees]'))
    route['filename'] = route.pop('Filename')
    if route.get('matched_index'):
        route['matched_index'] = route.pop('Best snapshot index')
        route['ws'] = route.pop('Window start')
        route['we'] = route.pop('Window end')
    imgs = []
    for i in route['filename']:
        #careful the filenames contain a leading space
        im_path = os.path.join(route_path, i.strip())
        img = cv.imread(im_path, cv.IMREAD_GRAYSCALE)
        imgs.append(img)
    unwraper = Unwraper(imgs[0])
    resizer = resize(img_shape)
    for i, im in enumerate(imgs):
        im = unwraper.unwarp(im)
        im = resizer(im)
        imgs[i] = im
    route['imgs'] = imgs
    return route


route_id=1
combo = {'shape':(180, 50), 'histeq':True}
pipe = Pipeline(**combo)

pm_logs = ['pm0', 'pm1', 'pm2', 'pm3', 'pm4'] 
asmw_logs = ['asmw0', 'asmw1', 'asmw2', 'asmw3', 'asmw4'] 

route_path = os.path.join(fwd, '2023-09-12', f'route{route_id}')
fig_save_path = os.path.join(route_path, 'analysis')
check_for_dir_and_create(fig_save_path)

# route data
route = load_testing_logs(route_path)
ref_imgs = route['imgs']
# plt.imshow(ref_imgs[0])
# plt.show()
# route = BoBRoute(path=route_path, read_imgs=True, unwraper=Unwraper)
# ref_imgs = route.get_imgs()
ref_imgs = pipe.apply(ref_imgs)



#trial data
trial_dir= '20230912_120039-pm0'
logs_path = os.path.join(route_path, 'testing')
trial = load_testing_logs(logs_path, trial_dir)
trial_imgs = trial['imgs']
trial_imgs = pipe.apply(trial_imgs)

trial_imgs = trial_imgs[-4:]

global_mins = np.zeros((len(trial_imgs), len(ref_imgs)))
labels = []
for i, im in enumerate(trial_imgs):
    #get the RDF field
    labels.append(f'test-img({i})')
    rdff = rmf(im, ref_imgs, matcher=mae, d_range=(-90, 90))
    ridf_mins = np.min(rdff, axis=1)
    global_mins[i, :] = ridf_mins

plot_multiline(global_mins, True, labels=labels)