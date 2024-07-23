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
    imgs_gen = read_img_gen(data_path, route['filename'])
    im0 = next(imgs_gen)
    unwraper = Unwraper(im0)
    im0 = unwraper.unwarp(im0)
    im0 = pipe.apply(im0)
    imgs.append(im0)
    for im in imgs_gen:
        im = unwraper.unwarp(im)
        if pipe:
            im = pipe.apply(im)
        imgs.append(im)

    route['imgs'] = imgs
    return route

# pm_logs = ['pm0', 'pm1', 'pm2', 'pm3', 'pm4'] 
# asmw_logs = ['asmw0', 'asmw1', 'asmw2', 'asmw3', 'asmw4'] 

#Params
route_id=2
trial_name = '20230918_170735'
#pm_trial_name = pm_logs[1]


combo = {'shape':(180, 50), 'histeq':True}
pipe = Pipeline(**combo)



#route_path = os.path.join(fwd, '2023-09-11', f'route{route_id}')
route_path = os.path.join(fwd, '2023-09-18/demo')
route_path = '/its/home/sk526/ftl-trials-temp/2023-10-06/training'
sample_route_save_path = os.path.join(route_path, 'down-sampled')
check_for_dir_and_create(sample_route_save_path)

# route data
print(f'reading route from{route_path}')
route = load_testing_logs(route_path, pipe=pipe)
ref_imgs = route['imgs']

# plt.imshow(ref_imgs[1900], cmap='gray')
# plt.show()

for im, fi in zip(ref_imgs, route['filename']):
    file_name = os.path.join(sample_route_save_path, fi.strip())
    cv.imwrite(file_name, im)



# #trial data
# logs_path = os.path.join(route_path, 'testing')
# print(f'reading logs from{logs_path}', trial_name)
# trial = load_testing_logs(logs_path, trial_name, pipe=pipe )
# trial_imgs = trial['imgs']


# sample_trial_save_path = os.path.join(route_path, 'down-sampled', 'testing', trial_name)
# check_for_dir_and_create(sample_trial_save_path)

# for im, fi in zip(trial_imgs, trial['filename']):
#     file_name = os.path.join(sample_trial_save_path, fi.strip())
#     cv.imwrite(file_name, im)

# plt.imshow(trial_imgs[3000], cmap='gray')
# plt.show()