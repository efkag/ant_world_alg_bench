import sys
import os

from numpy.lib.npyio import save
path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)

import numpy as np
import numpy.ma as ma
import pickle
import cv2 as cv
from source.utils import load_route_naw, plot_route, seq_angular_error, animated_window, pre_process, check_for_dir_and_create
from source import seqnav
from source import perfect_memory as pm
from source.analysis import log_error_points

path = 'ODK_office/odk-mask.pickle'
with open(path, 'rb') as handle:
    mask = pickle.load(handle)

route_id = 1
path = 'new-antworld/exp1/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True, query=True, max_dist=0.1)

# matcher = 'mae'
# nav = pm.PerfectMemory(route['imgs'], matcher, deg_range=(-90, 90))
# recovered_heading = nav.navigate(route['qimgs'])

# traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
# traj['heading'] = np.array(traj['heading'])
# plot_route(route, traj)

# route images
imgs = route['imgs']
imgs = [cv.resize(im, (256, 35)) for im in imgs]
imgs = [im.astype(np.float64) for im in imgs]
imgsm = [ma.masked_array(im, mask) for im in imgs]
imgsm = [im.filled(np.nan) for im in imgsm]
route['imgs'] = imgsm 

# test images
qimgs = route['qimgs']
qimgs = [cv.resize(im, (256, 35)) for im in qimgs]
qimgs = [im.astype(np.float64) for im in qimgs]
qimgsm = [ma.masked_array(im, mask) for im in qimgs]
qimgsm = [im.filled(np.nan) for im in qimgsm]
route['qimgs'] = qimgsm

matcher = 'nanmae'
nav = pm.PerfectMemory(imgsm, matcher, deg_range=(-90, 90))
recovered_heading = nav.navigate(qimgs)

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
traj['heading'] = np.array(traj['heading'])
plot_route(route, traj)

path = os.path.join(fwd, 'odk-antworld')
check_for_dir_and_create(path, remove=True)
log_error_points(route, traj, nav, thresh=0.0, route_id=route_id, target_path=path)
