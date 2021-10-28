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
from source.utils import load_route_naw, plot_route, seq_angular_error, animated_window, pre_process
from source import seqnav
from source import perfect_memory as pm

path = 'ODK_office/odk-mask.pickle'
with open(path, 'rb') as handle:
    mask = pickle.load(handle)

route_id = 1
path = 'new-antworld/exp1/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=1, imgs=True, query=True, max_dist=0.1)

matcher = 'mae'
nav = pm.PerfectMemory(route['imgs'], matcher)
recovered_heading = nav.navigate(route['qimgs'])

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
traj['heading'] = np.array(traj['heading'])
plot_route(route, traj, save=True)

# route images
imgs = route['imgs']
imgs = [cv.resize(im, (256, 35)) for im in imgs]
imgs = [im.astype(np.int16) for im in imgs]
imgsm = [ma.masked_array(im, mask) for im in imgs]

# test images
qimgs = route['qimgs']
qimgs = [cv.resize(im, (256, 35)) for im in qimgs]
qimgs = [im.astype(np.int16) for im in qimgs]
qimgsm = [ma.masked_array(im, mask) for im in qimgs]

matcher = 'mae'
nav = pm.PerfectMemory(imgsm, matcher)
recovered_heading = nav.navigate(qimgs)

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
traj['heading'] = np.array(traj['heading'])
plot_route(route, traj, save=True)