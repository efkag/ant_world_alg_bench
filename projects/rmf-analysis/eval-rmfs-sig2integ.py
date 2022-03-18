import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from source.display import plot_multiline, plot_3d
from source.utils import rmf, cor_dist, save_image, mse, pre_process, pair_rmf
from source.routedatabase import Route
from source.analysis import  eval_pair_rmf

fig_save_path = os.path.join(cwd, 'projects', 'rmf-analysis')
size = (6, 4)
thresh = 0.0066

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)

sets = {'blur': True, 'shape': (180, 50)}
imgs = pre_process(route.get_imgs(), sets)
route_dict = route.get_route_dict()

rsims = pair_rmf(imgs, imgs, d_range=(-90, 90))

minima = np.min(rsims, axis=1)
maxima = np.max(rsims, axis=1)

depths = maxima - minima

integrals = np.trapz(rsims, axis=1)

sigint = depths/integrals

fig = plt.figure(figsize=size)
plt.scatter(range(len(sigint)), sigint)
plt.axhline(y=thresh, color='r', linestyle='--', label='threshold')
plt.xlabel('route indexes')
plt.ylabel('RMF depth to integral ratio')
plt.tight_layout()
plt.show()

print(np.argwhere(sigint > thresh).flatten())