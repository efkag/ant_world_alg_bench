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
from source.utils import rmf, cor_dist, save_image, pre_process
from source.routedatabase import Route
from source.analysis import flip_gauss_fit

fig_save_path = os.path.join(cwd, 'projects', 'rmf-analysis')
size = (7, 4)

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)

sets = {'blur': True, 'shape': (180, 50)}
imgs = pre_process(route.get_imgs(), sets)
route_dict = route.get_route_dict()

rsims = rmf(imgs[10], imgs[10:30], d_range=(-180, 180))


minima = np.min(rsims, axis=1)
maxima = np.max(rsims, axis=1)

depths = maxima - minima

integrals = np.trapz(rsims, axis=1)

sigint = depths/integrals

fig = plt.figure(figsize=size)
plt.plot(sigint)
plt.xlabel('route indexes')
plt.ylabel('RMF depth to integral ratio')
plt.show()



