from cProfile import label
import sys
import os

from numpy import size

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
cwd = os.getcwd()
sys.path.append(cwd)


import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
from source.display import plot_multiline, plot_3d
from source.utils import rmf, cor_dist, save_image, rotate
from source.routedatabase import Route
from source.analysis import flip_gauss_fit

fig_save_path = os.path.join(cwd, 'projects', 'rmf-analysis')
size = (5, 5)

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)
# df = pd.read_csv(fwd + '/office/training.csv')
# testdf = pd.read_csv(fwd + '/office/testing.csv')

# route = df.to_dict('list')
imgs = route.get_imgs()
search_angle = (-90, 90)
rsim = rmf(imgs[10], imgs[10], d_range=(-90, 90))

g_curve = flip_gauss_fit(rsim, d_range=search_angle)
x = np.arange(*search_angle)

fig_save_path = os.path.join(fig_save_path, 'rmf-gauss-fit.png')
fig = plt.figure(figsize=size)
plt.plot(x, rsim, label='RMF function')
plt.plot(x, g_curve, label='Gaussian curve')
plt.xlabel('degrees')
plt.ylabel('MAE/Gaussian function value')
plt.legend()
fig.savefig(fig_save_path)
plt.show()