from random import gauss
import sys
import os

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


path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)
# df = pd.read_csv(fwd + '/office/training.csv')
# testdf = pd.read_csv(fwd + '/office/testing.csv')

# route = df.to_dict('list')
imgs = route.get_imgs()[:20]
search_angle = (-90, 90)
rsim = rmf(imgs[10], imgs[10], d_range=(-90, 90))

def flip_gauss_fit(rsim, drange=(-180, 180), eta=0.65):
    # eta = np.radians(eta)
    degrees = np.arange(drange[0], drange[1])
    mu = degrees[np.argmin(rsim)]
    minimum = np.min(rsim)
    # depth of the RMF shape
    depth = np.max(rsim) - minimum
    # flipped gaussian fit
    # delta angles from the mean 
    d_angles = degrees-mu
    # fit the flipped gaussian to the RMF
    g_fit = depth*(1 - np.exp(-(d_angles**2)/(2*(eta**2)))) + minimum
    return g_fit


g_curve = flip_gauss_fit(rsim, drange=search_angle)
x = np.arange(len(rsim))
plt.plot(x, rsim)
plt.plot(x, g_curve)
plt.show()