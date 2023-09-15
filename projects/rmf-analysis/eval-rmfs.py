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
from source.display import plot_ridf_multiline, plot_3d
from source.utils import rmf, cor_dist, save_image, mse
from source.routedatabase import Route
from source.analysis import eval_rmf_fit

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)

search_angle = (-90, 90)

imgs = route.get_imgs()[:10]
er1 = eval_rmf_fit(imgs[5], imgs, d_range=search_angle)

imgs = route.get_imgs()[87:97]
er2 = eval_rmf_fit(imgs[5], imgs, d_range=search_angle)

plt.plot(er1)
plt.plot(er2)
plt.show()

plt.scatter(er1, er1)
plt.scatter(er2, er2)
plt.show()