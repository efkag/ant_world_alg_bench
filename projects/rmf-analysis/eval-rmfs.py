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
from source.utils import rmf, cor_dist, save_image
from source.routedatabase import Route
from source.analysis import flip_gauss_fit

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)

imgs = route.get_imgs()[:20]
search_angle = (-90, 90)
rsim = rmf(imgs[10], imgs[10], d_range=(-90, 90))