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
from source.imageproc.imgproc import Pipeline
from source.display import plot_ridf_multiline, plot_3d
from source.utils import rmf, cor_dist, save_image, rotate
from source.routedatabase import Route
from source.analysis import flip_gauss_fit

fig_save_path = os.path.join(cwd, 'projects', 'rmf-analysis')
size = (6, 3)

path =  os.path.join(cwd, 'datasets', 'new-antworld', 'curve_bins', 'route1')
path = '/home/efkag/ant_world_alg_bench/datasets/new-antworld/curve-bins/route1'
route = Route(path, route_id=1)
# df = pd.read_csv(fwd + '/office/training.csv')
# testdf = pd.read_csv(fwd + '/office/testing.csv')

# route = df.to_dict('list')
imgs = route.get_imgs()
params = {'blur': True,
        'shape': (180, 80), 
        #'edge_range': (180, 200)
        }
pipe = Pipeline(**params)
imgs = pipe.apply(imgs)
search_angle = (-180, 180)
rsim = rmf(imgs[300], imgs[301], d_range=(-180, 180))

g_curve = flip_gauss_fit(rsim, d_range=search_angle)
x = np.arange(*search_angle)


fig = plt.figure(figsize=size)
plt.plot(x, rsim, label='RMF function')
plt.plot(x, g_curve, label='Gaussian curve')
plt.xlabel('degrees')
plt.ylabel('MAE/Gaussian function value')
plt.legend()
fig.tight_layout()
fig_save_path = os.path.join(fig_save_path, 'rmf-gauss-fit.png')
fig.savefig(fig_save_path)
plt.show()