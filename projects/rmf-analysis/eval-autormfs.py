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
from source.utils import rmf, cor_dist, save_image, mse
from source.routedatabase import Route
from source.analysis import  eval_pair_rmf, d2i_eval
from source.imgproc import mod_dtype

type_change = mod_dtype(np.int16)

fig_save_path = os.path.join(cwd, 'projects', 'rmf-analysis')
size = None
thresh = 60

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)

imgs = route.get_imgs()
imgs = [type_change(im) for im in imgs]

# evaluate imge rmfs against themselves
fit_errors = eval_pair_rmf(imgs)


fig_save_path = os.path.join(fig_save_path, 'autormfs-gauss-eval.png')
fig = plt.figure(figsize=size)
plt.scatter(range(len(fit_errors)), np.log(fit_errors))
plt.axhline(y=np.log(thresh), color='r', linestyle='--', label='threshold')
plt.xlabel('route index')
plt.ylabel('log MSE w.r.t Gaussian curve ')
plt.legend()
# for i, txt in enumerate(fit_errors):
#     plt.annotate(i, (fit_errors[i], fit_errors[i]))
fig.savefig(fig_save_path)
plt.show()

print(np.argwhere(fit_errors > thresh).flatten())

###########################################################
thresh = 0.0034
imgs = route.get_imgs()
imgs = [type_change(im) for im in imgs]
d2i_evals = d2i_eval(imgs)

#fig_save_path = os.path.join(fig_save_path, 'autormfs-d2i-eval.png')
fig = plt.figure(figsize=size)
plt.scatter(range(len(d2i_evals)), d2i_evals)
plt.axhline(y=thresh, color='r', linestyle='--', label='threshold')
plt.xlabel('route index')
plt.ylabel('depth to integral ratio')
plt.legend()
# for i, txt in enumerate(fit_errors):
#     plt.annotate(i, (fit_errors[i], fit_errors[i]))
#fig.savefig(fig_save_path)
plt.show()

print(np.argwhere(d2i_evals > thresh).flatten())