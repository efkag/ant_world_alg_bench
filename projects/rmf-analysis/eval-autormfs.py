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
from source.utils import rmf, cor_dist, save_image, mse, check_for_dir_and_create
from source.routedatabase import Route
from source.analysis import  eval_pair_rmf, d2i_eval
from source.imageproc.imgproc import Pipeline

params = {#'blur': True,
        'shape': (180, 40),
        #'histeq': True, 
        #'edge_range': (180, 200),
        #'gauss_loc_norm': {'sig1':2, 'sig2':20},
        #'vcrop':1.
        }
pipe = Pipeline(**params)


size = None
route_id = 'route1'
fig_save_path = os.path.join(fwd, 'q_metrics', route_id)
check_for_dir_and_create(fig_save_path)
path =  os.path.join(cwd,'datasets', 'new-antworld', 'exp1', route_id)
route = Route(path, route_id=1)

imgs = route.get_imgs()
imgs = pipe.apply(imgs)

#evaluate imge rmfs against themselves
fit_errors = eval_pair_rmf(imgs)
perc = 75
thresh = np.percentile(fit_errors, perc)
iqr = np.subtract(*np.percentile(fit_errors, [75, 25]))
thresh = thresh  + 1.5 * iqr


fig = plt.figure(figsize=size)
plt.scatter(range(len(fit_errors)), fit_errors)
plt.axhline(y=thresh, color='r', linestyle='--', label='threshold')
plt.xlabel('route index')
plt.ylabel('MSE w.r.t Gaussian curve ')
plt.legend()
# for i, txt in enumerate(fit_errors):
#     plt.annotate(i, (fit_errors[i], fit_errors[i]))
fig.savefig(os.path.join(fig_save_path, 'autormfs-gauss-eval.png'))
plt.show()

idxs_oc = np.argwhere(fit_errors > thresh).flatten()
print(idxs_oc)

for i in idxs_oc:
    imfile = os.path.join(fig_save_path, f'img{i}.png')
    cv.imwrite(imfile, imgs[i])


###########################################################
perc = 75
imgs = route.get_imgs()
imgs =  pipe.apply(imgs)
d2i_evals = d2i_eval(imgs)
thresh = np.percentile(d2i_evals, perc)
iqr = np.subtract(*np.percentile(d2i_evals, [75, 25]))
thresh = thresh  + 1.5 * iqr

#fig_save_path = os.path.join(fig_save_path, 'autormfs-d2i-eval.png')
fig = plt.figure(figsize=size)
plt.scatter(range(len(d2i_evals)), d2i_evals)
plt.axhline(y=thresh, color='r', linestyle='--', label='threshold')
plt.xlabel('route index')
plt.ylabel('depth to integral ratio')
plt.legend()
# for i, txt in enumerate(fit_errors):
#     plt.annotate(i, (fit_errors[i], fit_errors[i]))
fig.savefig(os.path.join(fig_save_path, 'autormfs-d2i-eval.png'))
plt.show()

idxs_oc = np.argwhere(d2i_evals > thresh).flatten()
print(idxs_oc)

# for i in idxs_oc:
#     imfile = os.path.join(fig_save_path, f'img{i}.png')
#     cv.imwrite(imfile, imgs[i])