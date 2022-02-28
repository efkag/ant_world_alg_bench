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
from source.analysis import  eval_pair_rmf

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)

imgs = route.get_imgs()
# evaluate imge rmfs against themselves
fit_errors = eval_pair_rmf(imgs)
plt.scatter(range(len(fit_errors)), fit_errors)
# for i, txt in enumerate(fit_errors):
#     plt.annotate(i, (fit_errors[i], fit_errors[i]))
plt.show()

print(np.argwhere(fit_errors > 6).flatten())