import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
print(fwd)
cwd = os.getcwd()
sys.path.append(cwd)


import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
from source.display import plot_multiline, plot_3d
from source.utils import rmf, cor_dist, save_image, mse, pre_process, angular_error
from source.seqnav import SequentialPerfectMemory
from source.perfect_memory import PerfectMemory
from source.routedatabase import Route
from source.analysis import eval_rmf_fit, eval_pair_rmf

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)

matcher = 'mae'
sets = {'blur': True, 'shape': (180, 50)}

route_imgs = pre_process(route.get_imgs(), sets)
# evaluate imge rmfs against themselves
fit_errors = eval_pair_rmf(route_imgs)


test_imgs = pre_process(route.get_imgs(), sets)

nav = PerfectMemory(route_imgs, matcher)
recovered_heading = nav.navigate(test_imgs)

traj = route.get_xycoords()
traj['heading'] =  np.array(recovered_heading)
# plot_route(route, traj)

ang_err, _ = angular_error(route.get_route_dict(), traj)

np.save(os.path.join(fwd, 'ang_err'), ang_err)
np.save(os.path.join(fwd, 'fit_err'), fit_errors)

plt.scatter(ang_err, fit_errors)
plt.show()

