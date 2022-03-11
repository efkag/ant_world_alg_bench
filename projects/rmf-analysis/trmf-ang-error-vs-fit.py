import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
print(fwd)
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import matplotlib.pyplot as plt
from source.utils import rmf, mean_angle, squash_deg, pre_process, mse, weighted_mse
from source.routedatabase import Route
from source.analysis import flip_gauss_fit, gauss_curve


fig_save_path = os.path.join(cwd, 'projects', 'rmf-analysis')
size = (6, 4)

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)
imgs = route.get_imgs()
sets = {'blur': True, 'shape': (180, 50)}
imgs = pre_process(route.get_imgs(), sets)
route_dict = route.get_route_dict()

rsims = rmf(imgs[10], imgs[10:30], d_range=(-180, 180))

gauss_rmf = flip_gauss_fit(rsims[0])

# for r in rsims:
#     plt.plot(range(len(r)), r)
#     plt.plot(range(len(gauss_curve)), gauss_curve)
#     plt.show()

gauss_weights = gauss_curve(rsims[0])

rsims = list(rsims)
fit_errors = weighted_mse(gauss_rmf, rsims, weights=gauss_weights)

fig_save_path = os.path.join(fig_save_path, 'trmf-eval.png')
fig = plt.figure(figsize=size)
plt.title('')
plt.scatter(range(len(rsims)), fit_errors)
plt.xlabel('route index')
plt.ylabel('weighted mse w.r.t Gaussian')
fig.savefig(fig_save_path)
plt.show()


