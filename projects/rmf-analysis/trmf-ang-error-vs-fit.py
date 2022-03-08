import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
print(fwd)
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import matplotlib.pyplot as plt
from source.utils import rmf, mean_angle, squash_deg, pre_process, mse
from source.routedatabase import Route
from source.analysis import flip_gauss_fit

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)
imgs = route.get_imgs()
sets = {'blur': True, 'shape': (180, 50)}
imgs = pre_process(route.get_imgs(), sets)
route_dict = route.get_route_dict()

rsims = rmf(imgs[10], imgs[10:20], d_range=(-180, 180))

gauss_curve = flip_gauss_fit(rsims[0])

# for r in rsims:
#     plt.plot(range(len(r)), r)
#     plt.plot(range(len(gauss_curve)), gauss_curve)
#     plt.show()

fit_errors = [mse(i, gauss_curve) for i in rsims]

plt.scatter(range(len(rsims)), fit_errors)
plt.xlabel('index')
plt.ylabel('mse')
plt.show()


