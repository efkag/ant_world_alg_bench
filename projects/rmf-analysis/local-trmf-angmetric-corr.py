import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
print(fwd)
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import matplotlib.pyplot as plt
from source.utils import rmf, mean_angle, squash_deg, pre_process, weighted_mse
from source.routedatabase import Route
from source.analysis import flip_gauss_fit, gauss_curve

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)
imgs = route.get_imgs()
sets = {'blur': True, 'shape': (180, 50)}
imgs = pre_process(route.get_imgs(), sets)
route_dict = route.get_route_dict()

deg_range = (-90, 90)
degrees = np.arange(*deg_range)
margin = 5
gauss_mse = []
ang_errors = []
for i in range(10, 60):
    ref_h = route_dict['yaw'][i]
    im = imgs[i]
    test_ims = imgs[i-margin:i]
    test_ims.extend(imgs[i+1:i+margin+1])
    rsims = rmf(im, test_ims, d_range=deg_range)

    ref_rsim = rmf(im, im, d_range=deg_range)
    gauss_rmf = flip_gauss_fit(ref_rsim, d_range=deg_range)
    # evaluate 
    gauss_weights = gauss_curve(ref_rsim, d_range=deg_range)
    rsims = list(rsims)
    
    # plt.plot(gauss_rmf)
    # for rsim in rsims:
    #     plt.plot(rsim)
    # plt.show()

    fit_errors = weighted_mse(gauss_rmf, rsims, weights=gauss_weights)
    gauss_mse.extend(fit_errors)
    # Get the heading errors
    # indixes of the minima
    idxs = np.argmin(rsims, axis=1)
    h = []
    # get the headings for each image
    for j in idxs:
        new_h = ref_h + degrees[j]
        new_h = squash_deg(new_h)
        h.append(new_h)
    diff_a = ref_h - h
    er = np.abs((diff_a + 180) % 360 - 180)
    ang_errors.extend(er)


plt.scatter(ang_errors, gauss_mse)
plt.xlabel('angular error')
plt.ylabel('MSE')
plt.show()