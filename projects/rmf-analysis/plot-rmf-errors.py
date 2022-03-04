import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
print(fwd)
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import matplotlib.pyplot as plt
from source.utils import rmf, mean_angle, squash_deg
from source.routedatabase import Route

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)
imgs = route.get_imgs()
route_dict = route.get_route_dict()

ang_err = np.load(os.path.join(fwd, 'ang_err.npy'))
fit_err = np.load(os.path.join(fwd, 'fit_err.npy'))

# plt.scatter(ang_err, fit_err)
# for i, txt in enumerate(fit_err):
#     plt.annotate(i, (ang_err[i], fit_err[i]))
# plt.show()

idx = np.argwhere(fit_err > 6).flatten()

deg_range = (-180, 180)
degrees = np.arange(*deg_range)
margin = 5
ang_err = []
for i in idx:
    ref_h = route_dict['yaw'][i]
    ref_im = imgs[i]
    test_ims = imgs[i-margin:i]
    test_ims.extend(imgs[i+1:i+margin+1])
    rsims = rmf(ref_im, test_ims, d_range=(deg_range))
    # indixes of the minima
    idxs = np.argmin(rsims, axis=1)
    h = []
    # get the headings for each image
    for j in idxs:
        new_h = ref_h + degrees[j]
        new_h = squash_deg(new_h)
        h.append(new_h)
    
    mean_h = mean_angle(h)
    diff_a = ref_h - mean_h
    er = (diff_a + 180) % 360 - 180
    ang_err.append(er)




