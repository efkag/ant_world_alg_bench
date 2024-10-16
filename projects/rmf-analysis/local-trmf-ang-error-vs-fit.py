import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
print(fwd)
cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import matplotlib.pyplot as plt
from source.utils import rmf, mean_angle, squash_deg, pre_process
from source.routedatabase import Route

path =  os.path.join(cwd, 'new-antworld', 'exp1', 'route1')
route = Route(path, route_id=1)
imgs = route.get_imgs()
sets = {'blur': True, 'shape': (180, 50)}
imgs = pre_process(route.get_imgs(), sets)
route_dict = route.get_route_dict()

ang_err = np.load(os.path.join(fwd, 'ang_err.npy'))
fit_err = np.load(os.path.join(fwd, 'fit_err.npy'))

# plt.scatter(ang_err, fit_err)
# for i, txt in enumerate(fit_err):
#     plt.annotate(i, (ang_err[i], fit_err[i]))
# plt.show()
idx = np.argwhere(fit_err >= 500).flatten()

deg_range = (-180, 180)
degrees = np.arange(*deg_range)
margin = 5
ang_err = []
mse = []
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
    
    mse.extend([fit_err[i]]*len(idxs))
    diff_a = ref_h - h
    er = np.abs((diff_a + 180) % 360 - 180)
    ang_err.extend(er)

# main plot
plt.scatter(ang_err, mse)
plt.xlabel('angular error')
plt.ylabel('MSE')
plt.show()




