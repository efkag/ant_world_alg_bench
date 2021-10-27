import sys
import os
path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
from source.utils import nanmae, nan_cor_dist, rmf, rotate, load_route_naw
from source.display import plot_multiline, nans_imgshow

path = 'ODK_office/odk-mask.pickle'
with open(path, 'rb') as handle:
    mask = pickle.load(handle)

route_id = 1
path = 'new-antworld/exp1/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=1, imgs=True)

imgs = route['imgs']
im = cv.resize(imgs[0], (256, 35))
imm = ma.masked_array(im, mask)

# plt.imshow(imm, cmap='gray')
# plt.show()

sims = rmf(im, im, d_range=(-180, 180))
print(np.mean(im - rotate(90, im)))
masked_sims = rmf(imm, imm, d_range=(-180, 180))
print(np.mean(imm - rotate(90, imm)))
imn = imm.astype(np.float64)
imn = imn.filled(np.nan)
#nans_imgshow(imm)
print(np.nanmean(imn - rotate(90, imn)))
nan_sims = rmf(imn, imn, matcher=nanmae, d_range=(-180, 180))

sims = np.stack([sims, masked_sims, nan_sims], axis=0)
plot_multiline(sims, labels=['original', 'masked', 'nan-masked'])
