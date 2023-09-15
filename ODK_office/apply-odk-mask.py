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
from source.display import plot_ridf_multiline, nans_imgshow

path = 'ODK_office/odk-mask.pickle'
with open(path, 'rb') as handle:
    mask = pickle.load(handle)

route_id = 1
path = 'new-antworld/exp1/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=1, imgs=True)

imgs = route['imgs']
im = cv.resize(imgs[0], (256, 35))
im = im.astype(np.int32)
imm = ma.masked_array(im, mask)

# plt.imshow(rotate(90, imm), cmap='gray')
# plt.show()

# Original image
sims = rmf(im, im, d_range=(-180, 180))
imr = rotate(90, im)
print(np.mean(np.abs(im - imr)))
# Masked image
masked_sims = rmf(imm, imm, d_range=(-180, 180))
print(np.ma.mean(imm - rotate(90, imm)))

# Nan's image
imn = imm.astype(np.float64)
imn = imn.filled(np.nan)
#nans_imgshow(rotate(90, imn))
print(np.nanmean(imn - rotate(90, imn)))
nan_sims = rmf(imn, imn, matcher=nanmae, d_range=(-180, 180))


sims = np.stack([sims, masked_sims, nan_sims], axis=0)
plot_ridf_multiline(sims, labels=['original', 'masked', 'nan-masked'])

imm2 = np.ma.masked_invalid(imn).astype(np.int32)
print(imm.dtype, imm2.dtype)
print(np.array_equal(imm, imm2))
print(np.ma.allequal(imm, imm2))


## Plot images
rows = 3
columns = 1
fig = plt.figure(figsize=(10, 7))
fig.add_subplot(rows, columns, 1)
plt.imshow(imm, cmap='gray')
plt.title('masked')
fig.add_subplot(rows, columns, 2)
plt.imshow(imm2, cmap='gray')
plt.title('masked from nans')
fig.add_subplot(rows, columns, 3)
plt.imshow(imn, cmap='gray')
plt.title('nans')
plt.show()