import sys
import os
path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from source.display import nans_imgshow, plot_ridf_multiline, plot_3d
from source.analysis import rgb02nan, nanrgb2grey, nanrbg2greyweighted
from source.utils import nanmae, nan_cor_dist, rmf, cor_dist, save_image, rotate
import pickle

df = pd.read_csv(fwd + '/office/training.csv')
testdf = pd.read_csv(fwd + '/office/testing.csv')

route = df.to_dict('list')

test = testdf.to_dict('list')
print(test.keys())
snaps = []
for imgfile in route[' Filename']:
    imgfile = imgfile.replace(" ", "")
    img = cv.imread(fwd + '/' + imgfile)
    snaps.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

testimgs = []
for imgfile in test[' Filename']:
    imgfile = imgfile.replace(" ", "")
    img = cv.imread(fwd + '/office/' + imgfile)
    testimgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

save_image('original.png', snaps[0])
# plt.imshow(snaps[0])
# plt.show()

# set mask pixels to nan
snaps = rgb02nan(snaps)
testimgs = rgb02nan(testimgs)

# a = snaps[0]
# nans_imgshow(a)
# convert to greyscale
greysnaps = nanrgb2grey(snaps)
testimgs = nanrgb2grey(testimgs)

# a = greysnaps[0]
# nans_imgshow(a)

# alternative way to convert to greyscale
greysnaps2 = nanrbg2greyweighted(snaps)

# a = greysnaps2[0]
# nans_imgshow(a)

sims = rmf(testimgs[35], greysnaps[127], matcher=nanmae, d_range=(-180, 180))
save_image('test.png', testimgs[35])
save_image('train.png', greysnaps[127])


index = np.argmin(sims)
deg_range = (-180, 180)
degrees = np.arange(*deg_range)
h = int(degrees[index])
test_rotated = rotate(h, testimgs[35])
save_image('test rotated.png', test_rotated)

plot_ridf_multiline(sims, xlabel='Degrees', ylabel='Image Difference')
# plot_3d(sims, show=True)

heat = np.abs(testimgs[35] - greysnaps[127])
plt.imshow(heat, cmap='hot', interpolation='nearest')
plt.show()


deg_range = (-90, 90)
degrees = np.arange(*deg_range)
mindiff = []
best_index = []
heading = []
data = {'best_index': [], 'mindiff': [], 'heading': [], 'rsims': []}
for img in testimgs:
    rsims = rmf(img, greysnaps, matcher=nan_cor_dist, d_range=deg_range)

    indices = np.unravel_index(np.argmin(rsims, axis=None), rsims.shape)
    best_index.append(indices[0])
    mindiff.append(rsims[indices])
    heading.append(degrees[indices[1]])
    data['rsims'].append(rsims)

data['best_index'] = best_index
data['mindiff'] = mindiff
data['heading'] = heading

with open('odk_analysis_data_cc.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

