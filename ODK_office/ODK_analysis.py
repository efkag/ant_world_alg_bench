import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from source.display import nans_imgshow, plot_multiline, plot_3d
from source.analysis import rgb02nan, nanrgb2grey, nanrbg2greyweighted
from source.utils import nanmae, rmf
import pickle

df = pd.read_csv('office/training.csv')
testdf = pd.read_csv('office/testing.csv')

route = df.to_dict('list')

test = testdf.to_dict('list')
print(test.keys())
snaps = []
for imgfile in route[' Filename']:
    imgfile = imgfile.replace(" ", "")
    img = cv.imread(imgfile)
    snaps.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

testimgs = []
for imgfile in test[' Filename']:
    imgfile = imgfile.replace(" ", "")
    img = cv.imread('office/' + imgfile)
    testimgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))


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

sims = rmf(testimgs[0], greysnaps, matcher=nanmae, d_range=(-180, 180))

plot_3d(sims, show=False)


#plot_multiline(sims)
deg_range = (-180, 180)
degrees = np.arange(*deg_range)
mindiff = []
best_index = []
heading = []
data = {'best_index': [], 'mindiff': [], 'heading': [], 'rsims': []}
for img in testimgs:
    rsims = rmf(img, greysnaps, matcher=nanmae, d_range=deg_range)

    indices = np.unravel_index(np.argmin(rsims, axis=None), rsims.shape)

    best_index.append(indices[0])
    mindiff.append(rsims[indices])
    heading.append(degrees[indices[1]])
    data['rsims'].append(rsims)

data['best_index'] = best_index
data['mindiff'] = mindiff
data['heading'] = heading

with open('odk_analysis_data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

