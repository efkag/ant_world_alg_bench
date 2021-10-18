import sys
import os
path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
# from source.display import nans_imgshow, plot_multiline, plot_3d
from source.analysis import rgb02nan, nanrgb2grey, nanrbg2greyweighted
from source.utils import nanmae, nan_cor_dist, rmf, cor_dist, save_image, rotate
from source.display import plot_multiline
import pickle

data_path = fwd + '/robot_spin/'

filenames = os.listdir(data_path)

import re
def sort_alphanum( l ):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

filenames = sort_alphanum(filenames)
print(filenames)

imgs = []
for imgfile in filenames:
    img = cv.imread(os.path.join(data_path, imgfile))
    imgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

imgs = rgb02nan(imgs)
imgs = nanrgb2grey(imgs)

# get the headings of eahc of the images
sims = rmf(imgs[0], imgs, matcher=nanmae)
headings = np.argmin(sims, axis=1)

# Rotational similarity between the first image and itself
rsims = [] 
for h in headings:
    rim = rotate(h, imgs[0])
    rsims.append(nanmae(imgs[0], rim))
plt.scatter(headings, rsims, label='in-silica')

minsims = np.min(sims, axis=1)
plt.scatter(headings, minsims, label='actual rotation')

plt.legend()
plt.show()


