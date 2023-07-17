import sys
import os

# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from source.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rmse, check_for_dir_and_create
from source.display import imgshow, imghist

img_path = os.path.join('projects','FTL_env' , 'test_data', 'background_test2', 'snapshot_64.png')
empty_im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
print('empty img shape ', empty_im.shape)

img_path = os.path.join('projects','FTL_env' , 'test_data', 'plants_forward', 'snapshot_1.png')
pfor_im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
print('plant forwards img shape ', pfor_im.shape)

img_path = os.path.join('projects', 'FTL_env', 'test_data', 'plants_right', 'snapshot_1.png')
pri_im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
print('plant rightwards img shape ', pri_im.shape)


im_list = [empty_im, pfor_im, pri_im]
params = {'blur': True,
          'vcrop': .6,
        'shape': (180, 50), 
        #'edge_range': (180, 200)
        }
pipe = Pipeline(**params)
empty_im, pfor_im, pri_im = pipe.apply(im_list)

# imgshow(empty_im)
# imgshow(pfor_im)
# imgshow(pri_im)

fig, axes = plt.subplots(3, 1)
imghist(empty_im, ax=axes[0])
imghist(pfor_im, ax=axes[1])
imghist(pri_im, ax=axes[2])
plt.show()

# Auto RIDFS for empty and single plant
empty_ridf = rmf(empty_im, empty_im, matcher=mae, d_range=(-180, 180))
front_ridf = rmf(pfor_im, pfor_im, matcher=mae, d_range=(-180, 180))

fig, ax = plt.subplots()
fig.suptitle('auto RIDFs')
ax.plot(empty_ridf, label='empty image')
ax.plot(front_ridf, label='single plant image')
plt.legend()
plt.show()


# Ridf comparisons
deg_range = ((-180, 180))
empty_ridf = rmf(empty_im, pfor_im, matcher=mae, d_range=deg_range)
front_auto_ridf = rmf(pfor_im, pfor_im, matcher=mae, d_range=deg_range)
right_ridf = rmf(pri_im, pfor_im, matcher=mae, d_range=deg_range)
degrees = np.arange(*deg_range)

fig, ax = plt.subplots(figsize=(7,3))
ax.plot(degrees, empty_ridf, label='plant for. vs empty img')
ax.plot(degrees, front_auto_ridf, label='plant for. vs itself')
ax.plot(degrees, right_ridf, label='plant for. vs plant left')
ax.set_xlabel('Degrees')
ax.set_ylabel('MAE')
plt.legend()
plt.tight_layout(pad=0.5)
plt.show()


