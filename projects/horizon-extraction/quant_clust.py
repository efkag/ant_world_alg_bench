import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
cwd = os.getcwd()
sys.path.append(cwd)
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.stats import linregress
import source
from source.imgproc import canny
from source.imgproc import Pipeline

path = '/home/efkag/ant_world_alg_bench/projects/horizon-extraction'
path = os.path.join(path, 'img0.png')
img = cv.imread(path, cv.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray')
plt.show()
params = {'blur': True,
        'shape': (180, 80)
        #'gauss_loc_norm': {'sig1':2, 'sig2':20}
        }

pipe = Pipeline(**params)
pimg = pipe.apply(img)

# Quantazing
k = 7
pimg = np.floor(pimg/(pimg.max()/k))

plt.imshow(pimg, cmap='gray')
plt.show()