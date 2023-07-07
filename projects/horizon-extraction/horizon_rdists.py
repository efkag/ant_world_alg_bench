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

params = {'blur': True,
        'shape': (180, 80)
        #'gauss_loc_norm': {'sig1':2, 'sig2':20}
        }

pipe = Pipeline(**params)
img_small = pipe.apply(img)
plt.imshow(img_small, cmap='gray')
plt.show()


lower = 220
upper = 250

params = {'blur': True,
        'shape': (180, 80), 
        'edge_range': (lower, upper),
        #'gauss_loc_norm': {'sig1':2, 'sig2':20}
        }

pipe = Pipeline(**params)
edges = pipe.apply(img)

# plt.imshow(edges, cmap='gray')
# plt.show()

###### Get the points as x.y coords
y, x = np.where(edges > 0)

##### need to order points by x values
sort_ind = np.argsort(x)
x = np.take(x, sort_ind)
y = np.take(y, sort_ind)


### subtraction the max from y flips the edges points and oriantated them correctly
# thisis because the top left corner of the image is considered the origin (0 ,0)
# and adding the (no of vertical pixels - the max) pushe the pixes up to the corretc place
y = (y.max() - y ) + (80 - y.max())
# the below is without readjusting the edge points.
#y = (y.max() - y )
plt.scatter(x, y, s=10)
plt.imshow(edges, cmap='gray', extent=[0, 180, 0, 80])
plt.show()


### use the diff of the points as a signal
#points = np.vstack((x, y)).astype(np.float64)
#norms = np.linalg.norm(points, axis=0)
def norm2(x, y):
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dists = np.sqrt(dx**2+dy**2)
    return dists


dists = norm2(x, y)
dists =  np.append(dists, dists[-1])
plt.plot(x, dists)
plt.scatter(x, y)
plt.plot(x, y)
plt.show()

def horizon_ext(edges_img):
    y, x = np.where(edges_img > 0)
    ##### need to order points by x values
    sort_ind = np.argsort(x)
    x = np.take(x, sort_ind)
    y = np.take(y, sort_ind)
    # subtraction of the max from y flips the edges points and oriantated them correctly
    # thisis because the top left corner of the image is considered the origin (0 ,0)
    y = y.max() - y
    dists = norm2(x, y)
    return dists
