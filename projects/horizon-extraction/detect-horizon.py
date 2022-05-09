import imp
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

path = '/home/efkag/ant_world_alg_bench/projects/horizon-extraction'
path = os.path.join(path, 'img0.png')
img = cv.imread(path, cv.IMREAD_GRAYSCALE)

lower = 240
upper = 250


edge_detector = canny(lower, upper)

edges = edge_detector(img)

# plt.imshow(edges, cmap='gray')
# plt.show()

points = np.where(edges > 0)
points = np.vstack((points[1], points[0]))
line = linregress(points[0], points[1])

slope = line[0]
inter = line[1]
x = np.array([points[0].min(), points[0].max()])
y = slope*x + inter


plt.scatter(points[0], points[1])
plt.plot(x, y, c='r')
plt.show()



