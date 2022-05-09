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
from source.imgproc import Pipeline

path = '/home/efkag/ant_world_alg_bench/projects/horizon-extraction'
path = os.path.join(path, 'img0.png')
img = cv.imread(path, cv.IMREAD_GRAYSCALE)

lower = 240
upper = 250


params = {'blur': True,
        'shape': (180, 50), 
        'edge_range': (220, 250),
        #'gauss_loc_norm': {'sig1':2, 'sig2':20}
        }

pipe = Pipeline(**params)
edges = pipe.apply(img)

# plt.imshow(edges, cmap='gray')
# plt.show()

points = np.where(edges > 0)
points = np.vstack((points[1], points[0])).astype(np.float64)
line = linregress(points[0], points[1])

slope = line[0]
inter = line[1]
print(slope, inter)
x = np.array([points[0].min(), points[0].max()])
y = slope*x + points[1].min()


plt.scatter(points[0], points[1])
#plt.plot(x, y, c='r')
plt.show()

def project_on_line(a, b, p):
        ap = p-a
        ab = b-a
        projected = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
        return projected

a_point = np.hstack((x[0], y[0]))
b_point = np.hstack((x[1], y[1]))

projections = np.zeros_like(points)
for i in range(len(points[0])):
        p = points[:, i]
        proj = project_on_line(a_point, b_point, p)
        projections[:, i] = proj

plt.scatter(points[0], points[1])
plt.scatter(projections[0], projections[1])
plt.plot(x, y, c='r')
plt.show()

diffs = np.subtract(points, projections)
l2s = np.linalg.norm(diffs, axis=0)

# add the edge vector magnitude to the projected points
for i,l in enumerate(l2s):
        projections[:, i] = projections[:, i] +  l2s[i]

plt.scatter(points[0], points[1])
plt.scatter(projections[0], projections[1])
plt.plot(x, y, c='r')
plt.show()


