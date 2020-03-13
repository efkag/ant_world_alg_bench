import math
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
pi = math.pi
# -0.158586 * 1000, 10.2428 * 1000, -0.227704 * 1000, 10.0896 * 1000
x_min = -0.158586 * 1000
x_max = 10.2428 * 1000
y_min = -0.227704 * 1000
y_max = 10.0896 * 1000
step = 0.01


def binomial(i, n):
    """Binomial coefficient"""
    return math.factorial(n) / float(math.factorial(i) * math.factorial(n - i))


def bernstein(t, i, n):
    """Bernstein polynom"""
    return binomial(i, n) * (t ** i) * ((1 - t) ** (n - i))


def bezier(t, points):
    """Calculate coordinate of a point in the bezier curve"""
    n = len(points) - 1
    x = y = 0
    for i, pos in enumerate(points):
        bern = bernstein(t, i, n)
        x += pos[0] * bern
        y += pos[1] * bern
    return x, y


def bezier_curve_range(n, points):
    """Range of points in a curve bezier"""
    for i in range(n):
        t = i / float(n - 1)  # type: float
        yield bezier(t, points)


def bezier_curve_vert(n, verts):
    x_cord = []
    y_cord = []
    for i in range(n):
        t = i / float(n - 1)
        x, y = bezier(t, verts)
        x_cord.append(x)
        y_cord.append(y)
    return x_cord, y_cord


points = [
   [0., 0.3],   # P0
   [0.2, 1.],  # P1
   [1., 0.8],  # P2
   [1.8, 0.],  # P3
]

x_cord, y_cord = bezier_curve_vert(100, points)
xy = np.array(points)
plt.scatter(xy[:, 0], xy[:, 1])
plt.plot(x_cord, y_cord)
plt.show()
