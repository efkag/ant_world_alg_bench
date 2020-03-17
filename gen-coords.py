import math
from utils import plot_map, load_grid, line_incl, pol_2cart_headings
from matplotlib import pyplot as plt
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
    [972.296, 5041.41],
    [2072.296, 5341.41],
    [4012.296, 5351.41],
    [7512.296, 3051.41],
    [3012.296, 751.41],
    [312.296, 3351.41],
    [2412.296, 5051.41],
    [3412.296, 5351.41],
    [4412.296, 5451.41],
    [5412.296, 4451.41],
    [6412.296, 5451.41],
    [7412.296, 4451.41],
    [8412.296, 5451.41],
]

xy = np.array(points)
plt.plot(xy[:, 0], xy[:, 1])
plt.scatter(xy[:, 0], xy[:, 1])
plt.show()

x_route, y_route = bezier_curve_vert(400, points)
heading = line_incl(x_route, y_route)
u, v = pol_2cart_headings(heading)

x, y, w = load_grid()

plot_map(w, [x_route, y_route], vectors=[u, v], size=(15, 15))
