import math
from source.utils import line_incl, pol2cart_headings, check_for_dir_and_create, pol2cart
from matplotlib import pyplot as plt
import numpy as np
pi = math.pi


def meancurv2d(x, y):
    '''
    Calculates the mean curvature of a set of points (x, y) that belong to a curve.
    :param x:
    :param y:
    :return:
    '''
    # first derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)

    # second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # calculate the mean curvature from first and second derivatives
    curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5

    return np.mean(curvature)


def random_circle_points(r, no_of_points):
    '''
    Generates random points within a circle given the desired radius.
    It assumes the center is (0,0)
    :param r:
    :param no_of_points:
    :return:
    '''
    r = r * np.sqrt(np.random.rand(no_of_points))
    theta = np.random.rand(no_of_points) * 2 * pi

    x, y = pol2cart(r, theta)
    return x, y


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
        t = i / float(n - 1)  # type:float
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


def generate(mean, save_path, no_of_points=5, curve_points=100, plot=True, route_id=1):
    '''
    Generates control points froma distribution.
    Uses the control pints to generate points on a bezier curve.

    :param mean:
    :param save_path:
    :param no_of_points:
    :param curve_points:
    :param plot:
    :return:
    '''
    check_for_dir_and_create(save_path)
    mean = np.array(mean)
    cov = np.array([[1, 10], [10, 1]])
    xy = np.random.multivariate_normal(mean, cov, no_of_points)

    np.savetxt(save_path + "points.csv", X=xy, delimiter=',')

    x_route, y_route = bezier_curve_vert(curve_points, xy)
    # 90- rotates the origin of the degrees by 90 degrees counter clokwise
    # setting the origin (0 degrees) at the north
    heading = 90 - line_incl(x_route, y_route)

    z = 1.5
    data = np.array([x_route, y_route, [z] * len(x_route), heading])
    np.savetxt(save_path + 'route' + str(route_id) + '.csv', data, delimiter=',')

    print(meancurv2d(x_route, y_route))
    if plot:
        u, v = pol2cart_headings(90 - heading)
        plt.scatter(x_route, y_route)
        plt.quiver(x_route, y_route, u, v, scale=60)
        plt.scatter(xy[:, 0], xy[:, 1])
        plt.show()


# x, y = random_circle_points(10, 6)
# plt.scatter(x, y)
# plt.show()
# generate([0, 0], "../test_data/route1/", no_of_points=5, route_id=1)


np.random.seed(10)
mean = np.array([0, 0])
cov = np.array([[1, 10], [10, 1]])
xy = np.random.multivariate_normal(mean, cov, 6)

points = [
    [0., 0.],
    [0.5, 0.5],
    [1.0, 1.0],
    [2.5, 2.5],
]
# Show control points
xy = np.array(points)

np.savetxt('../XYZbins/new_points_3.csv', xy, delimiter=',')
plt.plot(xy[:, 0], xy[:, 1])
plt.scatter(xy[:, 0], xy[:, 1])

x_route, y_route = bezier_curve_vert(100, xy)
heading = 90 - line_incl(x_route, y_route)
# heading = line_incl(x_route, y_route)
u, v = pol2cart_headings(90 - heading)
# plt.plot(x_route, y_route)
plt.scatter(x_route, y_route)
plt.quiver(x_route, y_route, u, v, scale=50)
plt.show()

# Save to file
z = 1.5
path = '../XYZbins/new_route_3.csv'
data = np.array([x_route, y_route, [z]*len(x_route), heading])

# data.tofile(path)
np.savetxt(path, data, delimiter=',')

print(meancurv2d(x_route, y_route))

