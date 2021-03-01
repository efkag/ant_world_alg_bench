import antworld
import cv2
import numpy as np
from source.utils import check_for_dir_and_create, write_route, squash_deg
from source.gencoords import generate_from_points, generate_grid

# Old Seville data (lower res, but loads faster)
worldpath = antworld.bob_robotics_path + "/resources/antworld/world5000_gray.bin"
# z = 0.01 # m

# New Seville data
worldpath = antworld.bob_robotics_path + "/resources/antworld/seville_vegetation_downsampled.obj"
print(antworld.bob_robotics_path)
z = 1.5 # m (for some reason the ground is at ~1.5m for this world)

agent = antworld.Agent(720, 150)
(xlim, ylim, zlim) = agent.load_world(worldpath)
print(xlim, ylim, zlim)


def record_route(route, path, route_id=1):
    check_for_dir_and_create(path)
    x = route['x']
    y = route['y']
    z = route['z']
    headings = route['yaw']
    # Fixed high for now
    # TODO: in the future it may be a good idea to adap the coed to use the elevation
    #   and the pitch, roll noise
    z = 1.5
    route['filename'] = []

    for i, (xi, yi, h1) in enumerate(zip(x, y, headings)):
        agent.set_position(xi, yi, z)
        agent.set_attitude(h1, 0, 0)
        img = agent.read_frame()
        filename = path + "img%i.png" % i
        cv2.imwrite(filename, img)
        route['filename'].append("img%i.png" % i)

    write_route(path, route, route_id=route_id)


def rec_route_from_points(path, route_id=1, generator='circle', **kwargs):
    # Augment the directory containing all route
    # to create a new directory w.r.t the new route
    path = path + 'route' + str(route_id) + '/'
    check_for_dir_and_create(path)
    # Generate coordinates and write them to file
    route = generate_from_points(path, generator=generator, **kwargs)
    record_route(route, path, route_id=route_id)


def get_img(xy, deg):
    '''
    Render a greyscale image from the antworld given an xy position and heading
    :param xy:
    :param deg:
    :return:
    '''
    agent.set_position(xy[0], xy[1], z)
    agent.set_attitude(deg, 0, 0)
    return cv2.cvtColor(agent.read_frame(), cv2.COLOR_BGR2GRAY)


def update_position(xy, deg, r):
    rad = deg * (np.pi / 180)

    xx = xy[0] + (r * np.cos(rad))
    yy = xy[1] + (r * np.sin(rad))

    agent.set_position(xx, yy, z)
    agent.set_attitude(deg, 0, 0)

    img = agent.read_frame()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return (xx, yy), img


def test_nav(route, nav, r=0.05, t=100, sigma=0.1):
    # random initial position and heading
    # near the first location of the route
    # h = np.random.randint(0, 360)
    # x = route['x'][0]
    # x = np.random.normal(x, sigma)
    # y = route['y'][0]
    # y = np.random.normal(y, sigma)
    # xy = (x, y)
    xy = (route['x'][0], route['y'][0])
    h = route['yaw'][0]

    # Place agent to the initial position and render the image
    img = get_img(xy, h)

    # initialise the log variables
    headings = []
    headings.append(h)
    traj = np.empty((2, t))
    traj[0, 0] = xy[0]
    traj[1, 0] = xy[1]
    # Navigation loop
    for i in range(1, t):
        h = nav.get_heading(img)
        h = headings[-1] + h
        h = squash_deg(h)
        headings.append(h)
        # get new position and image
        xy, img = update_position(xy, h, r)
        traj[0, i] = xy[0]
        traj[1, i] = xy[1]

    headings = np.array(headings)
    trajectory = {'x': traj[0], 'y': traj[1], 'heading': headings}
    return trajectory, nav


def rec_grid(steps, path):
    path = path + 'grid' + str(steps) + '/'
    grid = generate_grid(steps)
    record_route(grid, path)


def bench(grid, route_ids):
    pass
    # TODO:
"""
Testing
"""

# # # # rec_grid(70, path='../new-antworld/')
# route_id = 3
# path = '../test_data/'
# rec_route_from_points(path, route_id=route_id, generator='line', start=-5, end=5, sigma=0.1)

#
# record_route(datapoints, "../new-antworld/route2/")
# agent.set_position(0, 0, z)
# agent.set_attitude(0, 0, 0)
#
# xy, img = update_position((0, 0), 45, 0.5)
#
# print(xy, img.shape)
#



# pitch = 0.0
# roll = 0.0
# x=0
# y=0
#
# deg = [0, 90, 180, 270, 360]
# # deg = [0, -90, -180, -270, -360]
#
#
# for i, yaw in enumerate(deg):
#     agent.set_position(x, y, z)
#     agent.set_attitude(yaw, pitch, roll)
#     im = agent.read_frame()
#     filename = "test_data/antworld%i.png" % yaw
#     cv2.imwrite(filename, im)

#
# ys = np.arange(0, 2, 0.1)
# yaw = 0
# imgid = 0
# for i in ys:
#     agent.set_position(x, y + i, z)
#     agent.set_attitude(yaw, pitch, roll)
#     im = agent.read_frame()
#     filename = "test_data/ys%i.png" % imgid
#     cv2.imwrite(filename, im)
#     imgid += 1