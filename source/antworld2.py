import antworld, cv2
import numpy as np
from source.utils import check_for_dir_and_create

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
#
# pitch = 0
# roll = 0
# x=0
# y=0
#
# deg = [0, 90, 180, 270, 360]
#
# for i, yaw in enumerate(deg):
#     agent.set_position(x, y, z)
#     agent.set_attitude(yaw, pitch, roll)
#     im = agent.read_frame()
#     filename = "test_data/antworld%i.png" % i
#     cv2.imwrite(filename, im)
#
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



def record_route(datapoints, path):
    check_for_dir_and_create(path)
    x = datapoints[0]
    y = datapoints[1]
    z = datapoints[2]
    headings = datapoints[3]
    z = 1.5

    for i, (xi, yi, h1) in enumerate(zip(x, y, headings)):
        agent.set_position(xi, yi, z)
        agent.set_attitude(h1, 0, 0)
        img = agent.read_frame()
        filename = path + "img%i.png" % i
        cv2.imwrite(filename, img)


def rec_route_from_points(path, target_path):
    datapoints = np.genfromtxt(path, delimiter=',')
    check_for_dir_and_create(target_path)
    np.savetxt(target_path + "route3.csv", datapoints, delimiter=',')
    record_route(datapoints, target_path)


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


def test_nav(path, nav, matcher='mae'):
    # initial position and heading
    xy = (0, 0)
    h = 0
    # step size for the agent movement (in cm)
    r = 0.2
    # timesteps to run the simulations
    t = 10

    data = np.genfromtxt(path + 'route3.csv', delimiter=',')
    x = data[0]
    y = data[1]
    z = data[2]
    headings = data[3]

    imgs = []
    for i in range(len(x)):
        img = cv2.imread(path + '/img' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
        imgs.append(img)

    # set up the navigator
    nav = nav(imgs, matcher)
    img = get_img(xy, h)
    headings = np.empty(t)
    traj = np.empty((2, t))
    for i in range(t):
        h = nav.get_heading(img)
        headings[i] = h
        xy, img = update_position(xy, h, r)
        traj[0, i] = xy[0]
        traj[1, i] = xy[1]

    return headings, traj, nav


"""
Testing
"""
# datapoints = np.genfromtxt('../XYZbins/new_route_1.csv', delimiter=',')
#
# record_route(datapoints, "../new-antworld/route2/")
# agent.set_position(0, 0, z)
# agent.set_attitude(0, 0, 0)
#
# xy, img = update_position((0, 0), 45, 0.5)
#
# print(xy, img.shape)

# rec_route_from_points('../XYZbins/new_route_3.csv', '../new-antworld/route3/')