import os
import antworld
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from source.utils import check_for_dir_and_create, write_route, squash_deg, pre_process, travel_dist, pol2cart
from source.gencoords import generate_from_points, generate_grid

# Old Seville data (lower res, but loads faster)
worldpath = antworld.bob_robotics_path + "/resources/antworld/world5000_gray.bin"
# z = 0.01 # m

# New Seville data
worldpath = antworld.bob_robotics_path + "/resources/antworld/seville_vegetation_downsampled.obj"
# print(antworld.bob_robotics_path)
z = 1.5 # m (for some reason the ground is at ~1.5m for this world)


class Agent:
    def __init__(self, pitch_roll_sig=None, repos_thresh=None):
        self.agent = antworld.Agent(720, 150)
        (xlim, ylim, zlim) = self.agent.load_world(worldpath)
        print(xlim, ylim, zlim)
        self.pitch_roll_noise = pitch_roll_sig
        self.z = z
        # holds current xy possition
        self.xy = None
        # hold current heading
        self.h = None
        # if the sigma of the gaussian noise is provided use a noise function for pitch and roll noise
        if pitch_roll_sig:
            self.noise = lambda : np.random.normal(scale=pitch_roll_sig)
        else:
            self.noise = lambda : 0
        self.repos_thresh = repos_thresh
        self.route = None
        self.trial_fail_count = None

    def record_route(self, route, path, route_id=1):
        check_for_dir_and_create(path)
        x = route['x']
        y = route['y']
        z = route['z']
        headings = route['yaw']
        # Fixed high for now
        # TODO: in the future it may be a good idea to adap the coed to use the elevation
        #   and the pitch, roll noise
        route['filename'] = []

        for i, (xi, yi, h1) in enumerate(zip(x, y, headings)):
            self.agent.set_position(xi, yi, z)
            self.agent.set_attitude(h1, 0, 0)
            img = self.agent.read_frame()
            filename = os.path.join(path, "img%i.png" % i )
            # filename = path + "img%i.png" % i
            cv2.imwrite(filename, img)
            route['filename'].append("img%i.png" % i)

        write_route(path, route, route_id=route_id)

    def rec_route_from_points(self, path, route_id=1, generator='circle', **kwargs):
        # Augment the directory containing all route
        # to create a new directory w.r.t the new route
        path = path + 'route' + str(route_id) + '/'
        check_for_dir_and_create(path)
        # Generate coordinates and write them to file
        route = generate_from_points(path, generator=generator, **kwargs)
        self.record_route(route, path, route_id=route_id)

    def get_img(self, xy, deg):
        '''
        Render a greyscale image from the antworld given an xy position and heading
        :param xy:
        :param deg:
        :return:
        '''
        self.agent.set_position(xy[0], xy[1], self.z)
        self.agent.set_attitude(deg, self.noise(), self.noise())
        return cv2.cvtColor(self.agent.read_frame(), cv2.COLOR_BGR2GRAY)

    def update_position(self, xy, deg, r):
        rad = deg * (np.pi / 180)
        # x and y are inverted because the antworld had the 0 degree origin north
        # this corresponds to the flipping of x, y when calculating the new Cartesian
        # direction
        y, x = pol2cart(r, rad)

        xx = xy[0] + x
        yy = xy[1] + y

        img = self.get_img((x, yy), deg)

        return (xx, yy), img

    def test_nav(self, coords, nav, r=0.05, t=100, sigma=0.1, **kwargs):
        self.trial_fail_count = 0
        self.repos_thresh = kwargs.get('repos_thresh')
        # random initial position and heading
        # near the first location of the route
        if sigma:
            self.h = np.random.randint(0, 360)
            x = coords['x']
            x = np.random.normal(x, sigma)
            y = coords['y']
            y = np.random.normal(y, sigma)
            self.xy = (x, y)
        else:
            self.xy = (coords['x'], coords['y'])
            self.h = coords['yaw']

        # Place agent to the initial position and render the image
        img = self.get_img(self.xy, self.h)
        img = pre_process(img, kwargs)

        # initialise the log variables
        headings = []
        traj = np.empty((2, t))
        # traj[0, 0] = xy[0]
        # traj[1, 0] = xy[1]
        # Navigation loop
        for i in range(0, t):
            # log the coordinates and attitude
            traj[0, i] = self.xy[0]
            traj[1, i] = self.xy[1]
            headings.append(self.h)
            # get the new heading from teh navigator and format it properly
            self.h = nav.get_heading(img)
            self.h = headings[-1] + self.h
            self.h = squash_deg(self.h)

            # reposition the agent and get the new image
            self.xy, img = self.update_position(self.xy, self.h, r)
            self.check4reposition()
            img = self.get_img(self.xy, self.h)
            img = pre_process(img, kwargs)

        headings = np.array(headings)
        trajectory = {'x': traj[0], 'y': traj[1], 'heading': headings}
        return trajectory, nav
    
    def check4reposition(self):
        dist, xy = self.route.min_dist_from_route(self.xy)
        if dist >= self.repos_thresh:
            self.xy = xy
            self.trial_fail_count += 1

    def segment_test(self, route, nav, segment_length=3, **kwargs):
        trajectories = {'x': [], 'y': [], 'heading': []}
        # get starting indices for each segment
        indices, starting_coords = route.segment_route(segment_length)
        for i, coord in enumerate(starting_coords):
            nav.reset_window(indices[i])
            traj, nav = self.test_nav(coord, nav, **kwargs)
            # Append the segment trajectory to the log
            for k in trajectories:
                trajectories[k] = np.append(trajectories[k], traj[k])

        return trajectories, nav

    def run_agent(self, route, nav, segment_length=None, **kwargs):
        self.trial_fail_count = 0
        self.route = route
        if segment_length:
            return self.segment_test(route, nav, segment_length, **kwargs)
        else:
            coords = self.route.get_starting_coords()
            return self.test_nav(coords, nav, **kwargs)

    def rec_grid(self, steps, path):
        path = path + 'grid' + str(steps) + '/'
        grid = generate_grid(steps)
        self.record_route(grid, path)

    def get_trial_fail_count(self):
        return self.trial_fail_count if self.rec_grid else None


"""
Testing
"""
# agent = Agent()
# # # rec_grid(70, path='../new-antworld/')
# route_id = 3
# path = '../new-antworld/exp1/'
# path = '../test_data/'
# # agent.rec_route_from_points(path, route_id=route_id, generator='line', start=-2, end=2, sigma=0.2, curve_points=100)
# agent.rec_route_from_points(path, route_id=route_id, generator='gauss', mean=(0, 0), sigma=3, curve_points=100)

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