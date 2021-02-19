import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from source.utils import load_route_naw, plot_route, angular_error, squash_deg
from source import seqnav
from source import antworld2 as aw

route_id = 1
path = '../new-antworld/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id)

plot_route(route)

nav = seqnav.SequentialPerfectMemory
headings, xy, nav = aw.test_nav(path, nav, route_id=route_id)

traj = {'x': xy[0], 'y': xy[1], 'heading': headings}
#TODO: headings are also stored in the navigator class


plot_route(route, traj)

errors = angular_error(route, traj)
mean_error = sum(errors)/len(errors)
print('mean error:', mean_error)
