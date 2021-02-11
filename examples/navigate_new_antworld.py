import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from source.utils import load_route_naw, plot_route
from source import seqnav
from source import antworld2 as aw


path = '../new-antworld/route3/'
route = load_route_naw(path, route_id=3)

plot_route(route)


nav = seqnav.SequentialPerfectMemory
headings, xy, nav2 = aw.test_nav(path, nav)

traj = {'x': xy[0], 'y':xy[1], 'heading':headings}
#TODO: headings are also stored in the navigator class


plt.scatter(traj['x'], traj['y'])
plt.show()

plot_route(route, traj)

