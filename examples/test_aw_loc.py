import sys
import os
import cv2 as cv

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)


import numpy as np
import matplotlib.pyplot as plt
from source import antworld2 as aw
from source.tools.display import plot_route
from source.utils import check_for_dir_and_create
from source.routedatabase import Route


agent = aw.Agent()
# xy = (0, 0)
# h = 0

# io_path = os.path.join(fwd, 'test')
# check_for_dir_and_create(io_path)
# headings = [0, 45, 90, 180]

# for h in headings:
#     img = agent.get_img(xy, h)
#     #cv.imwrite(os.path.join(io_path, f'img{h}.png'), img)

#     route = {'x':[xy[0]], 'y':[xy[1]], 'yaw':np.array([h])}
#     plot_route(route, scale=100, save=False, path=os.path.join(io_path, f'map{h}.png'))
#     plt.show()


# xys = [(-0.5, 0), (-0.5, 0.25), (-0.5, 0.5), (-0.5, 0.75)]
# h = 0

# io_path = os.path.join(fwd, 'test')
# check_for_dir_and_create(io_path)
# headings = [0, 45, 90, 180]

# for i, xy in enumerate(xys):
#     img = agent.get_img(xy, h)
#     cv.imwrite(os.path.join(io_path, f'img{i}.png'), img)

#     route = {'x':[xy[0]], 'y':[xy[1]], 'yaw':np.array([h])}
#     plot_route(route, save=True, path=os.path.join(io_path, f'map{i}.png'))
#     plt.show()

##### use update function
# xy = (-0.5, 0)
# h = 45
# r = 0.25
# io_path = os.path.join(fwd, 'test')
# check_for_dir_and_create(io_path)


# for i in range(4):
#     xy, img = agent.update_position(xy, deg=h, r=r)
#     cv.imwrite(os.path.join(io_path, f'img{i}.png'), img)

#     route = {'x':[xy[0]], 'y':[xy[1]], 'yaw':np.array([h])}
#     plot_route(route, save=True, path=os.path.join(io_path, f'map{i}.png'))
#     plt.show()


########## test with a recorded route
io_path = os.path.join(fwd, 'test')
check_for_dir_and_create(io_path)
datatset_path = 'datasets/new-antworld/curve-bins'
route_id = 0
route_path = os.path.join(datatset_path, f"route{route_id}")
route = Route(path=route_path, route_id=route_id)

xy = route.get_xycoords()
xy = np.vstack((xy['x'], xy['y']))
yaw = route.get_yaw()

for i in range(len(xy[0])):
    img = agent.get_img((xy[0, i], xy[1, i]), yaw[i])
    cv.imwrite(os.path.join(io_path, f'img{i}.png'), img)

    route = {'x':xy[0][:i+1], 'y':xy[1][:i+1], 'yaw':np.array(yaw[:i+1])}
    plot_route(route, save=True, path=os.path.join(io_path, f'map{i}.png'))