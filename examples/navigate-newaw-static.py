from source.utils import load_route_naw, plot_route, angular_error, animated_window, pre_process
from source import seqnav
import numpy as np

route_id = 3
path = '../new-antworld/exp1/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True, query=True, max_dist=0.3)

plot_route(route)

window = 25
matcher = 'corr'
sets = {'shape': (180, 50), 'edge_range': (180, 200)}
route_imgs = pre_process(route['imgs'], sets)
test_imgs = pre_process(route['qimgs'], sets)

nav = seqnav.SequentialPerfectMemory(route_imgs, matcher)
recovered_heading, logs, window_log = nav.navigate(test_imgs, window)

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
traj['heading'] = np.array(traj['heading'])
plot_route(route, traj)

errors, _ = angular_error(route, traj)
np.mean(errors)

path = 'window_plots/'
animated_window(route, window_log, path=path)


