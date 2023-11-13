import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
cwd = os.getcwd()
sys.path.append(os.getcwd())

from source.utils import load_route_naw, plot_route, seq_angular_error, animated_window, pre_process
#from source import seqnav
from source.navs import perfect_memory, seqnav
from source.routedatabase import Route
from source.imgproc import Pipeline
import numpy as np

route_id = 1
path = 'new-antworld/exp1/route' + str(route_id) + '/'
route_path = os.path.join(os.getcwd(), 'test-routes/route1')
# path = '../test_data/route'+ str(route_id) + '/'
#route = load_route_naw(path, route_id=route_id, imgs=True, query=True, max_dist=0.2)
route = Route(path=route_path, route_id=1)
# plot_route(route)

window = -20
matcher = 'corr'
sets = {'blur': True, 'shape': (180, 50)}
pipe = Pipeline(**sets)
imgs = pipe.apply(route.get_imgs())
route_imgs = imgs[::2]
test_imgs = imgs[::3]
#test_imgs = pipe.apply(route['qimgs'], sets)

#nav = perfect_memory.PerfectMemory(route_imgs)
#recovered_heading = nav.navigate(test_imgs)
nav = seqnav.SequentialPerfectMemory(route_imgs, matcher=matcher, window=window)
recovered_heading, window_log = nav.navigate(test_imgs)

window_log = nav.get_window_log()

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
traj['heading'] = np.array(traj['heading'])
plot_route(route, traj)

errors, _ = seq_angular_error(route, traj)
print(np.mean(errors))



# diffs = window_log[:, 1] - window_log[:, 0]
# plt.plot(range(len(diffs)), diffs)
# plt.show()

path = 'window-plots/4'
animated_window(route, window_log, path=path)


