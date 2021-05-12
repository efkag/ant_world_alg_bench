from source.utils import load_route_naw, plot_route, seq_angular_error, animated_window, pre_process
from source import seqnav
import numpy as np

route_id = 3
path = '../new-antworld/exp1/route' + str(route_id) + '/'
# path = '../test_data/route'+ str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True, query=True, max_dist=0.2)

plot_route(route)

window = -20
matcher = 'corr'
sets = {'blur': True, 'shape': (180, 50), 'edge_range': (220, 240)}
route_imgs = pre_process(route['imgs'], sets)
test_imgs = pre_process(route['qimgs'], sets)

# nav = perfect_memory.PerfectMemory(route_imgs, matcher)
# recovered_heading = nav.navigate(test_imgs)
nav = seqnav.SequentialPerfectMemory(route_imgs, matcher, window=window)
recovered_heading, window_log = nav.navigate(test_imgs)

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
traj['heading'] = np.array(traj['heading'])
plot_route(route, traj)

errors, _ = seq_angular_error(route, traj)
print(np.mean(errors))

window_log = np.array(window_log)

# diffs = window_log[:, 1] - window_log[:, 0]
# plt.plot(range(len(diffs)), diffs)
# plt.show()

path = 'window-plots/4'
animated_window(route, window_log, path=path)


