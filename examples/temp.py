from source2 import load_route_naw, plot_route, angular_error, pre_process
from source2 import seqnav
import numpy as np
import matplotlib.pyplot as plt

route_id = 3
path = '../new-antworld/exp1/route' + str(route_id) + '/'
# path = '../test_data/route'+ str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True, query=True, max_dist=0.2)

plot_route(route)

window = 20
matcher = 'mae'
sets = {'shape': (180, 50)}#, 'edge_range': (180, 200)}
route_imgs = pre_process(route['imgs'], sets)
test_imgs = pre_process(route['qimgs'], sets)

nav = seqnav.SequentialPerfectMemory(route_imgs, matcher, window=window)
recovered_heading, window_log = nav.navigate(test_imgs)

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
traj['heading'] = np.array(traj['heading'])
plot_route(route, traj)

errors, min_dist_index = angular_error(route, traj)
print(np.mean(errors))

xs = route['x'][min_dist_index]
ys = route['y'][min_dist_index]

plt.scatter(xs, ys)
plt.plot(xs, ys)
plt.scatter(traj['x'], traj['y'])
for i, (x, y) in enumerate(zip(xs, ys)):
    plt.annotate(str(i), xy=(x, y))
for i, (x, y) in enumerate(zip(traj['x'], traj['y'])):
    plt.annotate(str(i), xy=(x, y))
plt.show()

