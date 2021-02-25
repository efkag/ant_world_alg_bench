from source.utils import load_route_naw, plot_route, angular_error
from source import seqnav
from source import perfect_memory
from source import antworld2 as aw

route_id = 1
path = '../new-antworld/route' + str(route_id) + '/'
path = '../test_data/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True)

plot_route(route)

nav = seqnav.SequentialPerfectMemory
# nav = perfect_memory.PerfectMemory
# set up the navigator
nav = nav(route['imgs'], 'mae', deg_range=(-180, 180))
traj, nav = aw.test_nav(route, nav, t=20, r=0.1)

#TODO: headings are also stored in the navigator class

plot_route(route, traj)

errors, _ = angular_error(route, traj)
mean_error = sum(errors)/len(errors)
print('mean error:', mean_error)
