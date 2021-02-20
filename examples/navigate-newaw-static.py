from source.utils import load_route_naw, plot_route, angular_error
from source import seqnav
import copy


route_id = 1
path = '../new-antworld/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True, query=True, max_dist=0.3)


plot_route(route)
# #check the query points order
# for i in range(1, len(route['qx'])):
#     r = copy.deepcopy(route)
#     r['qx'] = route['qx'][:i]
#     r['qy'] = route['qy'][:i]
#     plot_route(r)

window = 10

nav = seqnav.SequentialPerfectMemory(route['imgs'], 'mae')
recovered_heading, logs, window_log = nav.navigate(route['qimgs'], window)

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
# plot_route(route, traj)


for i, w in enumerate(window_log):
    plot_route(route, window=w, windex=i, save=True)

print(window_log)

