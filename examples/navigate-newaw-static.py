from source.utils import load_route_naw, plot_route, angular_error
from source import seqnav


route_id = 1
path = '../new-antworld/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True, query=True, max_dist=0.3)


plot_route(route)
window = 10

nav = seqnav.SequentialPerfectMemory(route['imgs'], 'mae')
recovered_heading, logs, window_log = nav.navigate(route['qimgs'], window)

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
plot_route(route, traj)

print(window_log)
