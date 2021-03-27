from source2 import load_route_naw, pre_process, animated_window
from source2 import seqnav
from source2 import antworld2 as aw

agent = aw.Agent()
route_id = 1
preprocessing = {'shape': (180, 50)}
path = '../new-antworld/exp1/route' + str(route_id) + '/'

# path = '../test_data/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True)

# plot_route(route)
nav = seqnav.SequentialPerfectMemory
# nav = perfect_memory.PerfectMemory
# set up the navigator
route['imgs'] = pre_process(route['imgs'], preprocessing)
nav = nav(route['imgs'], 'mae', window=10, deg_range=(-180,  180))
traj, nav = agent.segment_test(route, nav, segment_length=2, t=20, r=0.1, preproc=preprocessing)

window_log = nav.get_window_log()
path = 'window-plots/'
animated_window(route, window_log, traj=traj, path=path)
