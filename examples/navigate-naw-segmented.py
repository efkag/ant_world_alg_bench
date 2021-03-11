from source.utils import load_route_naw, plot_route, angular_error, pre_process
from source import seqnav
from source import perfect_memory
from source import antworld2 as aw

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
traj, index_log = agent.segment_test(route, nav, segment_length=2, t=20, r=0.1, preproc=preprocessing)
print(traj)
