from source.utils import load_route_naw, plot_route, angular_error
from source import seqnav
from source import perfect_memory
from source import antworld2 as aw

agent = aw.Agent()
route_id = 1
path = '../new-antworld/exp1/route' + str(route_id) + '/'

# path = '../test_data/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True)

# plot_route(route)
route_imgs = route['imgs']
nav = seqnav.SequentialPerfectMemory
# nav = perfect_memory.PerfectMemory
# set up the navigator
traj, index_log = agent.segment_test(route, nav, matcher='mae', window=10, segment_length=2, t=20, r=0.1, preproc={'shape': (180, 50)})
print(traj)
