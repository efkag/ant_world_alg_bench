import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())


from source.utils import load_route_naw, plot_route, pre_process, seq_angular_error
from source import seqnav
from source import antworld2 as aw

agent = aw.Agent(pitch_roll_sig=None)
route_id = 2
pre_processing = {'shape': (180, 50), 'edge_range': (180, 220)}
matcher = 'corr'
window = -20
# path = '../new-antworld/exp1/route' + str(route_id) + '/'
path = 'new-antworld/exp1/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True)

# plot_route(route)
route_imgs = pre_process(route['imgs'], pre_processing)
nav = seqnav.SequentialPerfectMemory
# nav = perfect_memory.PerfectMemory
# set up the navigator
nav = nav(route_imgs, matcher, deg_range=(-180, 180), window=window)
coords = {'x':route['x'][0], 'y':route['y'][0], 'yaw':route['yaw'][0]}
traj, nav = agent.test_nav(coords, nav, t=50, r=0.05, sigma=None, preproc=pre_processing)

#TODO: headings are also stored in the navigator class

plot_route(route, traj, scale=80)

errors, _ = seq_angular_error(route, traj)
mean_error = sum(errors)/len(errors)
print('mean error:', mean_error)
