import sys
import os
path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)

import numpy as np
import pickle
from source.utils import load_route_naw, plot_route, seq_angular_error, animated_window, pre_process
from source import seqnav
from source import perfect_memory as pm

path = 'ODK_office/odk-mask.pickle'
with open(path, 'rb') as handle:
    mask = pickle.load(handle)

route_id = 1
path = 'new-antworld/exp1/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=1, imgs=True, query=True)

matcher = 'mae'
nav = pm.PerfectMemory(route['imgs'], matcher)
recovered_heading = nav.navigate(route['qimgs'])

traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
traj['heading'] = np.array(traj['heading'])
plot_route(route, traj)
