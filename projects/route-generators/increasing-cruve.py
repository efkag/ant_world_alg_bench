import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
cwd = os.getcwd()
sys.path.append(cwd)

from source import antworld2 as aw
from source.utils import check_for_dir_and_create
from source.gencoords import generate_from_points
from source.utils import meancurv2d
from source.display import plot_route
import numpy as np

path = 'new-antworld/curve-bins'
agent = aw.Agent()

# sigmas = [.5, 1.5, 2.0, 2.5, 3.0]
# for i, sig in enumerate(sigmas):
#     agent.rec_route_from_points(path, route_id=i, 
#             generator='line', start=-10, end=10, 
#             sigma=sig, curve_points=600)

# agent.rec_route_from_points(path, route_id=0, 
#         generator='circle', r=10, curve_points=600)

# agent.rec_route_from_points(path, route_id=2,
#         generator='line', start=-10, end=10, sigma=5, curve_points=600)


#### refined generation
curve_targets = [0.9]
route_id = 15
while curve_targets:
        route_dict = generate_from_points(path, generator='circle', r=10, curve_points=600)
        k = meancurv2d(route_dict['x'], route_dict['y'])
        print(k)
        k = np.round(k, 2)
        if k > curve_targets[0]:
                print(route_id)
                #curve_targets.remove(k)
                temp_path = os.path.join(path, 'route' + str(route_id))
                check_for_dir_and_create(temp_path, remove=True)
                agent.record_route(route_dict, temp_path, route_id=route_id)
                plot_route(route_dict)
                route_id = route_id + 1
                
