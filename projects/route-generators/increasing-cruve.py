import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
cwd = os.getcwd()
sys.path.append(cwd)

from source import antworld2 as aw
from source.utils import check_for_dir_and_create

path = 'new-antworld/inc-curve2'
agent = aw.Agent()

# sigmas = [.5, 1.5, 2.0, 2.5, 3.0]
# for i, sig in enumerate(sigmas):
#     agent.rec_route_from_points(path, route_id=i, 
#             generator='line', start=-10, end=10, 
#             sigma=sig, curve_points=600)

agent.rec_route_from_points(path, route_id=4, 
        generator='circle', r=10, curve_points=600)

