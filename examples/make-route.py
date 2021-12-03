import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)


import numpy as np
import pandas as pd
from source import antworld2 as aw

parent = os.path.join(fwd, os.pardir)
path = os.path.join(os.path.abspath(parent),'new-antworld/exp1/route1/route1.csv' )

print(path)
points = pd.read_csv(path)

points['x'] = points['x'] + 1
 
route = points.to_dict('list')

new_path = os.path.abspath(os.path.join(path, os.pardir))
path = os.path.join(new_path, 'route1par')
print(path)
agent = aw.Agent()
agent.record_route(route, path, route_id=1)
