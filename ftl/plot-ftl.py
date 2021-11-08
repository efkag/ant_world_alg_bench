import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)

import numpy as np
import pandas as pd
# from ast import literal_eval

from source.utils import plot_route


route_id = 2
path = os.path.join(fwd, 'ftl-{}'.format(route_id), 'training.csv')
dt = pd.read_csv(path, index_col=False)

#print(type(route[' X'][0]))
print(dt.columns)
# route[' X'] = route[' X'].apply(literal_eval)

route = dt.to_dict('list')
route['x'] = route.pop(' X')
route['y'] = route.pop(' Y')
route['yaw'] = np.array(route.pop(' Rx'))

plot_route(route)
