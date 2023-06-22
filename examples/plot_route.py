import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, pol2cart_headings, plot_route, animated_window, check_for_dir_and_create, squash_deg
from source.routedatabase import Route, BoBRoute
import yaml
from matplotlib import pyplot as plt
from source.display import plot_route, plot_ftl_route
sns.set_context("paper", font_scale=1)

## Antworld routes
# path = 'temp_routes'
# route_id = 6
# path = os.path.join(path, f"route{route_id}")

# route = Route(path=path, route_id=route_id, read_imgs=False)
# route = route.get_route_dict()

# plot_route(route)

# Plot FTL route
route_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
route_id = 2
suffix = 'N-'
route_path = os.path.join(route_path, f"route{route_id}")
route_path = os.path.join(route_path, suffix)
route_path = route_path + str(1)


route = BoBRoute(path=route_path, route_id=route_id, read_imgs=False)
route = route.get_route_dict()
print(route['yaw'])
# for some reason the FTL data is shifted by 90 deg
u, v = pol2cart_headings(270 +  route['yaw'] )
plt.scatter(route['x'], route['y'])
plt.quiver(route['x'], route['y'], u, v)
plt.axis('equal')
plt.show()

# plot_ftl_route(route)
