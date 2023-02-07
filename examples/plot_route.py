import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
from source.routedatabase import Route
import yaml
from source.display import plot_route
sns.set_context("paper", font_scale=1)

path = 'temp_routes'
route_id = 6
path = os.path.join(path, f"route{route_id}")

route = Route(path=path, route_id=route_id, read_imgs=False)
route = route.get_route_dict()

plot_route(route)
