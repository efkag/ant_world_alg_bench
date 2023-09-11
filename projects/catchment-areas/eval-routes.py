import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from source.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rmse,  rotate, check_for_dir_and_create
from source.routedatabase import Route, BoBRoute, load_bob_routes, load_routes
from catch_areas import catch_areas_4route

# routes_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
# route_ids = [1, 2, 3]
# routes = load_bob_routes(routes_path, route_ids, suffix='N-')
# params = {'blur': True,
#           'vcrop':.6,
#           'shape': (360, 180), 
#         #'edge_range': (180, 200)
#         }


routes_path = '/its/home/sk526/ant_world_alg_bench/new-antworld/curve-bins'
route_ids = [*range(20)]
routes = load_routes(routes_path, route_ids)
params = {'blur': True,
          'shape': (180, 80), 
        #'edge_range': (180, 200)
        }

pipe = Pipeline(**params)

for r in routes:
    catch_areas_4route(r, pipe=pipe, index_step=10, in_translation=True, 
                       error_thresh=30)
