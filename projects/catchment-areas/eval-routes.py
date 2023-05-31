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
from source.routedatabase import Route, BoBRoute, load_bob_routes
from catch_areas import catch_areas_4route

routes_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
route_ids = [1, 2, 3]
routes = load_bob_routes(routes_path, route_ids, suffix='N-')

for r in routes:
    catch_areas_4route(r, index_step=10)
