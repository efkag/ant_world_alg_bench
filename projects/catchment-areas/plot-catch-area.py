import sys
import os

# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

from datetime import date
today = date.today()
string_date = today.strftime("%Y-%m-%d")

import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from source.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rmse, center_ridf, check_for_dir_and_create
from source.routedatabase import Route, BoBRoute
from source.unwraper import Unwraper
from catch_areas import trans_catch_areas, catch_areas
from source.display import imgshow

route_path = 'test-routes/FTLroutes/N-1-01'
route_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes/route1/N-1'

route = BoBRoute(path=route_path, read_imgs=True, unwraper=Unwraper)

imgs = route.get_imgs()

params = {'blur': True,
        'shape': (180, 80), 
        'vcrop':.6,
        #'edge_range': (180, 200)
        }
pipe = Pipeline(**params)
imgs = pipe.apply(imgs)


ridf_field, areas, area_lims = catch_areas(imgs[30], imgs[25:35])

imgshow(imgs[30], path=fwd, save_id='img')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ridf_field[5], label='RIDF')
left_lim = area_lims[5][0]
right_lim = area_lims[5][1]
ax.set_xlabel('angle [degrees]')
ax.set_ylabel('IDF')
plt.tight_layout(pad=0.5)
ax.scatter(range(left_lim, right_lim), ridf_field[5, left_lim:right_lim], label='catchment area points')
fig.savefig(os.path.join(fwd, 'catch.png'))
plt.legend()
plt.show()
