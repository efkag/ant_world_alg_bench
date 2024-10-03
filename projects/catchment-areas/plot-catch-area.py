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
from source.imageproc.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rmse, center_ridf, check_for_dir_and_create
from source.routedatabase import Route, BoBRoute
from source.unwraper import Unwraper
from catch_areas import trans_catch_areas, rot_catch_areas
from source.display import imgshow

route_path = 'test-routes/FTLroutes/N-1-01'
#route_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes/route1/N-1-01'

route = BoBRoute(path=route_path, read_imgs=True, unwraper=Unwraper)

imgs = route.get_imgs()

params = {'blur': True,
        'shape': (180, 80), 
        'vcrop':.6,
        #'edge_range': (180, 200)
        }
pipe = Pipeline(**params)
imgs = pipe.apply(imgs)

ref=30
start = 20
stop = 40
ridf_field, areas, area_lims = rot_catch_areas(imgs[ref], imgs[start:stop], route=route)

ref = 0
query = 10

cv.imwrite(os.path.join(fwd, 'refimg.jpg'), imgs[ref])
cv.imwrite(os.path.join(fwd, 'qimg.jpg'), imgs[query])

imgshow(imgs[ref])
imgshow(imgs[query])

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(range(-180, 180),ridf_field[ref], label='RIDF')
ax.plot(range(-180, 180),np.gradient(ridf_field[query]), label='RIDF gradient')
left_lim = area_lims[5][0]
right_lim = area_lims[5][1]
ax.set_xlabel('angle [degrees]')
ax.set_ylabel('IDF')
plt.tight_layout(pad=0.5)
ax.scatter(range(left_lim-180, right_lim-180), ridf_field[ref, left_lim:right_lim], s=10, label='catchment area points')
#ax.set_xticklabels([*range(-180, 180, 10)])
plt.legend()
fig.savefig(os.path.join(fwd, 'catch.png'))
fig.savefig(os.path.join(fwd, 'catch.pdf'))
plt.show()
