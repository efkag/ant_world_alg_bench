import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
cwd = os.getcwd()
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import yaml
from source.routedatabase import Route
from source.imageproc.imgproc import Pipeline
from source.utils import cor_dist, mae, check_for_dir_and_create, scale2_0_1, pair_rmf
from source.analysis import flip_gauss_fit, eval_gauss_rmf_fit, d2i_rmfs_eval
sns.set_context("paper", font_scale=1)

route_id = 1
path =  os.path.join(cwd, 'new-antworld', 'curve-bins', f'route{route_id}')
route = Route(path, route_id=route_id)
# pre-proc
combo = {'shape': (180, 80), 'gauss_loc_norm':{'sig1': 2, 'sig2': 20}}
pipe = Pipeline(**combo)
route_imgs = pipe.apply(route.get_imgs())

auto_rmfs_data = pair_rmf(route_imgs, route_imgs, matcher=cor_dist, d_range=(-90, 90))

qmterics = d2i_rmfs_eval(auto_rmfs_data)

q = [0,0.25,0.5,0.75,1]
qtiles = np.quantile(qmterics, q)
print(qtiles)

plt.bar(q, qtiles)
plt.show()

