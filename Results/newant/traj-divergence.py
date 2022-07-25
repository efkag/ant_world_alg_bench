import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from source.utils import check_for_dir_and_create, divergence_traj
from source.routedatabase import Route
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)

directory = '2022-07-14_mid_update'
data_save_path = os.path.join('Results', 'newant', directory)
data = pd.read_csv(os.path.join(data_save_path, 'results.csv'))
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)

routes = data['route_id']
divergence = []
for i, route_id in enumerate(routes):
    path = 'new-antworld/exp1/route' + str(route_id) + '/'
    route = Route(path, route_id=route_id)
    route = route.get_route_dict()

    traj = data.iloc[i]
    traj = traj.to_dict()
    traj['x'] = traj.pop('tx')
    traj['y'] = traj.pop('ty')

    dists = divergence_traj(route, traj)
    divergence.append(dists.tolist())

data['divergence'] = divergence

save_path = os.path.join(data_save_path, 'results.csv')
data.to_csv(save_path, index=False)