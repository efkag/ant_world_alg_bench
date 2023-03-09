import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from source.analysis import log_error_points
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import yaml
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
from source.routedatabase import Route
from source import antworld2 as aw
sns.set_context("paper", font_scale=1)


directory = '2023-01-20_mid_update'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)
data['matched_index'] = data['matched_index'].apply(literal_eval)


route_id = 5
window = 15
blur =  True
matcher = 'corr'
edge = 'False'# '(180, 200)'
loc_norm = 'False' # {'kernel_shape':(5, 5)}
gauss_loc_norm = "{'sig1': 2, 'sig2': 20}"
res = '(180, 80)'
threshold = 0
figsize = (10, 10)
title = 'D'

traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) 
                #& (data['edge'] == edge) 
                & (data['window'] == window) 
                & (data['blur'] == blur)
                #& (data['loc_norm'] == loc_norm) 
                & (data['gauss_loc_norm'] == gauss_loc_norm)
                & (data['route_id'] == route_id)
                ]
traj = traj.to_dict(orient='records')[0]


# static test query img sequence 
# start index
agent = aw.Agent()
start_i = 25
end_i = 40

txy = (traj['tx'][start_i], traj['ty'][start_i])
th = traj['th'][start_i]
q_img = agent.get_img(txy, th)

plt.imshow(q_img)
plt.show()
