import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import yaml
from source.utils import cor_dist, mae, check_for_dir_and_create
from source.routedatabase import Route
from source import antworld2 as aw
from source.seqnav import SequentialPerfectMemory
from source.imgproc import Pipeline
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

combo = {'shape': (180, 80), 'gauss_loc_norm':{'sig1': 2, 'sig2': 20}}

# read in route
route_path = os.path.join(routes_path, f"route{route_id}")
route = Route(route_path, route_id=route_id)
#apply pre-proc
pipe = Pipeline(**combo)

# filter data
traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) 
                #& (data['edge'] == edge) 
                & (data['window'] == window) 
                & (data['blur'] == blur)
                #& (data['loc_norm'] == loc_norm) 
                & (data['gauss_loc_norm'] == gauss_loc_norm)
                & (data['route_id'] == route_id)
                ]
traj = traj.to_dict(orient='records')[0]
traj['window_log'] = literal_eval(traj['window_log'])


# static test query img sequence 
# set up
# start end index
start_i = 25
end_i = 40
matcher = cor_dist
window_log = traj['window_log'][start_i]
window = -(window_log[1] - window_log[0])
mem_i = traj['matched_index'][start_i]
### get all the nessesary data

# pre-proc imgs
route_imgs = pipe.apply(route.get_imgs())
nav = SequentialPerfectMemory(route_imgs, matcher, deg_range=(-180, 180), window=window)
nav.reset_window(mem_i)
agent = aw.Agent()
headings = []
# loop
for i in range(start_i, end_i):
    txy = (traj['tx'][i], traj['ty'][i])
    th = traj['th'][i]
    q_img = pipe.apply(agent.get_img(txy, th))
    h = nav.get_heading(q_img)


plt.imshow(q_img)
plt.show()

