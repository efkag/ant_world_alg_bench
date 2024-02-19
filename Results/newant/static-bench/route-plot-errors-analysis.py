import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
from source.routedatabase import Route
import yaml
sns.set_context("paper", font_scale=1)

directory = '2023-11-23/2023-11-23_asmw'
results_path = os.path.join('Results', 'newant', 'static-bench',  directory)
fig_save_path = os.path.join('Results', 'newant', 'static-bench', directory, 'analysis')
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# with open(os.path.join(results_path, 'params.yml')) as fp:
#     params = yaml.load(fp)
# routes_path = params['routes_path']
routes_path = '/its/home/sk526/ant_world_alg_bench/datasets/new-antworld/exp1'

#data.drop(data[data['nav-name'] == 'InfoMax'].index, inplace=True)

# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)


# Plot a specific route
route_id = 5
fig_save_path = os.path.join(fig_save_path, f"route{route_id}")
check_for_dir_and_create(fig_save_path)
path = os.path.join(routes_path, f"route{route_id}")
window = -15
matcher = 'mae'
edge = 'False' #(180, 220)' 
res = '(180, 40)'
blur = True
g_loc_norm = 'False'#"{'sig1': 2, 'sig2': 20}"
loc_norm = 'False'
threshold = 0

figsize = (4, 4)
title = None


traj = data.loc[(data['matcher'] == matcher) 
                & (data['res'] == res) 
                & (data['edge'] == edge) 
                & (data['blur'] == blur) 
                & (data['window'] == window) 
                #& (data['gauss_loc_norm'] == g_loc_norm) 
                # & (data['loc_norm'] == loc_norm) 
                & (data['route_id'] == route_id)]

print(traj)
# traj = data.to_dict(orient='records')[0]
if window:
    w_log = literal_eval(traj['window_log'].to_list()[0])

errors = traj['errors'].tolist()
errors = np.array(errors[0])
traj = {'x': np.array(traj['tx'].tolist()[0]),
        'y': np.array(traj['ty'].tolist()[0]),
        'heading': np.array(traj['th'].tolist()[0])}

route = Route(path, route_id=route_id)
route = route.get_route_dict()
if threshold:
    index = np.argwhere(errors >= threshold).ravel()
    traj['x'] = traj['x'][index]
    traj['y'] = traj['y'][index]
    traj['heading'] = traj['heading'][index]

temp_save_path = os.path.join(fig_save_path, f'route{route_id}.w{window}.m{matcher}.res{res}.edge{edge}.glocnorm{g_loc_norm}.thres{threshold}.pdf')

print(temp_save_path)

fig, ax = plt.subplots(figsize=figsize)
plot_route(route, traj, title=title, ax=ax)


######################################################

# route_id = 1
# window = 30
# matcher = 'corr'
# edge = 'False' #(180, 220)' 
# res = '(180, 50)'
# blur = True
# g_loc_norm = 'False'#"{'sig1': 2, 'sig2': 20}"
# loc_norm = 'False'
# threshold = 20
# repeat_no = 0

# figsize = (4, 4)
# title = None


# traj = data.loc[(data['matcher'] == matcher) 
#                 & (data['res'] == res) 
#                 & (data['edge'] == edge) 
#                 & (data['blur'] == blur) 
#                 & (data['window'] == window) 
#                 #& (data['gauss_loc_norm'] == g_loc_norm) 
#                 # & (data['loc_norm'] == loc_norm) 
#                 & (data['route_id'] == route_id)]

# nav_name = traj["nav-name"].tolist()[0]
# errors = traj['errors'].tolist()
# errors = np.array(errors[0])
# traj = {'x': np.array(traj['tx'].tolist()[0]),
#         'y': np.array(traj['ty'].tolist()[0]),
#         'heading': np.array(traj['th'].tolist()[0])}
# if threshold:
#     index = np.argwhere(errors >= threshold).ravel()
#     traj['x'] = traj['x'][index]
#     traj['y'] = traj['y'][index]
#     traj['heading'] = traj['heading'][index]

# ax.scatter(traj['x'], traj['y'], label=f'{nav_name}')
######################################################


plt.legend()
fig.tight_layout()
fig.savefig(fig_save_path)
plt.show()
plt.close(fig)



# if window:
#     temp_path = os.path.join(fig_save_path,'window-plots')
#     animated_window(route, w_log, traj=traj, path=temp_path, size=figsize, title=None)
