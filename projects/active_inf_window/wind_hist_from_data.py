import sys
import os

import yaml.loader
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
from source.utils import check_for_dir_and_create
from source.tools.results import filter_results, read_results
import seaborn as sns
import yaml
import numpy as np
sns.set_context("paper", font_scale=1)


directory = '2024-08-21'
dataset = 'campus'
results_path = os.path.join('Results', dataset,  directory)
fig_save_path = os.path.join(results_path, 'analysis')
check_for_dir_and_create(fig_save_path)

data = read_results(os.path.join(results_path, 'results.csv'))
#routes_path = 'data/outdoors/clean/stanmer/route1'
routes_path = f'data/{dataset}'
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp, Loader=yaml.FullLoader)
sample_step = params['sample_step'][0]


ref_id = 1
rep_id = 2
ylimits = (0, 300)
nav_name = 'SMW(500)'
#{'sig1': 2, 'sig2': 20}
filters = {'nav-name': nav_name, 
           'res':'(180, 45)','blur':True,
           #'vcrop':0,
           'mask':False,
           'gauss_loc_norm':"{'sig1': 2, 'sig2': 20}", 
           #'edge':'False',
           'matcher':'mae',
           'ref_route': ref_id,
           'rep_id': rep_id
           }
fig_save_path = os.path.join(fig_save_path)
check_for_dir_and_create(fig_save_path)
traj = filter_results(data, **filters)

traj = traj.to_dict(orient='records')[0]




midxs = traj['matched_index']
hist = np.ones(traj['window'])
w_log = traj['window_log']

plt.ion()
rects = plt.bar(range(traj['window']), hist)


for i, (mi, w_log) in enumerate(zip(midxs, w_log)):
    print(w_log, mi)
    hist[(mi-w_log[0])] += 1
    plt.cla()
    plt.title(f'test index: {i}')
    pw = hist/hist.sum()
    plt.bar(range(traj['window']), hist)
    plt.pause(0.005)
    
