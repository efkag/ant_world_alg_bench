import os, sys
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from source.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rotate, check_for_dir_and_create, meancurv2d
from source.routedatabase import Route, load_routes
from ast import literal_eval
import yaml


directory = '2023-11-22/combined'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
check_for_dir_and_create(fig_save_path)
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']

#data.drop(data[data['nav-name'] == 'InfoMax'].index, inplace=True)

# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)

#metric =  'mean_error'
# metric = 'errors'
metric =  'trial_fail_count'
method = sum
figsize= (7, 5)

# data = data.groupby('window')[metric].apply(sum).to_frame(metric).reset_index()
# sns.boxplot(data=data, x='window', y=metric)
# plt.show()



window = 0
data = data.loc[data['window'] == window]
data = data.groupby('route_id')[metric].apply(method).to_frame(metric).reset_index()


# Get the curvatures here
curvatures = []

routes = load_routes(routes_path, data['route_id'], read_imgs=False)
for route in routes:
    route_dict = route.get_route_dict()
    k = meancurv2d(route_dict['x'], route_dict['y'])
    curvatures.append(k)

data['curvature'] = np.round((curvatures), decimals=4)
# sns.scatterplot(data=data, x='route_id', y=metric)

data = data.explode(metric)


# x = data['route_id'].to_numpy(dtype=np.float)
# y = data[metric].to_numpy(dtype=np.float)
# sns.violinplot(x=x, y=y)
#plt.show()


## boxplot
fig, ax = plt.subplots(figsize=figsize)
ax.set_title(f'curv-perf-w{window}')
sns.barplot(data=data, x="curvature", y=metric, ax=ax)
ax.set_ylim(0, 60)
ax.tick_params(axis='x', labelrotation=45)
plt.tight_layout(pad=0)
fig.savefig(os.path.join(fig_save_path, f'curv-perf-met:{metric}-w:{window}'))
plt.show()