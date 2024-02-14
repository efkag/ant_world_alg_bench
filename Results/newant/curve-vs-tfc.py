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
from source.tools.results import filter_results, read_results
from ast import literal_eval
import yaml


directory = '2024-01-22/combined'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
check_for_dir_and_create(fig_save_path)
data = read_results(os.path.join(results_path, 'results.csv'))
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']

#data.drop(data[data['nav-name'] == 'InfoMax'].index, inplace=True)

#metric =  'mean_error'
# metric = 'errors'
metric =  'trial_fail_count'
method = sum
figsize= (10, 5)

# data = data.groupby('window')[metric].apply(sum).to_frame(metric).reset_index()
# sns.boxplot(data=data, x='window', y=metric)
# plt.show()



window = 20
#data = data.loc[data['window'] == window]
data = data.groupby(['route_id', 'nav-name'])[metric].apply(method).to_frame(metric).reset_index()

# Get the curvatures here
curvatures = {}

routes = load_routes(routes_path, data['route_id'].unique(), read_imgs=False)
for route in routes:
    route_dict = route.get_route_dict()
    k = round(meancurv2d(route_dict['x'], route_dict['y']), 4)
    curvatures[int(route.get_route_id())] = k


data['curvature'] = pd.Series(dtype='float')

for key in curvatures:    
    data.loc[data['route_id'] == key,'curvature'] = curvatures[key]


#data['curvature'] = np.round((curvatures), decimals=4)
# sns.scatterplot(data=data, x='route_id', y=metric)

data = data.explode(metric)
#import pdb; pdb.set_trace()

# x = data['route_id'].to_numpy(dtype=np.float)
# y = data[metric].to_numpy(dtype=np.float)
# sns.violinplot(x=x, y=y)
#plt.show()


## boxplot
fig, ax = plt.subplots(figsize=figsize)
ax.set_title(f'curv-perf-w{window}')
sns.barplot(data=data, x="curvature", y=metric, hue='nav-name', ax=ax)
#sns.boxplot(data=data, x="curvature", y=metric, ax=ax)
#ax.set_ylim(0, 60)
ax.tick_params(axis='x', labelrotation=45)
plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, f'curv-perf-met:{metric}'))
plt.show()