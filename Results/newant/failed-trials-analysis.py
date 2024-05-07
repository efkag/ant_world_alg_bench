import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from ast import literal_eval
from source.tools.results import filter_results, read_results
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
sns.set_context("paper", font_scale=1)

# general paths
directory = '2024-03-07'
results_path = os.path.join('Results', 'newant',  directory)
fig_save_path = os.path.join(results_path, 'analysis')

data = read_results(os.path.join(results_path, 'results.csv'))
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']





#data = data.loc[data['nav-name'] != 'InfoMax']
# imax_df = data.loc[data['nav-name'] == 'InfoMax']
# data = pd.concat([data, imax_df])

# Plot a specific route
for route_id in range(20):
#route_id = 2
   if route_id+1:
      repeat_no = 0
      save_path = os.path.join(fig_save_path, f"route{route_id}")
      check_for_dir_and_create(save_path)

   filters = {'route_id':route_id, #'num_of_repeat': repeat_no,
            'res':'(180, 40)','blur':True, 'matcher':'mae', 'edge':False,
            }
   df = filter_results(data, **filters)


   #################
   # in case of repeats
   method = list
   #df = df.groupby(['window', 'route_id'])["trial_fail_count"].apply(method).to_frame("trial_fail_count").reset_index()
   ##### if the dataset had nav-names
   df = df.groupby(['nav-name', 'route_id'])["trial_fail_count"].apply(method).to_frame("trial_fail_count").reset_index()
   print(df['nav-name'].unique())
   order = ['A-SMW(15)', 'A-SMW(20)', 'PM', 'SMW(10)', 'SMW(15)', 'SMW(20)', 'SMW(25)', 
            'SMW(30)', 'SMW(40)', 'SMW(50)', 'SMW(75)', 'SMW(100)', 'SMW(150)',
            'SMW(200)', 'SMW(300)', 'SMW(500)']

   df = df.explode("trial_fail_count")

   figsize = (8., 4)
   fig, ax = plt.subplots(figsize=figsize)
   ax.set_title(f'route: {route_id}')
   #ax.set_ylim(0, 20)
   #sns.barplot(x="window", y="trial_fail_count", df=df, ax=ax, estimator=method, capsize=.2, ci=None)
   sns.boxplot(data=df, x="nav-name", y="trial_fail_count",  ax=ax, order=order)

   ax.tick_params(axis='x', labelrotation=90)
   ax.set_xlabel('Navigation Algorithm')
   ax.set_ylabel('Mean TFC')
   plt.tight_layout()
   # path = os.path.join(fig_save_path, f'route[{route_id}]-failed trials.png')
   temp_save_path = os.path.join(save_path, f'swarm-failed-trials{filters}.png')
   fig.savefig(temp_save_path)
   temp_save_path = os.path.join(save_path, 'swarm-failed-trials.pdf')
   fig.savefig(temp_save_path)
   plt.show()


################# joint plot

# fig, axs = plt.subplots(2, 1, figsize=figsize)

# ax = axs[0]
# cols = ['steelblue', 'orange', 'green', 'red', 'purple', 'grey']
# ax.set_title('All Navigation Algorithms')
# sns.barplot(x="nav-name", y="trial_fail_count", data=data, ax=ax, 
#             estimator=method, capsize=.2, ci=None, palette=cols)
# ax.set_xlabel('navigation algorithm')
# ax.set_ylabel('mean TFC')

# ax = axs[1]
# ax.set_title('Temporal Algorithms')
# cols = ['steelblue', 'green', 'red', 'purple', 'grey']
# data = data.drop(data[data['nav-name'] == 'InfoMax'].index)
# sns.barplot(x="nav-name", y="trial_fail_count", data=data, ax=ax, 
#             estimator=method, capsize=.2, ci=None, palette=cols)
# ax.set_xlabel('navigation algorithm')
# ax.set_ylabel('mean TFC')
# plt.tight_layout()
# temp_save_path = os.path.join(fig_save_path, 'failed-trials-joinplot.png')
# fig.savefig(temp_save_path)
# plt.show()
