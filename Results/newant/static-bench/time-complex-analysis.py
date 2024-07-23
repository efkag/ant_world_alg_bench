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
from source.analysis import perc_outliers
from source.utils import check_for_dir_and_create
from source.tools.results import filter_results, read_results
sns.set_context("paper", font_scale=1)



# general paths
directory = 'static-bench/time_comp/2024-05-13'
results_path = os.path.join('Results', 'newant',  directory)
fig_save_path = os.path.join(results_path, 'analysis')
check_for_dir_and_create(fig_save_path)

data = read_results(os.path.join(results_path, 'results.csv'))
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']


filters = {'res':'(180, 40)','blur':True, 'matcher':'mae', 'edge':False,
        }
df = filter_results(data, **filters)
df['seconds'] = df['seconds'].apply(literal_eval)


df = df.explode('seconds')
df['seconds'] = pd.to_numeric(df['seconds'])

####### Box plot

figsize = (6, 3)
fig, ax = plt.subplots(figsize=figsize)

ax = sns.boxplot(x="nav-name", y="seconds", data=df, ax=ax)
plt.yscale("log")  
#window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']
#ax.set_xticklabels(window_labels)
ax.set_xlabel('navigation algorithm')
ax.set_ylabel('Runtime (s), log scale')
ax.tick_params(axis='x', labelrotation=90)
plt.tight_layout()
fig.savefig(fig_save_path + '/time-complex-all.png')
fig.savefig(fig_save_path + '/time-complex-all.pdf')
plt.show()
#################


####### SMW vs PM
figsize = (3, 3)
matcher = 'corr'
edge = '(220, 240)'
blur = True
res = '(180, 40)'
nav1 = 'A-SMW(15)'
nav2 = 'PM'
data = data.loc[(data['matcher'] == matcher) & (data['res'] == res) &
                (data['blur'] == blur)]

df = data.groupby(['nav-name'])['seconds'].apply(list).to_frame('seconds').reset_index()
y = df.loc[df['nav-name'] == nav1]['seconds'].to_numpy()[0]
x = df.loc[df['nav-name'] == nav2]['seconds'].to_numpy()[0]
fig, ax = plt.subplots(figsize=figsize)
ax = sns.scatterplot(x=x, y=y, ax=ax, s=100)
ax.set(xlabel=f'{nav2} (s)', ylabel=f'{nav1} (s)')
plt.tight_layout()
fig.savefig(fig_save_path + '/({}, {}).m{}.res{}.b{}.png'.format(nav1, nav2, matcher, res, blur))
fig.savefig(fig_save_path + '/({}, {}).m{}.res{}.b{}.pdf'.format(nav1, nav2, matcher, res, blur))

plt.show()
###################