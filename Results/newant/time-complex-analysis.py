import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import seaborn as sns
from ast import literal_eval
from source.analysis import perc_outliers
from source.tools.results import filter_results, read_results
sns.set_context("paper", font_scale=1)


# general paths
directory = '2024-05-13'
results_path = os.path.join('Results', 'newant',  directory)
fig_save_path = os.path.join(results_path, 'analysis')

data = read_results(os.path.join(results_path, 'results.csv'))
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']


filters = {'res':'(180, 40)','blur':True, 'matcher':'mae', 'edge':False,
        }
df = filter_results(data, **filters)

df['seconds'] = df['seconds'].apply(literal_eval)
df = df.explode('seconds')


####### Box plot
figsize = (5, 3)
fig, ax = plt.subplots(figsize=figsize)
ax = sns.boxplot(x="nav-name", y="seconds", data=data, ax=ax)
# window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']
# ax.set_xticklabels(window_labels)
ax.set_xlabel('Navigation Algorithm')
ax.set_ylabel('Runtime (s)')
#ax.set_title("B", loc="left")
plt.tight_layout()
fig.savefig(fig_save_path + f'time-complex-all.png')
plt.show()
#################


####### SMW vs PM
figsize = (3, 3)
matcher = 'mae'
edge = '(220, 240)'
blur = True
res = '(180, 50)'
w1 = 0
w2 = -20
data = data.loc[(data['matcher'] == matcher) & (data['res'] == res) &
                (data['blur'] == blur)]

df = data.groupby(['window'])['seconds'].apply(list).to_frame('seconds').reset_index()
y = df.loc[df['window'] == w1]['seconds'].to_numpy()[0]
x = df.loc[df['window'] == w2]['seconds'].to_numpy()[0]
fig, ax = plt.subplots(figsize=figsize)
ax = sns.scatterplot(x=x, y=y, ax=ax, s=100)
ax.set(ylabel='PM (s)', xlabel='ASMW (s)')
ax.set_title("A", loc="left")
plt.tight_layout(pad=0)
fig.savefig(fig_save_path + '/({}, {}).m{}.res{}.b{}.png'.format(w1, w2, matcher, res, blur))
plt.show()
###################