import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
from source.utils import check_for_dir_and_create
import seaborn as sns
import numpy as np
from source.tools.results import filter_results, read_results
sns.set_context("paper", font_scale=1)


directory = 'static-bench/2024-05-10_mid'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join(results_path, 'analysis')
check_for_dir_and_create(fig_save_path)

data = read_results(os.path.join(results_path, 'results.csv'))
# data = pd.read_csv('exp4.csv')


figsize = (10, 5)

filters = {'res':'(180, 40)','blur':True, 'matcher':'mae', 
           #'edge':False,
        }
df = filter_results(data, **filters)

metric = 'errors'
grouping_func = sum
method = np.count_nonzero
aae_threshold = 30 #degrees
dist_threshold = 0.3 #m

def new_perf_metric(r):
    mask = (np.array(r['errors']) >= aae_threshold) | (np.array(r['dist_diff']) >= dist_threshold)
    return method(mask)/mask.size

df['mod_aae'] = df.apply(new_perf_metric, axis=1)

'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=figsize)
#plt.title(matcher + ', route:' + str(route_id))
# Group then back to dataframe
df = df.groupby(['nav-name'])[metric].apply(grouping_func).to_frame(metric).reset_index()
print(df['nav-name'].unique())
order = ['A-SMW(20)', 'PM', 'SMW(10)', 'SMW(15)', 'SMW(20)', 'SMW(25)', 
        'SMW(30)', 'SMW(40)', 'SMW(50)', 'SMW(75)', 'SMW(100)', 'SMW(150)',
        'SMW(200)', 'SMW(300)', 'SMW(500)']
df = df.explode(metric)
df[metric] = pd.to_numeric(df[metric])

sns.violinplot(data=df, x='nav-name', y=metric, order=order, cut=0, ax=ax)

# ax.set_ylim([-1, 180])
ax.set_ylabel('AAE')
ax.set_xlabel('navigation algorithm')
ax.tick_params(axis='x', rotation=90)

plt.tight_layout()
fig_path = os.path.join(fig_save_path, f'{metric}_{filters}.png')
fig.savefig(fig_path)
fig_path = os.path.join(fig_save_path, f'{metric}_{filters}.pdf')
fig.savefig(fig_path)
plt.show()