import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
from source.utils import check_for_dir_and_create
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)



#fig_save_path = '/home/efkag/Desktop/perf'
fig_save_path = os.path.join(fwd, 'analysis')
check_for_dir_and_create(fig_save_path)
results_path = os.path.join(fwd, 'combined-results2.csv')
data = pd.read_csv(results_path)
# data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)


check_for_dir_and_create(fig_save_path)
matcher = 'corr'
edge = 'False'#'(220, 240)'  # 'False'
blur = True
figsize = (6, 3)
res = '(180, 40)'
route = data.loc[(data['matcher'] == matcher) 
                 & (data['edge'] == edge) 
                 & (data['res'] == res) 
                 & (data['blur'] == blur)]
window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']

'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=figsize)
#plt.title('m{}.res{}.b{}.e{}.png'.format(matcher, res, blur, edge))
# Group then back to dataframe
df = route.groupby(['nav-name'])['errors'].apply(sum).to_frame('errors').reset_index()
df = df.explode('errors')
df['errors'] = pd.to_numeric(df['errors'])
#v_data = df['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=df, x='nav-name', y='errors', cut=0, ax=ax)
# labels = df['window'].tolist()
#ax.set_xticklabels(window_labels)
ax.set_ylabel('AAE')
ax.set_xlabel('navigation algorithm')
plt.tight_layout()
fig_path = os.path.join(fig_save_path, f'm{matcher}.res{res}.b{blur}.e{edge}.png')
fig.savefig(fig_path)
fig_path = os.path.join(fig_save_path, f'm{matcher}.res{res}.b{blur}.e{edge}.pdf')
fig.savefig(fig_path)
plt.show()