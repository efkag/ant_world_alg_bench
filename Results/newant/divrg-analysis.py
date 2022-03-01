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



# fig_save_path = '/home/efkag/Desktop/perf'
fig_save_path = 'Results/newant/2022-02-25'
data = pd.read_csv('Results/newant/2022-02-25/updated.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['divergence'] = data['divergence'].apply(literal_eval)

check_for_dir_and_create(fig_save_path)
matcher = 'corr'
edge = 'False'  # 'False'
blur = True
figsize = (4, 3)
res = '(180, 50)'
route = data.loc[(data['matcher'] == matcher) & (data['edge'] == edge) &
                 (data['res'] == res) & (data['blur'] == blur)]


'''
Plot errors vs window sizes for a combo of parameters
'''
fig, ax = plt.subplots(figsize=figsize)
plt.title('m{}.res{}.b{}.e{}.png'.format(matcher, res, blur, edge))
# Group then back to dataframe
df = route.groupby(['window'])['divergence'].apply(sum).to_frame('divergence').reset_index()
v_data = df['divergence'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=v_data, cut=0, ax=ax)
window_labels = df['window'].unique().tolist()
ax.set_xticklabels(window_labels)
ax.set_ylabel('Divergence')
ax.set_xlabel('Window size')
plt.tight_layout(pad=0)

fig_save_path = os.path.join(fig_save_path, 'divergence-m{}.res{}.b{}.e{}.png'.format(matcher, res, blur, edge))
fig.savefig(fig_save_path)
plt.show()