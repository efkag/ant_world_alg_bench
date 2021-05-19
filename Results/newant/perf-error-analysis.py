import pandas as pd
import matplotlib.pyplot as plt
from source.utils import check_for_dir_and_create
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)



# fig_save_path = '/home/efkag/Desktop/perf'
fig_save_path = '/home/efkag/Desktop/cont/perf'
data = pd.read_csv('combined-results-cont.csv')
# data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)


check_for_dir_and_create(fig_save_path)
matcher = 'corr'
edge = '(220, 240)'  # 'False'
blur = False
figsize = (4, 3)
res = '(90, 25)'
route = data.loc[(data['matcher'] == matcher) & (data['edge'] == edge) &
                 (data['res'] == res) & (data['blur'] == blur)]
window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']

'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=figsize)
plt.title('m{}.res{}.b{}.e{}.png'.format(matcher, res, blur, edge))
# Group then back to dataframe
df = route.groupby(['window'])['errors'].apply(sum).to_frame('errors').reset_index()
v_data = df['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
sns.violinplot(data=v_data, cut=0, ax=ax)
# labels = df['window'].tolist()
ax.set_xticklabels(window_labels)
ax.set_ylabel('Angular error')
ax.set_xlabel('Window size')
plt.tight_layout(pad=0)

fig.savefig(fig_save_path + '/m{}.res{}.b{}.e{}.png'.format(matcher, res, blur, edge))
plt.show()
