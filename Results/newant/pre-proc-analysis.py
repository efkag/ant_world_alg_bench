import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)


def to_array(x):
    return np.fromstring(x[1:-1], dtype=np.int, sep=' ').tolist()


fig_save_path = '/home/efkag/Desktop/pre-proc'
# data = pd.read_csv('combined-results.csv')
data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)


matcher = 'corr'
edge = '(220, 240)' # 'False'
figsize = (4, 3)
resolutions = ['(360, 100)', '(180, 50)', '(90, 25)']
data = data.loc[(data['matcher'] == matcher)
                 & (data['edge'] == edge)]

window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']
'''
Plot performance lineplot for window error for different reses.
'''
fig, ax = plt.subplots(figsize=figsize)
percentile = 90
x_labels = np.sort(data['window'].unique())
grouped = data.groupby(['window', 'res'])['errors'].apply(sum).to_frame('errors').reset_index()
x = np.arange(len(x_labels))
for r in resolutions:
    res = grouped.loc[grouped['res'] == r]['errors'].to_list()
    res = np.array(res)
    percentiles = np.percentile(res, percentile, axis=1)
    sns.lineplot(y=percentiles, x=x, ax=ax, label=r, linewidth=2)
    sns.scatterplot(y=percentiles, x=x, ax=ax)

ax.set_ylabel(str(percentile) + 'th percentile angular error')
ax.set_xlabel('window')
# ax.set_xticks(x_labels)
# ax.set_yscale('log')
# ax.set_ylim(10**0.5, 10**2.3)
# ax.set_title("A", loc="left")
plt.tight_layout(pad=0)
ax.set_xticklabels(x_labels)
fig.savefig(fig_save_path + '/lines.m.{}.edge{}.png'.format(matcher, edge))
plt.show()


'''
Box plot for each resolution and window
'''
figsize = (7, 4)
hue = []
x = []
y = []
for r in resolutions:
    res = grouped.loc[grouped['res'] == r]['errors'].to_list()
    for i, row in enumerate(res):
        x.extend([i] * len(row))
        hue.extend([r] * len(row))
        y.extend(row)

fig, ax = plt.subplots(figsize=figsize)
sns.boxplot(x=x, y=y, hue=hue, ax=ax)
plt.tight_layout(pad=0)
ax.set_xticklabels(window_labels)
plt.legend(loc=1)
fig.savefig(fig_save_path + '/box.m{}.edge{}.png'.format(matcher, edge))
plt.show()



