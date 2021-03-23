import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1)


def to_array(x):
    return np.fromstring(x[1:-1], dtype=np.int, sep=' ').tolist()


fig_save_path = '/home/efkag/Desktop/matchers'
# data = pd.read_csv('combined-results.csv')
data = pd.read_csv('exp4.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)


edge = '(220, 240)' # 'False'
res = '(180, 50)'
figsize = (4, 3)
matchers = ['mae', 'corr']


df = data.loc[(data['res'] == res) & (data['edge'] == edge)]
'''
Plot for idf v cc matchers with one specific resolution
'''
fig, ax = plt.subplots(figsize=figsize)
window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']
plt.title(res + edge)
# Group then back to dataframe
grouped = df.groupby(['window', 'matcher'])['errors'].apply(sum).to_frame('errors').reset_index()

hue = []
x = []
y = []
for m in matchers:
    df = grouped.loc[grouped['matcher'] == m]['errors'].to_list()
    for i, row in enumerate(df):
        x.extend([i] * len(row))
        hue.extend([m] * len(row))
        y.extend(row)
sns.violinplot(x=x, y=y, hue=hue, ax=ax, cut=0, split=True, inner="quart")
plt.tight_layout(pad=0)
ax.set_xticklabels(window_labels)
fig.savefig(fig_save_path + '/matchers.res{}.edge{}.png'.format(res, edge))
plt.show()


'''
IDF with blur and CC with edges
'''
cc_df = data.loc[(data['res'] == res) & (data['matcher'] == 'corr') & (data['edge'] == edge)]
cc_df = cc_df.groupby(['window'])['errors'].apply(sum).to_frame('errors').reset_index()
idf_df = data.loc[(data['res'] == res) & (data['matcher'] == 'mae') & (data['edge'] == 'False')]
idf_df = idf_df.groupby(['window'])['errors'].apply(sum).to_frame('errors').reset_index()
fig, ax = plt.subplots(figsize=figsize)
window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']
plt.title('CC with edges and IDF with Blur')
for m, df in zip(matchers, [idf_df, cc_df]):
    df = df['errors'].to_list()
    for i, row in enumerate(df):
        x.extend([i] * len(row))
        hue.extend([m] * len(row))
        y.extend(row)
sns.violinplot(x=x, y=y, hue=hue, ax=ax, cut=0, split=True, inner="quart")
plt.tight_layout(pad=0)
ax.set_xticklabels(window_labels)
fig.savefig(fig_save_path + '/matchers.res{}.edge{}.blur.png'.format(res, edge))
plt.show()
