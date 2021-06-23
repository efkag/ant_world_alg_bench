import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.analysis import perc_outliers
sns.set_context("paper", font_scale=1)


fig_save_path = '/home/efkag/Desktop/ASMW'
data = pd.read_csv('combined-results2.csv')
# data = pd.read_csv('combined-results-cont.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['seconds'] = data['seconds'].astype(float)


####### Box plot
figsize = (5, 3)
fig, ax = plt.subplots(figsize=figsize)
ax = sns.boxplot(x="window", y="seconds", data=data, ax=ax)
window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']
ax.set_xticklabels(window_labels)
ax.set_xlabel('Window size')
ax.set_ylabel('Runtime (s)')
ax.set_title("B", loc="left")
plt.tight_layout(pad=0)
fig.savefig(fig_save_path + '/time-complex-all.png')
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