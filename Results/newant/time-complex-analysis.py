import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.analysis import perc_outliers
sns.set_context("paper", font_scale=1)


fig_save_path = '/home/efkag/Desktop/'
data = pd.read_csv('combined-results2.csv')
# data = pd.read_csv('combined-results-cont.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['seconds'] = data['seconds'].astype(float)

figsize = (5, 5)

####### Box plot
fig, ax = plt.subplots(figsize=figsize)
ax = sns.boxplot(x="window", y="seconds", data=data, ax=ax)
window_labels = ['Adaptive (20)', 'PM', 'w=15', 'w=20', 'w=25', 'w=30']
ax.set_xticklabels(window_labels)
ax.set_xlabel('Window size')
ax.set_ylabel('Runtime (s)')
plt.tight_layout(pad=0)
plt.show()
#################


####### SMW vs PM
matcher = 'mae'
edge = '(220, 240)'
blur = True
res = '(180, 50)'
data = data.loc[(data['matcher'] == matcher) & (data['res'] == res) &
                (data['blur'] == blur)]

df = data.groupby(['window'])['seconds'].apply(list).to_frame('seconds').reset_index()
y = df.loc[df['window'] == 0]['seconds'].to_numpy()[0]
x = df.loc[df['window'] == -20]['seconds'].to_numpy()[0]
fig, ax = plt.subplots(figsize=figsize)
ax = sns.scatterplot(x=x, y=y, ax=ax, s=100)
ax.set(ylabel='PM (s)', xlabel='ASMW (s)')
plt.tight_layout(pad=0)
plt.show()
