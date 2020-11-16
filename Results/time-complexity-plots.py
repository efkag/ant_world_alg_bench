import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=1.5)
def mean(l):
    return sum(l)/len(l)

data = pd.read_csv('time_complexities.csv', index_col=False)
# Extract the data only
spm = data['spm']
pm = data['pm']

# fig = plt.subplots(figsize=(2.4, 2.4))
fig = plt.subplots(figsize=(7, 7))
ax = sns.scatterplot(x=data['spm'], y=data['pm'], s=100)
ax.set(ylabel='PM (seconds)', xlabel='SMW (seconds)')
# ax.set_title("A", loc="left")
# plt.tight_layout(pad=0)
ax.figure.savefig('Figures/time-complex-scatter.png')
plt.show()


data = pd.read_csv('bench-results-spm.csv')
data_pm = pd.read_csv('bench-results-pm.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['seconds'] = data['seconds'].apply(literal_eval)


'''
Mean time v mean error for each window
'''
fig, ax = plt.subplots(figsize=(2.4, 2.4))
# For correlations
time_data = data[(data['matcher'] == 'corr') & (data['pre-proc'].str.contains('edge'))]
error_data = time_data.groupby(['window'])['errors'].apply(sum)
error_data = error_data.tolist()
error_data = [mean(x) for x in error_data]
seconds_data = time_data.groupby(['window'])['seconds'].apply(sum)
seconds_data = seconds_data.tolist()
seconds_data = [mean(x) for x in seconds_data]
sns.lineplot(y=error_data, x=seconds_data, ax=ax, label='CC, blur', linewidth=1)
sns.scatterplot(y=error_data, x=seconds_data, ax=ax)

time_data = data[(data['matcher'] == 'idf') & (data['pre-proc'].str.contains('edge'))]
error_data = time_data.groupby(['window'])['errors'].apply(sum)
error_data = error_data.tolist()
error_data = [mean(x) for x in error_data]
seconds_data = time_data.groupby(['window'])['seconds'].apply(sum)
seconds_data = seconds_data.tolist()
seconds_data = [mean(x) for x in seconds_data]
sns.lineplot(y=error_data, x=seconds_data, ax=ax, label='IDF, blur', linewidth=1)
sns.scatterplot(y=error_data, x=seconds_data, ax=ax)
ax.set_title("B", loc="left")
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_ylabel('mean error')
ax.set_xlabel('mean runtime in seconds')
plt.tight_layout(pad=0)

ax.figure.savefig('Figures/time-complex-wrt-window-blur.pdf')
plt.show()

