import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from ast import literal_eval
sns.set(font_scale=2.5)
f = lambda x: textwrap.fill(x.get_text(), 15)
# errors,matcher,mean error,pre-proc,seq,tested routes,window
# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians')


fig_save_path = 'violins.png'
data = pd.read_csv('bench-results-spm.csv')
data_pm = pd.read_csv('bench-results-pm.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data_pm['errors'] = data_pm['errors'].apply(literal_eval)

pos = data['window'].unique()
pos = np.append(pos, pos[-1] + 1)
v_data = data.groupby(['window'])['errors'].apply(sum).tolist()
v_data_pm = data_pm['errors'].sum()
v_data.append(v_data_pm)
# Take the absolute values of the angle diffs
v_data = [list(map(abs, row)) for row in v_data]
#a = np.array([np.array(row) for row in v_data])
# Filter list for values greater than x using lamda function
greater_than = lambda n: n > 20.0
v_data = [list(filter(greater_than, row)) for row in v_data]
ax = sns.violinplot(data=v_data)
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(30, 15))

# Plot violin by window size
labels = data['window'].unique()
labels = list(map(str, labels))
labels.append('pm')
v_data = data.groupby(['window'])['errors'].apply(sum).tolist()
v_data_pm = data_pm['errors'].sum()
v_data.append(v_data_pm)

v_data = [list(map(abs, row)) for row in v_data]
axis = sns.violinplot(data=v_data, cut=0, ax=ax[0])
axis.set_xticklabels(labels)
axis.set_ylabel('Degree error')
axis.set_xlabel('Window size')
#plt.show()


# Plot violin for idf v corr
v_data = data.groupby(['matcher'])['errors'].apply(sum).tolist()
labels = data['matcher'].unique()
pos = np.arange(len(labels))
# fig, ax = plt.subplots()
v_data = [list(map(abs, row)) for row in v_data]
axis = sns.violinplot(data=v_data, cut=0, ax=ax[1])
axis.set_ylabel('Degree error')
axis.set_xlabel('Matcher type')
axis.set_xticks(np.arange(len(labels)))
axis.set_xticklabels(labels)
fig.savefig(fig_save_path)
plt.show()

# Plot pre-processing violin plot
fig_save_path = 'violins-pre-proc.png'
v_data = data.groupby(['pre-proc'])['errors'].apply(sum).tolist()
labels = data['pre-proc'].unique()
fig, ax = plt.subplots(figsize=(20, 20))
v_data = [list(map(abs, row)) for row in v_data]
axis = sns.violinplot(data=v_data, ax=ax)
axis.set_ylabel('Degree error')
axis.set_xlabel('Pre-Processing')
axis.set_xticks(np.arange(len(labels)))
axis.set_xticklabels(labels)
axis.set_xticklabels(map(f, axis.get_xticklabels()))
fig.savefig(fig_save_path)
plt.show()


# Plot violin for idf v corr for large window values
# Use only window values larger than x
v_data = data[data['window'] > 12]
v_data = v_data.groupby(['matcher'])['errors'].apply(sum).tolist()
labels = data['matcher'].unique()
pos = np.arange(len(labels))
fig, ax = plt.subplots()
parts = ax.violinplot(v_data, pos, points=100, widths=0.3, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)
parts['cmeans'].set_edgecolor('red')
ax.set_ylabel('Degree error')
ax.set_xlabel('Matcher type')
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
plt.show()


# Plot pre-processing violin plot
# Use only window values larger than x
v_data = data[data['window'] > 12]
v_data = v_data.groupby(['pre-proc'])['errors'].apply(sum).tolist()
labels = data['pre-proc'].unique()
pos = np.arange(len(labels))
fig, ax = plt.subplots()
parts = ax.violinplot(v_data, pos, points=100, widths=0.3, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)
parts['cmeans'].set_edgecolor('red')
ax.set_ylabel('Degree error')
ax.set_xlabel('Pre-Processing')
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_xticklabels(map(f, ax.get_xticklabels()))
plt.show()
