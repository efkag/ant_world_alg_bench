import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from ast import literal_eval
sns.set(font_scale=2.5)
f = lambda x: textwrap.fill(x.get_text(), 15)
# errors,matcher,mean error,pre-proc,seq,tested routes,window



fig_save_path = 'violins.png'
data = pd.read_csv('bench-results-spm.csv')
data_pm = pd.read_csv('bench-results-pm.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data_pm['errors'] = data_pm['errors'].apply(literal_eval)


'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=(30, 15))
# plt.title('correlation and high-res edges (360, 75)')
plt.title('idf and low-res blur (90, 25)')
v_data = data.loc[(data['matcher'] == 'idf') & (data['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
v_data_pm = data_pm[(data_pm['matcher'] == 'idf') & (data_pm['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
# v_data = data.loc[(data['matcher'] == 'corr') & (data['pre-proc'] == '{\'edge_range\': (180, 200), \'shape\': (360, 75)}')]
v_data = v_data['errors'].tolist()
v_data_pm = v_data_pm['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
v_data.append(v_data_pm[0])
v_data = [list(map(abs, row)) for row in v_data]
sns.violinplot(data=v_data, cut=0, ax=ax)
labels = data['window'].unique().tolist()
labels.append('pm')
ax.set_xticklabels(labels)
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
# fig_save_path = 'Figures/correlation and high-res edges.png'
fig_save_path = 'Figures/idf and low-res blur.png'
fig.savefig(fig_save_path)
plt.show()


'''
Plot idf and correlation percentiles
'''
fig, ax = plt.subplots(figsize=(15, 15))
plt.title('idf v correlation')
corr_data = data.loc[(data['matcher'] == 'corr')]
idf_data = data.loc[(data['matcher'] == 'idf')]
percentile = 85
corr_data = corr_data.groupby(['window'])['errors'].apply(sum).tolist()
idf_data = idf_data.groupby(['window'])['errors'].apply(sum).tolist()
corr_data = [list(map(abs, row)) for row in corr_data]
idf_data = [list(map(abs, row)) for row in idf_data]
idf_percentiles = np.percentile(np.array(idf_data), percentile, axis=1)
corr_percentiles = np.percentile(np.array(corr_data), percentile, axis=1)
x = np.arange(6, 21)
sns.lineplot(y=idf_percentiles, x=x, ax=ax, label='idf', linewidth=5)
sns.lineplot(y=corr_percentiles, x=x,  ax=ax, label='correlation', linewidth=5)
ax.set_yticks(np.arange(0, 160), 20)
labels = data['window'].unique()
ax.set_ylabel(str(percentile) + ' percentile error')
ax.set_xlabel('Window size')
fig_save_path = 'Figures/idf vs corr percentiles.png'
fig.savefig(fig_save_path)
plt.show()



'''
Violin plots for window and pm across all hyper-parameters
'''
fig_save_path = 'Figures/performance-violins.png'
fig, ax = plt.subplots(figsize=(30, 15))
plt.title('Performance across all parameters')
# Plot violin by window size
labels = data['window'].unique()
labels = list(map(str, labels))
labels.append('pm')
v_data = data.groupby(['window'])['errors'].apply(sum).tolist()
v_data_pm = data_pm['errors'].sum()
v_data.append(v_data_pm)
v_data = [list(map(abs, row)) for row in v_data]
axis = sns.violinplot(data=v_data, cut=0, ax=ax)
axis.set_xticklabels(labels)
axis.set_ylabel('Degree error')
axis.set_xlabel('Window size')
#plt.show()


# # Plot violin for idf v corr
# v_data = data.groupby(['matcher'])['errors'].apply(sum).tolist()
# labels = data['matcher'].unique()
# # fig, ax = plt.subplots()
# v_data = [list(map(abs, row)) for row in v_data]
# axis = sns.violinplot(data=v_data, cut=0, ax=ax[1])
# axis.set_ylabel('Degree error')
# axis.set_xlabel('Matcher type')
# axis.set_xticks(np.arange(len(labels)))
# axis.set_xticklabels(labels)
fig.savefig(fig_save_path)
plt.show()


# Plot pre-processing violin plot
fig_save_path = 'violins-pre-proc.png'
v_data = data.groupby(['pre-proc'])['errors'].apply(sum).tolist()
labels = data['pre-proc'].unique()
fig, ax = plt.subplots(figsize=(20, 20))
v_data = [list(map(abs, row)) for row in v_data]
axis = sns.violinplot(data=v_data, cut=0, ax=ax)
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


'''
Plot violin for windows for angle error greater than X
'''
greater_than = lambda n: n < 20.0
pos = data['window'].unique()
pos = np.append(pos, pos[-1] + 1)
v_data = data.groupby(['window'])['errors'].apply(sum).tolist()
v_data_pm = data_pm['errors'].sum()
v_data.append(v_data_pm)
# Take the absolute values of the angle diffs
v_data = [list(map(abs, row)) for row in v_data]
# Filter list for values greater than x using lamda function
v_data = [list(filter(greater_than, row)) for row in v_data]
ax = sns.violinplot(data=v_data)
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
plt.show()
