import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=0.8)
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
fig, ax = plt.subplots(figsize=(15, 9))
plt.title('correlation and high-res edges (360, 75)')
# plt.title('rmse and low-res blur (90, 25)')
# v_data = data.loc[(data['matcher'] == 'rmse') & (data['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
# v_data_pm = data_pm[(data_pm['matcher'] == 'rmse') & (data_pm['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
v_data = data.loc[(data['matcher'] == 'corr') & (data['pre-proc'] == '{\'edge_range\': (180, 200), \'shape\': (360, 75)}')]
v_data_pm = data_pm[(data_pm['matcher'] == 'corr') & (data_pm['pre-proc'].str.contains('(180, 200)'))]
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
fig_save_path = 'Figures/correlation and high-res edges.png'
# fig_save_path = 'Figures/rmse and low-res blur.png'
fig.savefig(fig_save_path)
plt.show()



'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=(4.8, 2.4))
v_data = data.loc[(data['matcher'] == 'rmse') & (data['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
v_data_pm = data_pm[(data_pm['matcher'] == 'rmse') & (data_pm['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
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
# ax.set_title("A", loc="left")
plt.tight_layout(pad=0)
fig_save_path = 'Figures/rmse and low-res blur.pdf'
fig.savefig(fig_save_path)
plt.show()

'''
Zoomed version of the above
'''
fig, ax = plt.subplots(figsize=(4.8, 2.4))
sns.violinplot(data=v_data, cut=0, ax=ax)
labels = data['window'].unique().tolist()
labels.append('pm')
ax.set_xticklabels(labels)
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
plt.ylim([0, 30])
plt.xlim([5.5, 15.5])
ax.set_title("B", loc="left")
plt.tight_layout(pad=0)
fig_save_path = 'Figures/rmse-and-low-res-blur-zoomed.pdf'
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


# # Plot violin for rmse v corr
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


'''
Window greater than size 11
Plot for one specific matcher with one specific pre-proc
'''
less_than = lambda n: n <= 50.0
fig, ax = plt.subplots(figsize=(15, 9))
plt.title('rmse and low-res blur (90, 25)')
v_data = data[data['window'] >= 12]
labels = v_data['window'].unique().tolist()
v_data = v_data.loc[(v_data['matcher'] == 'rmse') & (v_data['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
v_data_pm = data_pm[(data_pm['matcher'] == 'rmse') & (data_pm['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
v_data = v_data['errors'].tolist()
v_data_pm = v_data_pm['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
v_data.append(v_data_pm[0])
v_data = [list(map(abs, row)) for row in v_data]
v_data = [list(filter(less_than, row)) for row in v_data]
sns.violinplot(data=v_data, cut=0, ax=ax)
labels.append('pm')
ax.set_xticklabels(labels)
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
fig_save_path = 'Figures/window>12-error<50-rmse-and-low-res-blur.png'
fig.savefig(fig_save_path)
plt.show()


# # Plot pre-processing violin plot
# fig_save_path = 'violins-pre-proc.png'
# v_data = data.groupby(['pre-proc'])['errors'].apply(sum).tolist()
# labels = data['pre-proc'].unique()
# fig, ax = plt.subplots(figsize=(20, 20))
# v_data = [list(map(abs, row)) for row in v_data]
# axis = sns.violinplot(data=v_data, cut=0, ax=ax)
# axis.set_ylabel('Degree error')
# axis.set_xlabel('Pre-Processing')
# axis.set_xticks(np.arange(len(labels)))
# axis.set_xticklabels(labels)
# axis.set_xticklabels(map(f, axis.get_xticklabels()))
# fig.savefig(fig_save_path)
# plt.show()
#
#
# '''
# Plot violin for rmse v corr for large window values
# '''
# # Use only window values larger than x
# v_data = data[data['window'] >= 12]
# v_data = v_data.groupby(['matcher'])['errors'].apply(sum).tolist()
# labels = data['matcher'].unique()
# pos = np.arange(len(labels))
# fig, ax = plt.subplots()
# parts = ax.violinplot(v_data, pos, points=100, widths=0.3, showmeans=True,
#                       showextrema=True, showmedians=True, bw_method=0.5)
# parts['cmeans'].set_edgecolor('red')
# ax.set_ylabel('Degree error')
# ax.set_xlabel('Matcher type')
# ax.set_xticks(np.arange(len(labels)))
# ax.set_xticklabels(labels)
# plt.show()
#
#
# '''
# Plot violin for windows for angle error greater than X
# '''
# greater_than = lambda n: n < 20.0
# pos = data['window'].unique()
# pos = np.append(pos, pos[-1] + 1)
# v_data = data.groupby(['window'])['errors'].apply(sum).tolist()
# v_data_pm = data_pm['errors'].sum()
# v_data.append(v_data_pm)
# # Take the absolute values of the angle diffs
# v_data = [list(map(abs, row)) for row in v_data]
# # Filter list for values greater than x using lamda function
# v_data = [list(filter(greater_than, row)) for row in v_data]
# ax = sns.violinplot(data=v_data, cut=True)
# ax.set_ylabel('Degree error')
# ax.set_xlabel('Window size')
# plt.show()
