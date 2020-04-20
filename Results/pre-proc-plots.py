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
Plot performance lineplot for window error for pre-processing with blur.
'''
fig, ax = plt.subplots(figsize=(20, 10))
percentile = 90
plt.title(str(percentile) + 'th percentile of correlation matching errors with blur')
corr_data = data.loc[(data['matcher'] == 'corr')]
# Filter out the blured pre-proc only
corr_data = corr_data[corr_data['pre-proc'].str.contains('blur')]
grouped = corr_data.groupby(['window', 'pre-proc'])['errors'].apply(sum)
corr_data = np.array(grouped.tolist())
percentiles = np.percentile(np.array(corr_data), percentile, axis=1)
percentiles = np.transpose(np.reshape(percentiles, (15, 3)))
x = np.arange(6, 21)
x_labels = data['window'].unique()
sns.lineplot(y=percentiles[1], x=x_labels, ax=ax, label='blur, (360, 75)', linewidth=5)
sns.lineplot(y=percentiles[0], x=x_labels, ax=ax, label='blur, (180, 50) ', linewidth=5)
sns.lineplot(y=percentiles[2], x=x_labels, ax=ax, label='blur, (90, 25)', linewidth=5)
ax.set_ylabel(str(percentile) + ' percentile error')
ax.set_xlabel('window')
fig_save_path = 'Figures/correlation matching errors with blur.png'
fig.savefig(fig_save_path)
plt.show()


'''
Plot performance lineplot for window error for pre-processing with edges.
'''
fig, ax = plt.subplots(figsize=(20, 10))
percentile = 90
plt.title(str(percentile) + 'th percentile of correlation matching errors with edges')
corr_data = data.loc[(data['matcher'] == 'corr')]
# Filter out the blured pre-proc only
corr_data = corr_data[corr_data['pre-proc'].str.contains('edge')]
grouped = corr_data.groupby(['window', 'pre-proc'])['errors'].apply(sum)
corr_data = np.array(grouped.tolist())
percentiles = np.percentile(np.array(corr_data), percentile, axis=1)
percentiles = np.transpose(np.reshape(percentiles, (15, 3)))
x = np.arange(6, 21)
x_labels = data['window'].unique()
sns.lineplot(y=percentiles[0], x=x_labels, ax=ax, label='edges, (360, 75)', linewidth=5)
sns.lineplot(y=percentiles[1], x=x_labels, ax=ax, label='edges, (180, 50) ', linewidth=5)
sns.lineplot(y=percentiles[2], x=x_labels, ax=ax, label='edges, (90, 25)', linewidth=5)
ax.set_ylabel(str(percentile) + ' percentile error')
ax.set_xlabel('window')
fig_save_path = 'Figures/correlation matching errors with edges.png'
fig.savefig(fig_save_path)
plt.show()


# '''
# Plot heatmap for window v pre-processing.
# '''
# fig, ax = plt.subplots(figsize=(20, 10))
# percentile = 95
# plt.title(str(percentile) + 'th percentile of correlation matching errors window vs pre-processing')
# corr_data = data.loc[(data['matcher'] == 'corr')]
# grouped = corr_data.groupby(['window', 'pre-proc'])['errors'].apply(sum)
# h_data = np.array(grouped.tolist())
# h_percentiles = np.percentile(np.array(h_data), percentile, axis=1)
# temp = np.reshape(h_percentiles, (15, 4))
# x_labels = ['edges, (360, 75)', 'blur, (180, 50) ',
#             'blur, (360, 75)', 'blur, (90, 25)']
# y_labels = data['window'].unique()
# sns.heatmap(data=temp, ax=ax, xticklabels=x_labels, yticklabels=y_labels,
#             annot=True, fmt=".2f")
# plt.yticks(rotation=0)
# ax.set_ylabel('window')
# ax.set_xlabel('pre-processing')
# fig_save_path = 'Figures/heatmap window v pre-proc.png'
# fig.savefig(fig_save_path)
# plt.show()