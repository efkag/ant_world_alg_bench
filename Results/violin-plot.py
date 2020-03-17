import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
f = lambda x: textwrap.fill(x.get_text(), 25)
# errors,matcher,mean error,pre-proc,seq,tested routes,window
# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians')


fig_save_path = 'violins.png'
data = pd.read_csv('bench-results-large.csv')
print(data.columns)
# Convert list of strings to actual list of lists
data['errors'] = pd.eval(data['errors'])


fig, ax = plt.subplots(2, 3, figsize=(30, 15))
ax[0][0].axis('off')
ax[0][2].axis('off')
ax[1][1].axis('off')


# Plot violin by window size
pos = data['window'].unique()
v_data = data.groupby(['window'])['errors'].apply(sum).tolist()
parts = ax[0][1].violinplot(v_data, pos, points=100, widths=0.7, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)
parts['cmeans'].set_edgecolor('red')
ax[0][1].set_ylabel('Degree error')
ax[0][1].set_xlabel('Window size')
#plt.show()


# Plot violin for idf v corr
v_data = data.groupby(['matcher'])['errors'].apply(sum).tolist()
labels = data['matcher'].unique()
pos = np.arange(len(labels))
# fig, ax = plt.subplots()
parts = ax[1][0].violinplot(v_data, pos, points=100, widths=0.3, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)
parts['cmeans'].set_edgecolor('red')
ax[1][0].set_ylabel('Degree error')
ax[1][0].set_xlabel('Matcher type')
ax[1][0].set_xticks(np.arange(len(labels)))
ax[1][0].set_xticklabels(labels)
# plt.show()

# Plot pre-processing violin plot
v_data = data.groupby(['pre-proc'])['errors'].apply(sum).tolist()
labels = data['pre-proc'].unique()
pos = np.arange(len(labels))
# fig, ax = plt.subplots()
parts = ax[1][2].violinplot(v_data, pos, points=100, widths=0.3, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)
parts['cmeans'].set_edgecolor('red')
ax[1][2].set_ylabel('Degree error')
ax[1][2].set_xlabel('Pre-Processing')
ax[1][2].set_xticks(np.arange(len(labels)))
ax[1][2].set_xticklabels(labels)
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
