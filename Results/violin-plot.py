import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# errors,matcher,mean error,pre-proc,seq,tested routes,window
# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians')

data = pd.read_csv('bench-results-test.csv')
print(data.columns)
# Convert list of strings to actual list of lists
data['errors'] = pd.eval(data['errors'])

# Plot violin by window size
pos = data['window'].unique()
v_data = data.groupby(['window'])['errors'].apply(sum).tolist()
fig, ax = plt.subplots()
parts = ax.violinplot(v_data, pos, points=100, widths=0.7, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)
parts['cmeans'].set_edgecolor('red')
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
plt.show()


# Plot violin for idf v corr
v_data = data.groupby(['matcher'])['errors'].apply(sum).tolist()
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
v_data = data.groupby(['pre-proc'])['errors'].apply(sum).tolist()
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
plt.show()
