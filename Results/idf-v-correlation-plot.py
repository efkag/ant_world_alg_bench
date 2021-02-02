import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from ast import literal_eval
sns.set_context("paper", font_scale=0.8)
f = lambda x: textwrap.fill(x.get_text(), 15)
# errors,matcher,mean error,pre-proc,seq,tested r


data = pd.read_csv('bench-results-spm.csv')
data_pm = pd.read_csv('bench-results-pm.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data_pm['errors'] = data_pm['errors'].apply(literal_eval)

'''
Plot rmse and correlation blur percentiles
'''
fig, ax = plt.subplots(figsize=(15, 9))
plt.title('rmse v correlation')
blur_data = data[data['pre-proc'].str.contains('blur')]
corr_data = blur_data.loc[(blur_data['matcher'] == 'corr')]
idf_data = blur_data.loc[(blur_data['matcher'] == 'rmse')]
percentile = 85
corr_data = corr_data.groupby(['window'])['errors'].apply(sum).tolist()
idf_data = idf_data.groupby(['window'])['errors'].apply(sum).tolist()
corr_data = [list(map(abs, row)) for row in corr_data]
idf_data = [list(map(abs, row)) for row in idf_data]
idf_percentiles = np.percentile(np.array(idf_data), percentile, axis=1)
corr_percentiles = np.percentile(np.array(corr_data), percentile, axis=1)
x = np.arange(6, 21)
sns.lineplot(y=idf_percentiles, x=x, ax=ax, label='rmse', linewidth=5)
sns.lineplot(y=corr_percentiles, x=x,  ax=ax, label='correlation', linewidth=5)
ax.set_yticks(np.arange(0, 170), 20)
labels = data['window'].unique()
ax.set_ylabel(str(percentile) + 'th percentile error')
ax.set_xlabel('Window size')
fig_save_path = 'Figures/blured rmse vs corr percentiles.eps'
fig.savefig(fig_save_path)
plt.show()


'''
Back to back violin for rmse v corr for all windows
'''
hue = []
x = []
y = []
for i, row in enumerate(idf_data):
    x.extend([i + 6] * len(row))
    hue.extend(['rmse'] * len(row))
    y.extend(row)

for i, row in enumerate(corr_data):
    x.extend([i + 6] * len(row))
    hue.extend(['corr'] * len(row))
    y.extend(row)


fig, ax = plt.subplots(figsize=(4.8, 2.4))
sns.violinplot(x=x, y=y, hue=hue, cut=0, split=True, inner="quart", ax=ax)
plt.ylim([0, 30])
ax.set_xlabel('Window size')
ax.set_ylabel('Error')
ax.set_title("A", loc="left")
plt.tight_layout(pad=0)
fig_save_path = 'Figures/blur-rmse-vs-corr-violins.pdf'
fig.savefig(fig_save_path)
plt.show()



'''
Plot rmse and correlation edges percentiles
'''
fig, ax = plt.subplots(figsize=(15, 9))
plt.title('rmse v correlation')
blur_data = data[data['pre-proc'].str.contains('edge')]
corr_data = blur_data.loc[(blur_data['matcher'] == 'corr')]
idf_data = blur_data.loc[(blur_data['matcher'] == 'rmse')]
percentile = 85
corr_data = corr_data.groupby(['window'])['errors'].apply(sum).tolist()
idf_data = idf_data.groupby(['window'])['errors'].apply(sum).tolist()
corr_data = [list(map(abs, row)) for row in corr_data]
idf_data = [list(map(abs, row)) for row in idf_data]
idf_percentiles = np.percentile(np.array(idf_data), percentile, axis=1)
corr_percentiles = np.percentile(np.array(corr_data), percentile, axis=1)
x = np.arange(6, 21)
sns.lineplot(y=idf_percentiles, x=x, ax=ax, label='rmse', linewidth=5)
sns.lineplot(y=corr_percentiles, x=x,  ax=ax, label='correlation', linewidth=5)
ax.set_yticks(np.arange(0, 170), 20)
labels = data['window'].unique()
ax.set_ylabel(str(percentile) + 'th percentile error')
ax.set_xlabel('Window size')
fig_save_path = 'Figures/edges rmse vs corr percentiles.eps'
fig.savefig(fig_save_path)
plt.show()



'''
SPM secondary results that include matched index diff 
'''
data = pd.read_csv('spm-secondary/bench-results-spm.csv')
data['errors'] = data['errors'].apply(literal_eval)
data['abs index diff'] = data['abs index diff'].apply(literal_eval)

# Get correlation data
edge_data = data[data['pre-proc'].str.contains('edge')]
corr_data = edge_data.loc[(edge_data['matcher'] == 'corr')]
corr_error_data = corr_data.groupby(['window'])['errors'].apply(sum).tolist()
corr_error_data = [list(map(abs, row)) for row in corr_error_data]
corr_idx_data = corr_data.groupby(['window'])['abs index diff'].apply(sum).tolist()

corr_data = []
for i,row in enumerate(corr_idx_data):
    errors = []
    for j, v in enumerate(row):
        if v <= 5:
            errors.append(corr_error_data[i][j])
    corr_data.append(errors)

# Get IDf data
idf_data = edge_data.loc[(edge_data['matcher'] == 'rmse')]
idf_error_data = idf_data.groupby(['window'])['errors'].apply(sum).tolist()
idf_error_data = [list(map(abs, row)) for row in idf_error_data]
idf_idx_data = idf_data.groupby(['window'])['abs index diff'].apply(sum).tolist()

idf_data = []
for i,row in enumerate(idf_idx_data):
    errors = []
    for j, v in enumerate(row):
        if v <= 5:
            errors.append(idf_error_data[i][j])
    idf_data.append(errors)

# Get the percentiles
percentile = 85
corr_percentiles = [np.percentile(np.array(row), percentile) for row in corr_data]
idf_percentiles = [np.percentile(np.array(row), percentile) for row in idf_data]

'''
Plot rmse and correlation edges percentiles
'''
fig, ax = plt.subplots(figsize=(15, 9))
plt.title('rmse v correlation')
x = np.arange(6, 21)
sns.lineplot(y=idf_percentiles, x=x, ax=ax, label='rmse', linewidth=5)
sns.lineplot(y=corr_percentiles, x=x,  ax=ax, label='correlation', linewidth=5)
# ax.set_yticks(np.arange(0, 170), 20)
labels = data['window'].unique()
ax.set_ylabel(str(percentile) + 'th percentile error')
ax.set_xlabel('Window size')
fig_save_path = 'Figures/edges-rmse-vs-corr-percentiles-wrt-index.png'
fig.savefig(fig_save_path)
plt.show()


'''
Back to back violin for rmse v corr for all windows
'''
hue = []
x = []
y = []
for i, row in enumerate(idf_data):
    x.extend([i + 6] * len(row))
    hue.extend(['rmse'] * len(row))
    y.extend(row)

for i, row in enumerate(corr_data):
    x.extend([i + 6] * len(row))
    hue.extend(['corr'] * len(row))
    y.extend(row)


fig, ax = plt.subplots(figsize=(4.8, 2.4))
sns.violinplot(x=x, y=y, hue=hue, cut=0, split=True, inner="quart", ax=ax)
plt.ylim([0, 30])
ax.set_xlabel('Window size')
ax.set_ylabel('Error')
ax.set_title("B", loc="left")
plt.tight_layout(pad=0)
fig_save_path = 'Figures/edges-rmse-vs-corr-violins-wrt-index.pdf'
fig.savefig(fig_save_path)
plt.show()
