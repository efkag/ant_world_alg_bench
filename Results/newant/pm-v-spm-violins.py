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
data = pd.read_csv('results.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)

'''
Plot for one specific matcher with one specific pre-proc
'''
fig, ax = plt.subplots(figsize=(15, 9))
plt.title('mae and low res (180, 50)')
# plt.title('rmse and low-res blur (90, 25)')
# v_data = data.loc[(data['matcher'] == 'rmse') & (data['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
# v_data_pm = data_pm[(data_pm['matcher'] == 'rmse') & (data_pm['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
v_data = data.loc[(data['matcher'] == 'mae') & (data['window'] > 0)]
v_data_pm = data[(data['matcher'] == 'mae') & (data['window'] == 0)]
v_data = v_data['errors'].tolist()
v_data_pm = v_data_pm['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
v_data.append(v_data_pm[0])
sns.violinplot(data=v_data, cut=0, ax=ax)
labels = reversed(data['window'].unique().tolist())
ax.set_xticklabels(labels)
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
# fig_save_path = 'Figures/correlation and high-res edges.png'
# # fig_save_path = 'Figures/rmse and low-res blur.png'
# fig.savefig(fig_save_path)
plt.show()


'''
The same but for corr
'''
fig, ax = plt.subplots(figsize=(15, 9))
plt.title('corr and low res (180, 50)')
# plt.title('rmse and low-res blur (90, 25)')
# v_data = data.loc[(data['matcher'] == 'rmse') & (data['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
# v_data_pm = data_pm[(data_pm['matcher'] == 'rmse') & (data_pm['pre-proc'] == '{\'shape\': (90, 25), \'blur\': True}')]
v_data = data.loc[(data['matcher'] == 'corr') & (data['window'] > 0)]
v_data_pm = data[(data['matcher'] == 'corr') & (data['window'] == 0)]
v_data = v_data['errors'].tolist()
v_data_pm = v_data_pm['errors'].tolist()
# Here i use index 0 because the tolist() func above returns a single nested list
v_data.append(v_data_pm[0])
sns.violinplot(data=v_data, cut=0, ax=ax)
labels = reversed(data['window'].unique().tolist())
ax.set_xticklabels(labels)
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
# fig_save_path = 'Figures/correlation and high-res edges.png'
# # fig_save_path = 'Figures/rmse and low-res blur.png'
# fig.savefig(fig_save_path)
plt.show()


data = pd.read_csv('wresults.csv')
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
v_data_10 = data.loc[(data['matcher'] == 'corr') & (data['window'] == 10)]
v_data_15 = data[(data['matcher'] == 'corr') & (data['window'] == 15)]