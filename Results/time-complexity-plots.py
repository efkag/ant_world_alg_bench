import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)

data = pd.read_csv('time_complexities.csv', index_col=False)
# Extract the data only
spm = data['spm']
pm = data['pm']


fig = plt.subplots(figsize=(20, 20))
ax = sns.barplot(data=[spm, pm])
plt.ylabel('Seconds')
plt.xlabel('Algorithms')
ax.set(xticklabels=['smp', 'pm'])
ax.figure.savefig('time-complex-barplot.png')
plt.show()


fig = plt.subplots(figsize=(20, 20))
ax = sns.scatterplot(x=data['spm'], y=data['pm'])
ax = sns.lineplot(x=data['spm'], y=data['pm'])
ax.set(ylabel='pm (seconds)', xlabel='spm (seconds)')
ax.figure.savefig('time-complex-scatter.png')
plt.show()
