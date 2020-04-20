import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2.5)

data = pd.read_csv('time_complexities.csv', index_col=False)
# Extract the data only
spm = data['spm']
pm = data['pm']


fig = plt.subplots(figsize=(10, 10))
ax = sns.barplot(data=[spm, pm])
plt.ylabel('Seconds')
plt.xlabel('Algorithms')
ax.set(title='A', xticklabels=['spm', 'pm'])
ax.figure.savefig('time-complex-barplot.png')
plt.show()


fig = plt.subplots(figsize=(15, 15))
ax = sns.scatterplot(x=data['spm'], y=data['pm'], s=400)
ax.set(title='B', ylabel='pm (seconds)', xlabel='spm (seconds)')
ax.figure.savefig('Figures/time-complex-scatter.png')
plt.show()
