import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('time_complexities.csv', index_col=False).values
# Extract the data only
data = data[:, 1:]
fig = plt.subplots(figsize=(20, 20))
ax = sns.barplot(data=data)
plt.ylabel('Seconds')
plt.xlabel('Algorithms')
ax.set(xticklabels=['smp', 'pm'])
ax.figure.savefig('time-complex-barplot.png')
plt.show()
