import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# errors,matcher,mean error,pre-proc,seq,tested routes,window
# for partname in ('cbars','cmins','cmaxes','cmeans','cmedians')

data = pd.read_csv('bench-results.csv')
# Convert list of strings to actual list of lists
data['errors'] = pd.eval(data['errors'])

# Use only window data
v_data = data['errors'].tolist()


fig, ax = plt.subplots()
pos = data['window'].tolist()
parts = ax.violinplot(v_data, pos, points=100, widths=0.7, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)
# parts['cmeans'].set_edgecolor('red')
ax.set_ylabel('Degree error')
ax.set_xlabel('Window size')
plt.show()

data.groupby(['window'])['errors'].apply(list)
