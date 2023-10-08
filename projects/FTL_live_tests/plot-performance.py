import sys
import os

fwd = os.path.dirname(__file__)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_context('talk')

path = os.path.join(fwd, "performance_data.csv")
data = pd.read_csv(path, delimiter=",")

print(np.mean(data['Time [ms]']))
print(np.median(data['Time [ms]']))

fig = plt.figure(figsize=(10, 7))
axis = sns.violinplot(x="Route", y="Time [ms]", hue="Algorithm", data=data, cut=0)

axis.axhline(4.2, linestyle="--", color="gray", label='FLaME')

axis.axhline(2.86, linestyle="dotted", color="blue")

plt.legend()
plt.show()

smw = data[data['Algorithm'] == 'SMW']
print(np.mean(smw['Time [ms]']))