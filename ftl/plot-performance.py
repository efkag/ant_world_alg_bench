import sys
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_context('talk')

data = pd.read_csv("performance_data.csv", delimiter=",")


axis = sns.violinplot(x="Route", y="Time [ms]", hue="Algorithm", data=data)

plt.show()
