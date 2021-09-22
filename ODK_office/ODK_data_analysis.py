import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from source.display import nans_imgshow, plot_multiline, plot_3d
from source.utils import pol2cart_headings
import pickle

df = pd.read_csv('office/training.csv')
testdf = pd.read_csv('office/testing.csv')

route = df.to_dict('list')

test = testdf.to_dict('list')
print(test.keys())

with open('odk_analysis_data_cc.pickle', "rb") as handler:
    data = pickle.load(handler)
print(data.keys())


ldiff = np.array(test[' Lowest difference'])
mindiff = np.array(data['mindiff'])

plt.plot(range(len(ldiff)), ldiff/np.max(ldiff), label='on robot')
## plt.plot(range(len(mindiff)), mindiff/65536, label='in python')
plt.plot(range(len(mindiff)), mindiff, label='in python')
plt.legend()
plt.show()

h1 = test[' Best heading [degrees]']
h2 = data['heading']
plt.plot(range(len(h1)), h1, label='on robot')
plt.scatter(range(len(h1)), h1, label='on robot')
plt.plot(range(len(h2)), h2, label='in python')
plt.scatter(range(len(h2)), h2, label='in python')
plt.legend()
plt.show()

u, v = pol2cart_headings(h1)
plt.quiver(range(len(h1)), 0, u, v, scale=30, label='on robot', color='r')
u, v = pol2cart_headings(h2)
plt.quiver(range(len(h2)), 0.5, u, v, scale=30, label='in python', color='g')
plt.show()

index = test[' Best snapshot index']
index2 = data['best_index']
plt.plot(range(len(index)), index, label='on robot')
plt.plot(range(len(index2)), index2, label='in python')
plt.legend()
plt.show()
