import sys
import os

# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from ast import literal_eval

path = 'projects/catchment-areas/2023-06-09'
folder = 'tran_eval'
in_translation = True
path = os.path.join(path, folder)
routes = [1, 2, 3]




data = []
for r in routes:
    file_path = os.path.join(path, f'route{r}-results', 'results.csv')
    df = pd.read_csv(file_path)
    if not in_translation:
        df['area'] = df['area'].apply(literal_eval)
        df['area_cm'] = df['area_cm'].apply(literal_eval)
        df = df.explode('area_cm')
    data.append(df)


data = pd.concat(data)

# data = data.groupby('route_id')["area"].apply(sum).to_frame("area").reset_index()

#data.explode('area')

data.boxplot('area', 'route_id')
plt.suptitle('')
ax = plt.gca()
ax.set_title('')
plt.xlabel('route')
plt.ylabel('catchment area size')
plt.show()

# in cm 
data.boxplot('area_cm', 'route_id')
plt.suptitle('')
ax = plt.gca()
ax.set_title('')
plt.xlabel('route')
plt.ylabel('catchment area size [cm]')
plt.show()


x = data['route_id'].to_numpy()
y = data['area'].to_numpy()
# plt.boxplot( )
# plt.show()

