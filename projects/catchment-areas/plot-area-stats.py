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

path = 'projects/catchment-areas'
folder = 'rot_eval'
in_translation = False
path = os.path.join(path, folder)
routes = [1, 2, 3]


data = []
for r in routes:
    file_path = os.path.join(path, f'route{r}-results', 'results.csv')
    df = pd.read_csv(file_path)
    if not in_translation:
        df['area'] = df['area'].apply(literal_eval)
        df = df.explode('area')
    data.append(df)


data = pd.concat(data)

# data = data.groupby('route_id')["area"].apply(sum).to_frame("area").reset_index()

#data.explode('area')

data.boxplot('area', 'route_id')
plt.show()


x = data['route_id'].to_numpy()
y = data['area'].to_numpy()
# plt.boxplot( )
# plt.show()

