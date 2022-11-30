import os, sys

# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from source.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rotate, check_for_dir_and_create
from source.routedatabase import Route
from ast import literal_eval

fig_save_path = os.path.join(fwd, 'figures')
check_for_dir_and_create(fig_save_path)

directory = '2022-11-23_mid_update'
fig_save_path = os.path.join('Results', 'newant', directory)
data = pd.read_csv(os.path.join(fig_save_path, 'results.csv'), index_col=False)
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)

#metric =  'mean_error'
metric = 'errors'
#metric =  'trial_fail_count'

window = 20
data = data.loc[data['window'] == window]
data = data.groupby('route_id')[metric].apply(sum).to_frame(metric).reset_index()

# sns.scatterplot(data=data, x='route_id', y=metric)

data = data.explode(metric)
x = data['route_id'].to_numpy(dtype=np.float)
y = data[metric].to_numpy(dtype=np.float)
sns.violinplot(x=x, y=y)
plt.show()
