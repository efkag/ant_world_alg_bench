import sys
import os

from seaborn import widgets

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
sns.set_context('talk')
# from ast import literal_eval
from source.utils import check_for_dir_and_create, mae


def load_logs(route_id, fname):
    path = os.path.join(fwd, 'ftl-{}'.format(route_id), fname)
    dt = pd.read_csv(path, index_col=False)

    print(dt.columns)

    route = dt.to_dict('list')
    route['x'] = np.array(route.pop(' X'))
    route['y'] = np.array(route.pop(' Y'))
    route['yaw'] = np.array(route.pop(' Rx'))
    return route

def mean_velocity(logs):
    t = log['Time [s]']
    dx = np.diff(log['x'])
    dy = np.diff(log['y'])
    dxy = np.sqrt(dx**2 + dy**2)
    dt = np.diff(t)
    v = dxy/dt
    return np.mean(v)

pm_logs = ['pm0.csv', 'pm1.csv', 'pm2.csv'] 
smw_logs = ['smw0.csv', 'smw1.csv', 'smw2.csv']
prefix = 'ftl-'

pm_mvs = []
smw_mvs = []
for i in range(1, 3):
    for l in pm_logs:
        path = os.path.join(fwd, 'ftl-{}'.format(i), 'training.csv')
        log = load_logs(i, 'testing_' + l)
        mv = mean_velocity(log)
        pm_mvs.append(mv)
        print(mv, 'mm/s')

    for l in smw_logs:
        path = os.path.join(fwd, 'ftl-{}'.format(i), 'training.csv')
        log = load_logs(i, 'testing_' + l)
        mv = mean_velocity(log)
        smw_mvs.append(mv)
        print(mv, 'mm/s')


fig = plt.figure()
x = np.arange(len(pm_mvs))
width = 0.5
plt.bar(x, pm_mvs, width=width, label='pm')
plt.bar(x+width, smw_mvs, width=width, label='smw')

#plt.xticks(['route-1', 'route-2, route-3'])
plt.legend()
plt.show()
