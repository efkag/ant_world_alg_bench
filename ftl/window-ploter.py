import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
# from ast import literal_eval
from source.utils import load_route_naw, plot_route

def load_logs(route_id, fname):
    path = os.path.join(fwd, 'ftl-{}'.format(route_id), fname)
    dt = pd.read_csv(path, index_col=False)

    print(dt.columns)

    route = dt.to_dict('list')
    route['x'] = route.pop(' X')
    route['y'] = route.pop(' Y')
    route['yaw'] = np.array(route.pop(' Rx'))
    return route

pm_logs = ['pm0.csv', 'pm1.csv', 'pm2.csv'] 
smw_logs = ['smw0.csv', 'smw1.csv', 'smw2.csv']

route_id = 2
path = os.path.join(fwd, 'ftl-{}'.format(route_id), 'training.csv')
dt = pd.read_csv(path, index_col=False)

print(dt.columns)

route = dt.to_dict('list')
route['x'] = route.pop(' X')
route['y'] = route.pop(' Y')
route['yaw'] = np.array(route.pop(' Rx'))


fig = plt.figure(figsize=(10, 10))
plt.plot(route['x'], route['y'], label='training')

log = load_logs(route_id, 'testing_smw1.csv')


for i, (ws, we) in enumerate(zip(log[' Window start'], log[' Window end'])):
    plt.plot(route['x'], route['y'], label='training')
    plt.plot(route['x'][ws:we], route['y'][ws:we], label='window')

    plt.plot(log['x'][:i], log['y'][:i], '--', label='smw')

    plt.draw()
    plt.pause(0.5)
    plt.clf()