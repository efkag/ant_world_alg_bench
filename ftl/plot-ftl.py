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

pm_logs = ['pm0.csv', 'pm1.csv', 'pm2.csv'] 
smw_logs = ['smw0.csv', 'smw1.csv', 'smw2.csv']

def load_logs(route_id, fname):
    path = os.path.join(fwd, 'ftl-{}'.format(route_id), fname)
    dt = pd.read_csv(path, index_col=False)

    print(dt.columns)

    route = dt.to_dict('list')
    route['x'] = route.pop(' X')
    route['y'] = route.pop(' Y')
    route['yaw'] = np.array(route.pop(' Rx'))
    return route

route_id = 5
path = os.path.join(fwd, 'ftl-{}'.format(route_id), 'training.csv')
dt = pd.read_csv(path, index_col=False)

print(dt.columns)

route = dt.to_dict('list')
route['x'] = route.pop(' X')
route['y'] = route.pop(' Y')
route['yaw'] = np.array(route.pop(' Rx'))

# plot_route(route)


fig = plt.figure(figsize=(10, 10))
plt.plot(route['x'], route['y'], label='training')

plt.scatter(route['x'][0], route['y'][0])
plt.annotate('Start', (route['x'][0], route['y'][0]))
for i, log in enumerate(pm_logs):
    log = 'testing_' + log
    r = load_logs(route_id, log)
    plt.plot(r['x'], r['y'], '--', label='pm{}'.format(i))
    plt.scatter(r['x'][-1], r['y'][-1])
    plt.annotate('pm{} ends'.format(i), (r['x'][-1], r['y'][-1]))


for i, log in enumerate(smw_logs):
    log = 'testing_' + log
    r = load_logs(route_id, log)
    plt.plot(r['x'], r['y'], linestyle='dashdot' , label='smw{}'.format(i))
    plt.scatter(r['x'][-1], r['y'][-1])
    #plt.annotate('smw{} ends'.format(i), (r['x'][-1], r['y'][-1]))

plt.legend()
plt.tight_layout()
plt.show()