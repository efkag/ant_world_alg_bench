import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(path)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
# from ast import literal_eval
from source.utils import load_route_naw, plot_route

pm_logs = ['pm0.csv', 'pm1.csv', 'pm2.csv'] 
smw_logs = ['smw0.csv', 'smw1.csv', 'smw2.csv']

def load_logs(data_path, dname=''):
    data_path = os.path.join(data_path, dname, 'database_entries.csv')
    print(f'reading logs from{dname}')
    dt = pd.read_csv(data_path, index_col=False)

    route = dt.to_dict('list')
    route = dt.to_dict('list')
    route['x'] = np.array(route.pop('X [mm]'))
    route['y'] = np.array(route.pop('Y [mm]'))
    route['yaw'] = np.array(route.pop('Heading [degrees]'))
    return route

def mean_velocity(logs):
    t = logs['Time [s]']
    dx = np.diff(logs['x'])
    dy = np.diff(logs['y'])
    dxy = np.sqrt(dx**2 + dy**2)
    print(np.sum(dxy))
    dt = np.diff(t)
    v = dxy/dt
    return np.mean(v)

route_id = 4 if len(sys.argv) < 2 else int(sys.argv[1])
path = os.path.join(fwd, 'ftl-{}'.format(route_id), 'training.csv')
dt = pd.read_csv(path, index_col=False)

print(dt.columns)

route = dt.to_dict('list')
route['x'] = route.pop(' X')
route['y'] = route.pop(' Y')
route['yaw'] = np.array(route.pop(' Rx'))

mean_velocity(route)

# plot_route(route)

# Load background image
# **NOTE** OpenCV uses BGR and Matplotlib uses RGB so convert
background = cv2.imread(os.path.join(fwd, "top-down-fliped.png"))
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(3, 3))
plt.plot(route['x'], route['y'], label='training')

plt.scatter(route['x'][0], route['y'][0])
plt.annotate('Start', (route['x'][0], route['y'][0]))

# Show background
# **NOTE** the extents should correspond to EXPERIMENT_AREA_X and EXPERIMENT_AREA_Y in aligner.py
plt.imshow(background, extent=(-3000.0, 3000.0, 3000.0, -3000.0))

# plt.xlabel("X [mm]")
# plt.ylabel("Y [mm]")

# for i, log in enumerate(pm_logs):
#     log = 'testing_' + log
#     r = load_logs(route_id, log)
#     plt.plot(r['x'], r['y'], '--', label='pm{}'.format(i))
#     plt.scatter(r['x'][-1], r['y'][-1])
#     plt.annotate('pm{} ends'.format(i), (r['x'][-1], r['y'][-1]))


for i, log in enumerate(smw_logs):
    log = 'testing_' + log
    r = load_logs(route_id, log)
    plt.plot(r['x'], r['y'], linestyle='dashdot' , label='asmw{i}'.format(i))
    plt.scatter(r['x'][-1], r['y'][-1])
    #plt.annotate('smw{} ends'.format(i), (r['x'][-1], r['y'][-1]))

plt.legend()
plt.tight_layout()
plt.show()
