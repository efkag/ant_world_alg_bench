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

def load_testing_logs(route_path, dname):
    data_path = os.path.join(route_path, dname, 'database_entries.csv')
    dt = pd.read_csv(data_path, index_col=False)

    route = dt.to_dict('list')
    route['x'] = route.pop('X [mm]')
    route['y'] = route.pop(' Y [mm]')
    route['yaw'] = np.array(route.pop(' Heading [degrees]'))
    return route

route_id=4
pm_logs = ['pm0', 'pm1', 'pm2', 'pm3', 'pm4'] 
asmw_logs = ['asmw0', 'asmw1', 'asmw2', 'asmw3', 'asmw4'] 

route_path = os.path.join(fwd, 'ftl-live-tests', f'r{route_id}')
route_data = os.path.join(route_path, 'database_entries.csv')
dt = pd.read_csv(route_data, index_col=False)
print(dt.columns)

route = dt.to_dict('list')
route['x'] = route.pop('X [mm]')
route['y'] = route.pop(' Y [mm]')
route['yaw'] = np.array(route.pop(' Heading [degrees]'))

background = cv2.imread(os.path.join(fwd, "warped.png"))
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(10, 10))
plt.plot(route['x'], route['y'], label='training')

plt.scatter(route['x'][0], route['y'][0])
plt.annotate('Start', (route['x'][0], route['y'][0]))
# Show background
# **NOTE** the extents should correspond to EXPERIMENT_AREA_X and EXPERIMENT_AREA_Y in aligner.py
plt.imshow(background, extent=(-3000.0, 3000.0, 3000.0, -3000.0))

plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")


'''
Plot trials
'''
# logs_path = os.path.join(route_path, 'testing')
# for i, log in enumerate(pm_logs):
#     r = load_testing_logs(logs_path, log)
#     plt.plot(r['x'], r['y'], '--', label='pm{}'.format(i))
#     plt.scatter(r['x'][-1], r['y'][-1])
#     plt.annotate('pm{} ends'.format(i), (r['x'][-1], r['y'][-1]))


logs_path = os.path.join(route_path, 'testing')
for i, log in enumerate(asmw_logs):
    r = load_testing_logs(logs_path, log)
    plt.plot(r['x'], r['y'], '--', label='asmw{}'.format(i))
    plt.scatter(r['x'][-1], r['y'][-1])
    plt.annotate('asmw{} ends'.format(i), (r['x'][-1], r['y'][-1]))





plt.legend()
plt.tight_layout()
plt.show()