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


pm_logs = ['pm0.csv', 'pm1.csv', 'pm2.csv'] 
smw_logs = ['smw0.csv', 'smw1.csv', 'smw2.csv']

def load_logs(route_id, fname):
    path = os.path.join(fwd, 'ftl-{}'.format(route_id), fname)
    dt = pd.read_csv(path, index_col=False)

    print(dt.columns)

    route = dt.to_dict('list')
    route['x'] = np.array(route.pop(' X'))
    route['y'] = np.array(route.pop(' Y'))
    route['yaw'] = np.array(route.pop(' Rx'))
    return route

route_id = 3
path = os.path.join(fwd, 'ftl-{}'.format(route_id), 'training.csv')
dt = pd.read_csv(path, index_col=False)

print(dt.columns)

route = dt.to_dict('list')
route['x'] = np.array(route.pop(' X'))
route['y'] = np.array(route.pop(' Y'))
route['yaw'] = np.array(route.pop(' Rx'))

# Load background image
# **NOTE** OpenCV uses BGR and Matplotlib uses RGB so convert
background = cv2.imread(os.path.join(fwd, "warped.jpg"))
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(10, 10))
plt.plot(route['x'], route['y'], label='training')
plt.scatter(route['x'][0], route['y'][0])
plt.annotate('Start', (route['x'][0], route['y'][0]))

# Show background
# **NOTE** the extents should correspond to EXPERIMENT_AREA_X and EXPERIMENT_AREA_Y in aligner.py
plt.imshow(background, extent=(-3000.0, 3000.0, 3000.0, -3000.0))

log = 'testing_' + pm_logs[2]
r = load_logs(route_id, log)
plt.plot(r['x'], r['y'], linestyle='dashdot' , label='pm')
plt.scatter(r['x'][-1], r['y'][-1])
plt.annotate('end', (r['x'][-1], r['y'][-1]))

best_i = r[' Best snapshot index']

rx = route['x'][best_i]
ry = route['y'][best_i]

tx = r['x']
ty = r['y']

xs = np.column_stack((rx, tx))
ys = np.column_stack((ry, ty))

for x, y in zip(xs, ys):
    plt.plot(x, y, c='k', linewidth=0.8)


plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")
plt.legend()
plt.tight_layout()
plt.show()
