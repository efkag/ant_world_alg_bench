import sys
import os

path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from source.utils import check_for_dir_and_create
sns.set_context('paper')

def load_testing_logs(route_path, dname):
    data_path = os.path.join(route_path, dname, 'database_entries.csv')
    dt = pd.read_csv(data_path, index_col=False)
    dt.rename(str.strip, axis='columns', inplace=True, errors="raise")
    route = dt.to_dict('list')
    route['x'] = np.array(route.pop('X [mm]'))
    route['y'] = np.array(route.pop('Y [mm]'))
    route['yaw'] = np.array(route.pop('Heading [degrees]'))
    route['matched_index'] = route.pop('Best snapshot index')
    return route

pm_logs = ['pm0', 'pm1', 'pm2', 'pm3', 'pm4'] 
asmw_logs = ['asmw0', 'asmw1', 'asmw2', 'asmw3', 'asmw4'] 

directory = '2023-10-03'
route_id=3
trial_name = asmw_logs[1]
title = 'A'

route_path = os.path.join(fwd, directory, f'route{route_id}')
fig_save_path = os.path.join(route_path, 'analysis')
check_for_dir_and_create(fig_save_path)
route_data = os.path.join(route_path, 'database_entries.csv')
dt = pd.read_csv(route_data, index_col=False)
print(dt.columns)

#route data
route = dt.to_dict('list')
route['x'] = np.array(route.pop('X [mm]'))
route['y'] = np.array(route.pop(' Y [mm]'))
route['yaw'] = np.array(route.pop(' Heading [degrees]'))


#trial data
logs_path = os.path.join(route_path, 'testing')
trial = load_testing_logs(logs_path, trial_name )


background = cv2.imread(os.path.join(fwd, "top-down-fliped.png"))
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)




'''
Here all the xy are flipped and rotated by 270 degreee to  plot aproaproiately
-y, -x => x, y
'''


fig = plt.figure(figsize=(3, 3))
plt.title(title, loc='left')
plt.plot(route['x'], route['y'], label='training', linewidth=2)

plt.scatter(route['x'][0], route['y'][0])
plt.annotate('Start', (route['x'][0], route['y'][0]))
# Show background
# **NOTE** the extents should correspond to EXPERIMENT_AREA_X and EXPERIMENT_AREA_Y in aligner.py
plt.imshow(background, extent=(-3000.0, 3000.0, 3000.0, -3000.0))
plt.xticks([])
plt.yticks([])
# plt.xlabel("X [mm]")
# plt.ylabel("Y [mm]")


'''
Plot trial
'''




best_i = trial['matched_index']

# here also flip the signs and y, x around
rx = route['x'][best_i]
ry = route['y'][best_i]

tx = trial['x']
ty = trial['y']

xs = np.column_stack((rx, tx))
ys = np.column_stack((ry, ty))



for x, y in zip(xs, ys):
    plt.plot(x, y, c='k', linewidth=0.8)


plt.plot(trial['x'], trial['y'], '--', label=trial_name, linewidth=2)
plt.annotate(f'{trial_name} ends', (trial['x'][-1], trial['y'][-1]))



plt.legend(loc=4)
plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, f'matches-r({route_id})-{trial_name}.pdf'))
fig.savefig(os.path.join(fig_save_path, f'matches-r({route_id})-{trial_name}.png'))
plt.show()
