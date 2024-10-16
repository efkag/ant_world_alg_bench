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
sns.set_context('talk')

def load_testing_logs(route_path, dname):
    data_path = os.path.join(route_path, dname, 'database_entries.csv')
    dt = pd.read_csv(data_path, index_col=False)

    route = dt.to_dict('list')
    route['x'] = np.array(route.pop('X [mm]'))
    route['y'] = np.array(route.pop(' Y [mm]'))
    route['yaw'] = np.array(route.pop(' Heading [degrees]'))
    return route

route_id=2
pm_logs = ['pm0', 'pm1', 'pm2', 'pm3', 'pm4'] 
asmw_logs = ['asmw0', 'asmw1', 'asmw2', 'asmw3', 'asmw4'] 
title = None
trial_logs = pm_logs

route_path = os.path.join(fwd,'2023-10-03', f'route{route_id}')
fig_save_path = os.path.join(route_path, 'analysis')
check_for_dir_and_create(fig_save_path)
route_data = os.path.join(route_path, 'database_entries.csv')
dt = pd.read_csv(route_data, index_col=False)
print(dt.columns)

route = dt.to_dict('list')
route['x'] = np.array(route.pop('X [mm]'))
route['y'] = np.array(route.pop(' Y [mm]'))
route['yaw'] = np.array(route.pop(' Heading [degrees]'))

background = cv2.imread(os.path.join(fwd, "top-down-fliped.png"))
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)




'''
Here all the xy are flipped and rotated by 270 degreee to  plot aproaproiately
-y, -x => x, y
'''


fig = plt.figure(figsize=(5, 5))
#plt.title(title, loc='left')
plt.plot(route['x'], route['y'], label='training')

plt.scatter(route['x'][0], route['y'][0])
plt.annotate('Start', (route['x'][0], route['y'][0]))
# Show background
# **NOTE** the extents should correspond to EXPERIMENT_AREA_X and EXPERIMENT_AREA_Y in aligner.py
plt.imshow(background, extent=(-3000.0, 3000.0, 3000.0, -3000.0))

# plt.xlabel("X [mm]")
# plt.ylabel("Y [mm]")


'''
Plot trials
'''

# logs_path = os.path.join(route_path, 'testing')
# for i, log in enumerate(pm_logs):
#     r = load_testing_logs(logs_path, log)
#     # if i==0:
#     # use the label for the legend if it is the first iteration
#     plt.plot(-r['y'], -r['x'], ':', label=f'pm{i}')
#     # else:
#     #     plt.plot(-r['y'], -r['x'], ':')
#     plt.scatter(-r['y'][-1], -r['x'][-1])
#     #plt.annotate('pm{} ends'.format(i), (r['x'][-1], r['y'][-1]))

if 'pm' in trial_logs[0]:
    navalg = 'pm'
elif 'asmw' in trial_logs[0]:
    navalg = 'asmw'
else:
    raise Exception('Probably wrong trial directory name')

logs_path = os.path.join(route_path, 'testing')
for i, log in enumerate(trial_logs):
    r = load_testing_logs(logs_path, log)
    # use the label for the legend if it is the first iteration
    # if i==0:
    plt.plot(r['x'], r['y'], '--', label=f'{navalg}{i}')
    # else:
    #     plt.plot(-r['y'], -r['x'], '--')
    plt.scatter(r['x'][-1], r['y'][-1])
    #plt.annotate('asmw{} ends'.format(i), (r['x'][-1], r['y'][-1]))


plt.xticks([])
plt.yticks([])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)


plt.legend()
plt.tight_layout()
fig.savefig(os.path.join(fig_save_path, f'r({route_id})-{trial_logs}.png'))
fig.savefig(os.path.join(fig_save_path, f'r({route_id})-{trial_logs}.pdf'))
plt.show()