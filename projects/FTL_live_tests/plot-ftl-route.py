import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
from source.utils import animated_window, check_for_dir_and_create
from source.display import plot_ftl_route
from source.routedatabase import Route, BoBRoute
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale=1)


route_id = 3
repeating = True
#route_path = os.path.join(fwd, 'ftl-live-tests', f'r{route_id}')
route_path = os.path.join(f'/its/home/sk526/sussex-ftl-dataset/repeating-routes/route{route_id}')
fig_save_path = os.path.join(route_path, 'figures')
check_for_dir_and_create(fig_save_path)
if repeating:
    #the repeating routes
    repeats = [2, 3, 4, 5]
    #hte reference/training route
    ref_route = 1
    ref_path = os.path.join(route_path, f'N-{ref_route}')
else:
    ref_path = route_path

route_data = os.path.join(ref_path, 'database_entries.csv')
dt = pd.read_csv(route_data, index_col=False)
print(dt.columns)

route = dt.to_dict('list')
route['x'] = route.pop('X [mm]')
route['y'] = route.pop(' Y [mm]')
route['yaw'] = np.array(route.pop(' Heading [degrees]'))

background = cv2.imread(os.path.join(fwd, "warped.png"))
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(4, 4))
plt.plot(route['x'], route['y'], label='training')

plt.scatter(route['x'][0], route['y'][0])
plt.annotate('Start', (route['x'][0], route['y'][0]))


# Show background
# **NOTE** the extents should correspond to EXPERIMENT_AREA_X and EXPERIMENT_AREA_Y in aligner.py
plt.imshow(background, extent=(-3000.0, 3000.0, 3000.0, -3000.0))

plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")


def load_testing_logs(route_path, dname):
    data_path = os.path.join(route_path, dname, 'database_entries.csv')
    dt = pd.read_csv(data_path, index_col=False)

    route = dt.to_dict('list')
    route['x'] = route.pop('X [mm]')
    route['y'] = route.pop(' Y [mm]')
    route['yaw'] = np.array(route.pop(' Heading [degrees]'))
    return route

if repeating:
    for i, rid in enumerate(repeats):
        r = load_testing_logs(route_path, f'N-{rid}')
        plt.plot(r['x'], r['y'], '--')#, label=f'repeat{i}')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_save_path, f'route{route_id}'))
plt.show()