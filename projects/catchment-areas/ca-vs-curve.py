import sys
import os

# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import check_for_dir_and_create
from source.routedatabase import load_routes

save_path = os.path.join(fwd, 'figs')
check_for_dir_and_create(save_path)


#ANTWORLD2
###############################################
path = 'projects/catchment-areas/antworld2/2023-09-10'
folder = 'tran_eval'
in_translation = True
path = os.path.join(path, folder)


routes_path = '/its/home/sk526/ant_world_alg_bench/new-antworld/curve-bins'
route_ids = [*range(20)]

routes = load_routes(routes_path, route_ids)
curvatures = []
for route in routes:
    k = route.get_mean_curv()
    curvatures.append(k)

ant_data = []
for r in route_ids:
    file_path = os.path.join(path, f'route{r}-results', 'results.csv')
    df = pd.read_csv(file_path)
    # add the cuvatures to the dataframe
    df.loc[df['route_id'] == r, 'k'] = curvatures[r]
    if not in_translation:
        df['area'] = df['area'].apply(literal_eval)
        df['area_cm'] = df['area_cm'].apply(literal_eval)
        df = df.explode('area_cm')
    # The ant world is in millimetres
    df['area_cm'] = np.array(df['area_cm']) * 10 * 100
    ant_data.append(df)
ant_data = pd.concat(ant_data)


# CA vs Curvature
fig, ax = plt.subplots(figsize=(7, 3))
ant_data.boxplot('area_cm', 'k', ax=ax)

ax.set_title('')
# ax.set_xlabel('route')
ax.set_ylabel('catchment area size [cm]')
plt.tight_layout()
fig.savefig(os.path.join(fwd, 'figs', 'aw2_CA_vs_k'))
plt.show()