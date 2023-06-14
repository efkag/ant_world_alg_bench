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

save_path = os.path.join(fwd, 'figs')
check_for_dir_and_create(save_path)

##############################################
path = 'projects/catchment-areas/2023-06-09'
folder = 'tran_eval'
in_translation = True
path = os.path.join(path, folder)
routes = [1, 2, 3]

ftl_data = []
for r in routes:
    file_path = os.path.join(path, f'route{r}-results', 'results.csv')
    df = pd.read_csv(file_path)
    if not in_translation:
        df['area'] = df['area'].apply(literal_eval)
        df['area_cm'] = df['area_cm'].apply(literal_eval)
        df = df.explode('area_cm')
    ftl_data.append(df)
ftl_data = pd.concat(ftl_data)



###############################################
#ANTWORLD2
path = 'projects/catchment-areas/2023-06-13'
folder = 'tran_eval'
in_translation = True
path = os.path.join(path, folder)
routes = [0, 1, 2, 3, 4, 5, 6]

ant_data = []
for r in routes:
    file_path = os.path.join(path, f'route{r}-results', 'results.csv')
    df = pd.read_csv(file_path)
    if not in_translation:
        df['area'] = df['area'].apply(literal_eval)
        df['area_cm'] = df['area_cm'].apply(literal_eval)
        df = df.explode('area_cm')
    df['area_cm'] = np.array(df['area_cm']) * 10 * 100
    ant_data.append(df)
ant_data = pd.concat(ant_data)


# index space
fig, ax = plt.subplots(figsize=(7, 3))
ftl_data.boxplot('area', 'route_id', ax=ax)
plt.suptitle('')
ax = plt.gca()
ax.set_title('')
plt.xlabel('route')
plt.ylabel('catchment area size')
plt.show()

# in cm 
fig, ax = plt.subplots(figsize=(7, 3))
ftl_data.boxplot('area_cm', 'route_id', ax=ax)
plt.suptitle('')
ax = plt.gca()
ax.set_title('')
plt.xlabel('route')
plt.ylabel('catchment area size [cm]')
plt.tight_layout(pad=0.5)
fig.savefig(os.path.join(fwd, 'figs', 'FTL_CA_in_cm'))
plt.show()



############################################
#joint plot
# in cm 
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharey='row')
ftl_data.boxplot('area_cm', 'route_id', ax=axes[0])
axes[0].set_title('FTL')
axes[0].set_xlabel('route')
axes[0].set_ylabel('catchment area size [cm]')

ant_data.boxplot('area_cm', 'route_id', ax=axes[1])
axes[1].set_title('Antworld 2')
axes[1].set_xlabel('route')
axes[1].set_ylabel('catchment area size [cm]')



plt.suptitle('')
plt.tight_layout(pad=0.5)
fig.savefig(os.path.join(fwd, 'figs', 'joint'))
plt.show()



