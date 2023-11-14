import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from source.utils import load_route_naw, plot_route, animated_window, check_for_dir_and_create
sns.set_context("paper", font_scale=1)

directory = '2023-04-26/combined'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join(results_path, 'analysis')
check_for_dir_and_create(fig_save_path)
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
data['errors'] = data['errors'].apply(literal_eval)


# Choose a specific pre. processing
# route_id = 7
matcher = 'corr'
blur = True
# edge = 'False' 
res = '(180, 80)'
g_loc_norm = 'False' # "{'sig1': 2, 'sig2': 20}"
# loc_norm = 'False'
title = None

# imax_df = data.loc[data['nav-name'] == 'InfoMax']
# data = pd.concat([data, imax_df])

data = data.loc[(data['matcher'] == matcher) 
                & (data['res'] == res) 
                & (data['blur'] == blur) 
                #& (data['edge'] == edge) 
                & (data['gauss_loc_norm'] == g_loc_norm)
                #& (data['loc_norm'] == loc_norm)
                #& (data['route_id'] == route_id ) 
                # & (data['num_of_repeat'] == 0) 
             #& (data['route_id'] <= 4 ) #& (data['route_id'] < 10 )
                
]



#################
# in case of repeats
method = np.mean
   
#data = data.groupby(['window', 'route_id'])["trial_fail_count"].apply(method).to_frame("trial_fail_count").reset_index()
##### if the dataset had nav-names
grouped = data.groupby(['route_id', 'nav-name'])["trial_fail_count"].apply(method).to_frame("trial_fail_count").reset_index()
#data['errors'] = data['errors'].apply(np.array)

#data['mean_aae'] = np.mean(data['errors'])

grouped = grouped.pivot('route_id', 'nav-name', 'trial_fail_count')
grouped.to_csv(os.path.join(fig_save_path, 'grouped.csv'), index=False)

# pm = 
#use stats.ttest_ind(pm, asmw, equal_var=False)