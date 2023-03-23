import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import yaml
from source.utils import cor_dist, mae, check_for_dir_and_create, scale2_0_1
from source.analysis import flip_gauss_fit, eval_gauss_rmf_fit, d2i_rmfs_eval
sns.set_context("paper", font_scale=1)


directory = '2023-01-20_mid_update'
results_path = os.path.join('Results', 'newant', directory)
fig_save_path = os.path.join('Results', 'newant', directory, 'analysis')
with open(os.path.join(results_path, 'params.yml')) as fp:
    params = yaml.load(fp)
routes_path = params['routes_path']
data = pd.read_csv(os.path.join(results_path, 'results.csv'), index_col=False)
# Convert list of strings to actual list of lists
data['errors'] = data['errors'].apply(literal_eval)
data['dist_diff'] = data['dist_diff'].apply(literal_eval)
data['abs_index_diff'] = data['abs_index_diff'].apply(literal_eval)
data['tx'] = data['tx'].apply(literal_eval)
data['ty'] = data['ty'].apply(literal_eval)
data['th'] = data['th'].apply(literal_eval)
data['matched_index'] = data['matched_index'].apply(literal_eval)


route_id = 5
window = -15
blur =  True
matcher = 'corr'
edge = 'False'# '(180, 200)'
loc_norm = 'False' # {'kernel_shape':(5, 5)}
gauss_loc_norm = "{'sig1': 2, 'sig2': 20}"
res = '(180, 80)'
threshold = 0
figsize = (6, 3)


# filter data
traj = data.loc[(data['matcher'] == matcher) & (data['res'] == res) 
                #& (data['edge'] == edge) 
                & (data['window'] == window) 
                & (data['blur'] == blur)
                #& (data['loc_norm'] == loc_norm) 
                & (data['gauss_loc_norm'] == gauss_loc_norm)
                & (data['route_id'] == route_id)
                ]
traj = traj.to_dict(orient='records')[0]
traj['window_log'] = literal_eval(traj['window_log'])


traj['best_sims'] = literal_eval(traj['best_sims'])
traj['rmfs'] = np.load(os.path.join(results_path, traj['rmfs_file']+'.npy'), allow_pickle=True)



####
#some common window params
min_window = 10
window = None
####
# some update function each with their own criterion
def simple_log_update(prev_sim, curr_sim, window):
    if curr_sim > prev_sim or window <= min_window:
        window += round(min_window/np.log(window))
    else:
        window -= round(np.log(window))
    return window


def thresh_log_update(prev_sim, curr_sim, window, thresh=0.1):
    # threshold against the percentage change of match quality
    perc_cng = (curr_sim - prev_sim)/prev_sim
    if perc_cng > thresh or window <= min_window:
        window += round(min_window/np.log(window))
    else:
        window -= round(np.log(window))
    return window

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def sma_log_update(prev_sim, curr_sim, window, ):
    
    if curr_sim > thresh or window <= min_window:
        window += round(min_window/np.log(window))
    else:
        window -= round(np.log(window))
    return window


########
# Test a small section.
# set up
# start end index
start_i = 25
end_i = 40
window_log = []
window = traj['window_log'][start_i-1][1] - traj['window_log'][start_i-1][0]
for i in range(start_i, end_i):
    curr_sim = traj['best_sims'][i]
    prev_sim = traj['best_sims'][i-1]

    # add here a new criterion for window update
    window = thresh_log_update(prev_sim, curr_sim, window)
    window_log.append(window)


fig, ax1 = plt.subplots(figsize=figsize)
ax1.plot(traj['best_sims'][start_i:end_i], color='g', label='image diff.')

ax2 = ax1.twinx()
ax2.plot(np.diff(traj['window_log'][start_i:end_i], axis=1), color="orange", label='window size')
ax2.plot(window_log, label='thresh=0.1%')
ax1.legend(loc=2)
ax2.legend(loc=0)
plt.show()
plt.close()


####################
# replot for the entire trajectory
w_size = np.squeeze(np.diff(traj['window_log'], axis=1))
thresh = [.1, 0.05]
window_per_thresh = []
for th in thresh:
    window = w_size[0]
    window_log = []
    window_log.append(window)
    for i in range(len(traj['best_sims'])-1):
        curr_sim = traj['best_sims'][i+1]
        prev_sim = traj['best_sims'][i]

        # add here a new criterion for window update
        window = thresh_log_update(prev_sim, curr_sim, window, thresh=th)
        window_log.append(window)
    window_per_thresh.append(window_log)

###### 
# # test another criterion
# use eval metrics to get a score
rsims = []
for i in range(len(traj['rmfs'])):
    w = traj.get('window_log')[i]
    window_index_of_route_match = traj['matched_index'][i] - w[0]
    rsim = traj['rmfs'][i][window_index_of_route_match]
    rsims.append(rsim)
gauss_scores = eval_gauss_rmf_fit(rsims)
#weighted_gauss_scores = eval_gauss_rmf_fit(rsims, weighted=True)
#rsims = np.array(rsims)
d2i_scores = d2i_rmfs_eval(rsims)


fig, ax1 = plt.subplots(figsize=figsize)
#plt.title(title, loc="left")
#ax1.plot(range(len(traj['abs_index_diff'])), traj['abs_index_diff'], label='index missmatch')
ax1.set_ylim([0, 260])
ax1.plot(range(len(w_size)), w_size, label='window size')
for i, th in enumerate(thresh):
    ax1.plot(window_per_thresh[i], label=f'thresh={th}%')

ax1.set_ylabel('route index scale')

ax2 = ax1.twinx()
ax2.plot(range(len(traj['best_sims'])), traj['best_sims'], label='image diff.', color='g')
ax2.plot(gauss_scores, label='gauss', color='m')
ax2.set_ylim([0.0, 1.0])
ax2.set_ylabel(f'{matcher} image distance')
ax1.legend(loc=0)
ax2.legend(loc=2)
plt.show()


#######################
## plot the just scores 
fig, ax1 = plt.subplots(figsize=figsize)

ax1.plot(scale2_0_1(gauss_scores), label='gauss')
ax1.plot(scale2_0_1(d2i_scores), label='d2i')
ax1.set_ylabel('quality scores')
ax1.set_xlabel('test points')
#plt.plot(scale2_0_1(weighted_gauss_scores), label='w_gauss')
ax2 = ax1.twinx()
ax2.plot(range(len(traj['best_sims'])), traj['best_sims'], label='image diff.', color='g')
ax2.set_ylabel('cc image distance')
#ax2.set_ylim([0.0, 1.0])

plt.legend()
plt.show()


#######################
## plot the scores and windows
fig, ax1 = plt.subplots(figsize=figsize)
#plt.title(title, loc="left")
ax1.set_ylim([0, 260])
ax1.plot(range(len(w_size)), w_size, label='window size')
for i, th in enumerate(thresh):
    ax1.plot(window_per_thresh[i], label=f'thresh={th}%')

ax1.set_ylabel('route index scale')

ax2 = ax1.twinx()
ax2.plot(scale2_0_1(gauss_scores), label='gauss', color='r')
ax2.plot(scale2_0_1(d2i_scores), label='d2i', color='c')
ax2.plot(range(len(traj['best_sims'])), traj['best_sims'], label='best sim', color='m')
#ax2.set_ylim([0.0, 1.0])
ax2.set_ylabel('scores')
ax1.legend(loc=2)
ax2.legend(loc=0)
plt.show()