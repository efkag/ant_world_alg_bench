#!/usr/bin/python3

import sys
import pickle
import os
from source.utils import pre_process, calc_dists
from source import seqnav as spm, perfect_memory as pm
from source import infomax
import pandas as pd
import time
import uuid
import numpy as np
from source import antworld2 as aw
from source.routedatabase import Route, load_routes
from source.imgproc import Pipeline

print('Argument List:', str(sys.argv))

params_path = sys.argv[1]
cwd = os.getcwd()
params_path = cwd + '/' + params_path
dbfile = open(params_path, 'rb')
params = pickle.load(dbfile)
dbfile.close()

chunk = params['chunk']
route_ids = params['route_ids']
routes_path = params['routes_path']
results_path = params['results_path']
chunk_id = params['i']
num_of_repeats = params['num_of_repeats']

# Load all routes
routes = load_routes(routes_path, route_ids)

total_jobs = len(chunk) * len(route_ids)
jobs = 0

log = {'route_id': [], 'num_of_repeat':[], 'nav-name':[], 't':[], 
       'res': [], 'blur': [], 'loc_norm':[], 'gauss_loc_norm':[], 'edge': [],  
       'window': [], 'matcher': [], 'deg_range':[], 
       'segment_len': [], 'trial_fail_count':[], 'mean_error': [], 
       'seconds': [], 'errors': [], 'dist_diff': [], 'index_diff': [], 'window_log': [], 
       'matched_index': [], 'min_dist_index': [] ,  'tx': [], 'ty': [], 'th': [], 'ah': [], 'rmfs_file':[], 'best_sims':[],
       'wave':[], 'tfc_idxs':[]
       }

agent = aw.Agent()

#  Go though all combinations in the chunk
for combo in chunk:

    matcher = combo['matcher']
    window = combo.get('window')
    t = combo['t']
    segment_length = combo.get('segment_l')
    rpt = combo.get('repeat') # the repeat number
    for route in routes:  # for every route
        # route_path = os.path.join(routes_path, '/route' + str(route_id))
        # route = Route(route_path, route_id)
        agent.set_seed(rpt)
        tic = time.perf_counter()
        
        pipe = Pipeline(**combo)
        route_imgs = pipe.apply(route.get_imgs())
        # Run navigation algorithm
        #TODO: Need to select navigator instance based on 
        # information coming from the chunk combo
        if window:
            nav = spm.SequentialPerfectMemory(route_imgs, matcher, **combo)
        elif window == 0:
            nav = pm.PerfectMemory(route_imgs, **combo)
        else:
            infomaxParams = infomax.Params()
            nav = infomax.InfomaxNetwork(infomaxParams, route_imgs, **combo)
        # if segment_length:
        #     traj, nav = agent.segment_test(route, nav, segment_length=segment_length, t=t, r=r, sigma=None, preproc=combo)
        # else:
        #     coords = route.get_starting_coords()
        #     traj, nav = agent.test_nav(coords, nav, t=t, r=r, preproc=combo, sigma=None)
        
        traj, nav = agent.run_agent(route, nav, **combo)

        toc = time.perf_counter()
        time_compl = toc - tic
        # Get the errors and the minimum distant index of the route memory
        errors, min_dist_index = route.calc_errors(traj)
        # Difference between matched index and minimum distance index and distance between points
        matched_index = nav.get_index_log()
        if matched_index:
            index_diffs = np.subtract(min_dist_index, nav.get_index_log()).tolist()
            dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index).tolist()
        else:
            index_diffs = None
            dist_diff = None
        mean_route_error = np.mean(errors)
        window_log = nav.get_window_log()
        rec_headings = nav.get_rec_headings()
        rmf_logs = np.array(nav.get_rsims_log(), dtype=object)
        deg_range = nav.deg_range

        rmf_logs_file = f'{chunk_id}{jobs}_{uuid.uuid4().hex}'
        rmfs_path = os.path.join(results_path, rmf_logs_file)
        np.save(rmfs_path, rmf_logs)


        log['nav-name'].append(nav.get_name())
        log['route_id'].extend([route.get_route_id()])
        log['t'].append(t)
        log['num_of_repeat'].append(rpt)
        log['blur'].extend([combo.get('blur')])
        log['edge'].extend([combo.get('edge_range')])
        log['res'].append(combo.get('shape'))
        log['loc_norm'].append(combo.get('loc_norm'))
        log['gauss_loc_norm'].append(combo.get('gauss_loc_norm'))
        log['wave'].append(combo.get('wave'))
        log['window'].append(combo.get('window'))
        log['matcher'].extend([matcher])
        log['deg_range'].append(deg_range)
        log['segment_len'].append(segment_length)
        log['trial_fail_count'].append(agent.get_trial_fail_count())
        log['tfc_idxs'].append(agent.get_tfc_indices())
        log['mean_error'].append(mean_route_error)
        log['seconds'].append(time_compl)
        log['window_log'].append(window_log)
        log['rmfs_file'].append(rmf_logs_file)
        log['tx'].append(traj['x'].tolist())
        log['ty'].append(traj['y'].tolist())
        log['th'].append(traj['heading'].tolist())
        log['ah'].append(rec_headings)
        log['matched_index'].append(matched_index)
        log['min_dist_index'].append(min_dist_index)
        log['index_diff'].append(index_diffs)
        log['dist_diff'].append(dist_diff)
        log['errors'].append(errors)
        log['best_sims'].append(nav.get_best_sims())
        jobs += 1
        print('{} worker, jobs completed {}/{}'.format(chunk_id, jobs, total_jobs))

df = pd.DataFrame(log)
# check_for_dir_and_create('results')
save_path = os.path.join(results_path, '{}.csv'.format(chunk_id))
df.to_csv(save_path, index=False)
