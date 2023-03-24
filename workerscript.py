#!/usr/bin/python3

import sys
import pickle
import os
from source.utils import pre_process, calc_dists
from source import seqnav as spm, perfect_memory as pm
import pandas as pd
import time
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
# Load all routes
routes = load_routes(routes_path, route_ids)

total_jobs = len(chunk) * len(route_ids)
jobs = 0

log = {'route_id': [], 't':[], 'blur': [], 'edge': [], 'res': [], 'window': [],
       'matcher': [], 'deg_range':[], 'segment_len': [], 'trial_fail_count':[], 'mean_error': [], 
       'seconds': [], 'errors': [], 'dist_diff': [], 'abs_index_diff': [], 'window_log': [], 
       'matched_index': [], 'tx': [], 'ty': [], 'th': [], 'rmfs_file':[], 'best_sims':[],
       'loc_norm':[], 'gauss_loc_norm':[], 'wave':[]}
agent = aw.Agent()

#  Go though all combinations in the chunk
for combo in chunk:

    matcher = combo['matcher']
    window = combo['window']
    t = combo['t']
    segment_length = combo.get('segment_l')
    for route in routes:  # for every route
        # route_path = os.path.join(routes_path, '/route' + str(route_id))
        # route = Route(route_path, route_id)

        tic = time.perf_counter()
        
        pipe = Pipeline(**combo)
        route_imgs = pipe.apply(route.get_imgs())
        # Run navigation algorithm
        if window:
            nav = spm.SequentialPerfectMemory(route_imgs, matcher, deg_range=(-180, 180), **combo)
        else:
            nav = pm.PerfectMemory(route_imgs, matcher, deg_range=(-180, 180), **combo)

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
        abs_index_diffs = np.absolute(np.subtract(nav.get_index_log(), min_dist_index))
        dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index)
        mean_route_error = np.mean(errors)
        window_log = nav.get_window_log()
        rmf_logs = np.array(nav.get_rsims_log(), dtype=object)
        deg_range = nav.deg_range

        rmf_logs_file = 'rmfs' + str(chunk_id) + str(jobs)
        rmfs_path = os.path.join(results_path, rmf_logs_file)
        np.save(rmfs_path, rmf_logs)



        log['route_id'].extend([route.get_route_id()])
        log['t'].append(t)
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
        log['mean_error'].append(mean_route_error)
        log['seconds'].append(time_compl)
        log['window_log'].append(window_log)
        log['rmfs_file'].append(rmf_logs_file)
        log['tx'].append(traj['x'].tolist())
        log['ty'].append(traj['y'].tolist())
        log['th'].append(traj['heading'].tolist())
        log['matched_index'].append(matched_index)
        log['abs_index_diff'].append(abs_index_diffs.tolist())
        log['dist_diff'].append(dist_diff.tolist())
        log['errors'].append(errors)
        log['best_sims'].append(nav.get_best_sims())
        jobs += 1
        print('{} worker, jobs completed {}/{}'.format(chunk_id, jobs, total_jobs))

df = pd.DataFrame(log)
# check_for_dir_and_create('results')
save_path = os.path.join(results_path, '{}.csv'.format(chunk_id))
df.to_csv(save_path, index=False)
