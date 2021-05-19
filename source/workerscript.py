#!/usr/bin/python3

import sys
import pickle
import os

from utils import pre_process, calc_dists, load_route_naw, angular_error, check_for_dir_and_create
import seqnav as spm, perfect_memory as pm
import pandas as pd
import time
import numpy as np
import antworld2 as aw

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
chunk_id = params['i']

total_jobs = len(chunk) * len(route_ids)
jobs = 0

log = {'route_id': [], 'blur': [], 'edge': [], 'res': [], 'window': [],
       'matcher': [], 'mean_error': [], 'seconds': [], 'errors': [],
       'dist_diff': [], 'abs_index_diff': [], 'window_log': [],
       'tx': [], 'ty': [], 'th': []}
agent = aw.Agent()

#  Go though all combinations in the chunk
for combo in chunk:

    matcher = combo['matcher']
    window = combo['window']
    t = combo['t']
    r = combo['r']
    segment_length = combo['segment_l']
    window_log = None
    for route_id in route_ids:  # for every route
        route_path = routes_path + '/route' + str(route_id) + '/'
        route = load_route_naw(route_path, route_id=route_id, imgs=True)

        # Preprocess images
        route_imgs = route['imgs']
        route_imgs = pre_process(route_imgs, combo)
        # Run navigation algorithm
        if window:
            nav = spm.SequentialPerfectMemory(route_imgs, matcher, deg_range=(-180, 180), window=window)
        else:
            nav = pm.PerfectMemory(route_imgs, matcher, deg_range=(-180, 180))

        if segment_length:
            tic = time.perf_counter()
            traj, nav = agent.segment_test(route, nav, segment_length=segment_length, t=t, r=r, sigma=None, preproc=combo)
            toc = time.perf_counter()
        else:
            tic = time.perf_counter()
            traj, nav = agent.test_nav(route, nav, t=t, r=r, sigma=None, preproc=combo)
            toc = time.perf_counter()

        time_compl = toc - tic
        # Get the errors and the minimum distant index of the route memory
        errors, min_dist_index = angular_error(route, traj)
        # Difference between matched index and minimum distance index
        matched_index = nav.get_index_log()
        window_log = nav.get_window_log()
        abs_index_diffs = np.absolute(np.subtract(matched_index, min_dist_index))
        dist_diff = calc_dists(route, min_dist_index, matched_index)
        mean_route_error = np.mean(errors)
        log['route_id'].extend([route_id])
        log['blur'].extend([combo.get('blur')])
        log['edge'].extend([combo.get('edge_range')])
        log['res'].append(combo.get('shape'))
        log['window'].extend([window])
        log['matcher'].extend([matcher])
        log['mean_error'].append(mean_route_error)
        log['seconds'].append(time_compl)
        log['window_log'].append(window_log)
        log['tx'].append(traj['x'].tolist())
        log['ty'].append(traj['y'].tolist())
        log['th'].append(traj['heading'].tolist())
        log['abs_index_diff'].append(abs_index_diffs.tolist())
        log['dist_diff'].append(dist_diff.tolist())
        log['errors'].append(errors)
        jobs += 1
        print('{} worker, jobs completed {}/{}'.format(chunk_id, jobs, total_jobs))

df = pd.DataFrame(log)
check_for_dir_and_create('results')
df.to_csv('results/{}.csv'.format(chunk_id), index=False)
