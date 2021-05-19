from source.utils import pre_process, calc_dists, load_route_naw, seq_angular_error, check_for_dir_and_create
from source import seqnav as spm, perfect_memory as pm
import time
import itertools
import os
import pandas as pd
import numpy as np
from source import antworld2 as aw
import pickle
from subprocess import Popen
from queue import Queue, Empty
from threading import Thread
import sys


def get_grid_dict(params):
    grid = itertools.product(*[params[k] for k in params])
    grid_dict = []
    for combo in grid:
        combo_dict = {}
        for i, k in enumerate(params):
            combo_dict[k] = combo[i]
        grid_dict.append(combo_dict)

    grid_dict[:] = [x for x in grid_dict if remove_blur_edge(x)]
    grid_dict[:] = [x for x in grid_dict if not remove_non_blur_edge(x)]

    return grid_dict


def remove_blur_edge(combo):
    return not (combo['edge_range'] and combo['blur'])


def remove_non_blur_edge(combo):
    return not combo['edge_range'] and not combo['blur']


def bench(params, routes_path, route_ids):
    log = {'route_id': [], 'blur': [], 'edge': [], 'res': [], 'window': [],
           'matcher': [], 'mean_error': [], 'seconds': [], 'errors': [],
           'dist_diff': [], 'abs_index_diff': [], 'window_log': [],
           'tx': [], 'ty': [], 'th': []}
    agent = aw.Agent()

    grid = get_grid_dict(params)
    total_jobs = len(grid) * len(route_ids)
    jobs = 0
    #  Go though all combinations in the chunk
    for combo in grid:

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
            errors, min_dist_index = seq_angular_error(route, traj)
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

            # Increment the complete jobs shared variable
            jobs += 1
            print('jobs completed: {}/{}'.format(jobs, total_jobs))
    return log


def benchmark(results_path, routes_path, params, route_ids,  parallel=False, cores=None):

    assert isinstance(params, dict)
    assert isinstance(route_ids, list)

    if parallel:
        bench_paral(params, routes_path, route_ids, cores)
        # log = unpack_results(log)
    else:
        log = bench(params, routes_path, route_ids)
        bench_results = pd.DataFrame(log)
        bench_results.to_csv(results_path, index=False)


def _total_jobs(params):
    total_jobs = 1
    for k in params:
        total_jobs = total_jobs * len(params[k])
    print('Total number of jobs: {}'.format(total_jobs))
    return total_jobs


def get_grid_chunks(grid_gen, chunks=1):
    lst = list(grid_gen)
    return [lst[i::chunks] for i in range(chunks)]


def unpack_results(results):
    results = results.get()
    print(len(results), 'Results produced')
    log = results[0]
    for dictionary in results[1:]:
        for k in dictionary:
            log[k].extend(dictionary[k])
    return log


def bench_paral(params, routes_path, route_ids=None, cores=None):
    existing_cores = os.cpu_count()
    print(existing_cores, ' CPU cores found')
    if cores and cores <= existing_cores:
        existing_cores = cores


    grid = get_grid_dict(params)
    total_jobs = len(grid)

    if total_jobs < existing_cores:
        no_of_chunks = total_jobs
    else:
        no_of_chunks = existing_cores - 1
    # Generate a list of chunks of grid combinations
    chunks = get_grid_chunks(grid, no_of_chunks)
    print('{} combinations, testing on {} routes, running on {} cores'.format(total_jobs, len(route_ids), no_of_chunks))
    chunks_path = 'chunks'
    check_for_dir_and_create(chunks_path)
    print('Saving chunks in', chunks_path)

    # Pickle the parameter object to use in the worker script
    for i, chunk in enumerate(chunks):
        params = {'chunk': chunk, 'route_ids': route_ids, 'routes_path': routes_path, 'i': i}
        with open('chunks/chunk{}.p'.format(i), 'wb') as file:
            pickle.dump(params, file)
    print('{} chunks pickled'.format(no_of_chunks))

    processes = []
    for i, chunk in enumerate(chunks):
        cmd_list = ['python3', 'workerscript.py', 'chunks/chunk{}.p'.format(i)]
        p = Popen(cmd_list)
        processes.append(p)

    for p in processes:
        p.wait()


def enqueue_output(out, queue):
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()


def print_stdout_from_procs(processes):
    q = Queue()
    threads = []
    for p in processes:
        threads.append(Thread(target=enqueue_output, args=(p.stdout, q)))

    for t in threads:
        t.daemon = True
        t.start()

    while True:
        try:
            line = q.get_nowait()
        except Empty:
            pass
        else:
            sys.stdout.write(line)

        # break when all processes are done.
        if all(p.poll() is not None for p in processes):
            break

    for t in threads:
        t.join()

    for p in processes:
        p.stdout.close()

    print('All processes done')

