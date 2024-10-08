from source import utils
from source.utils import pre_process, calc_dists, load_route_naw, seq_angular_error, check_for_dir_and_create
from source import seqnav as spm, perfect_memory as pm
from source.routedatabase import Route
import time
import itertools
import os
import copy
import pandas as pd
import numpy as np
import pickle
from subprocess import Popen
from queue import Queue, Empty
from threading import Thread
import sys
import yaml


def get_grid_dict(params, nav_name=None):
    grid = itertools.product(*[params[k] for k in params])
    grid_dict = []
    for combo in grid:
        combo_dict = {}
        if nav_name: combo_dict['nav-class'] = nav_name
        for i, k in enumerate(params):
            combo_dict[k] = combo[i]
        grid_dict.append(combo_dict)

    return grid_dict

def remove_blur_edge(combo):
    return not (combo.get('edge_range') and combo.get('blur'))

def remove_non_blur_edge(combo):
    return not combo.get('edge_range') and not combo.get('blur') and not combo.get('gauss_loc_norm') and not combo.get('loc_norm')

def benchmark(results_path: str, routes_path: str, params: dict, nav_params:dict,
              route_ids: list,  parallel:bool =False, cores: int=1, num_of_repeats:int =None):

    assert isinstance(params, dict)
    assert isinstance(route_ids, list)

    if parallel:
        bench_paral(results_path, params, nav_params, routes_path, route_ids, cores, 
                    num_of_repeats=num_of_repeats)
        # log = unpack_results(log)
    else:
        bench_paral(results_path, params, nav_params, routes_path, route_ids, cores=1, 
                    num_of_repeats=num_of_repeats)
    #     log = bench(params, routes_path, route_ids)
    #     bench_results = pd.DataFrame(log)
    #     bench_results.to_csv(results_path, index=False)


def _total_jobs(params):
    total_jobs = 1
    for k in params:
        total_jobs = total_jobs * len(params[k])
    print('Total number of jobs: {}'.format(total_jobs))
    return total_jobs


def get_grid_chunks(grid_gen, chunks=1):
    lst = list(grid_gen)
    return [lst[i::chunks] for i in range(chunks)]


def bench_paral(results_path, params, nav_params, routes_path, route_ids=None, cores=None, num_of_repeats=None):
    # save the parameters of the test in a json file
    check_for_dir_and_create(results_path)
    check_for_dir_and_create(os.path.join(results_path, 'metadata'))
    param_path = os.path.join(results_path, 'params.yml')
    #params['route_ids'] = route_ids
    temp_params = copy.deepcopy(params)
    temp_params['routes_path'] = routes_path
    temp_params['route_ids'] = route_ids
    temp_params.update(nav_params)
    # if there are no repeats then only one repeat per route
    if not num_of_repeats:  
        num_of_repeats = 1
    temp_params['num_of_repeats'] = num_of_repeats
    #save params to yaml
    with open(param_path, 'w') as fp:
        yaml.dump(temp_params, fp)

    existing_cores = os.cpu_count()
    if cores and cores > existing_cores:
        cores = existing_cores - 1
    elif cores and cores <= existing_cores:
        cores = cores
    else:
        cores = existing_cores - 1
    print(existing_cores, ' CPU cores found. Using ', cores, ' cores')

    # global grid for pre-proc
    params_grid_list = get_grid_dict(params)
    nav_grid_list = []
    for nav_k in nav_params:
        nav_grid = get_grid_dict(nav_params[nav_k], nav_name=nav_k)
        nav_grid_list.extend(nav_grid)
    grid = []
    # for each navigator
    for combo_nav in nav_grid_list:
        # for each pre-proc 
        for combo_pp in params_grid_list:
            interim_dict = dict(combo_nav)
            interim_dict.update(combo_pp)
            grid.append(interim_dict)
    total_jobs = len(grid)

    if total_jobs < cores:
        no_of_chunks = total_jobs
    else:
        no_of_chunks = cores
    # Generate a list of chunks of grid combinations
    chunks = get_grid_chunks(grid, no_of_chunks)
    print('{} combinations, testing on {} routes, running on {} cores'.format(total_jobs, len(route_ids), no_of_chunks))
    chunks_path = 'chunks'
    check_for_dir_and_create(chunks_path, remove=True)
    print('Saving chunks in', chunks_path)

    # Pickle the parameter object to use in the worker script
    for i, chunk in enumerate(chunks):
        params = {'chunk': chunk, 'route_ids': route_ids, 
                'routes_path': routes_path, 
                'results_path':results_path, 
                'i': i,
                'num_of_repeats':num_of_repeats}
        with open('chunks/chunk{}.p'.format(i), 'wb') as file:
            pickle.dump(params, file)
    print('{} chunks pickled'.format(no_of_chunks))


    work_path = os.path.join(os.path.dirname(__file__), 'workerscript.py')
    work_path = 'workerscript.py'
    processes = []
    for i, chunk in enumerate(chunks):
        cmd_list = ['python3', work_path, 'chunks/chunk{}.p'.format(i)]
        p = Popen(cmd_list)
        processes.append(p)

    for p in processes:
        p.wait()

    # combine the results that each worker produces into one .csv file
    combine_results(results_path)

def combine_results(path):
    files = find_csv_filenames(path)
    r = []
    for f in files:
        filepath = os.path.join(path, f)
        r.append(pd.read_csv(filepath)) 
    results = pd.concat(r, ignore_index=True)
    path = os.path.join(path, 'results.csv')
    results.to_csv(path, index=False)

def find_csv_filenames(path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

