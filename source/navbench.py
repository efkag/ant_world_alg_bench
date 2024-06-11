import os
import copy
import pandas as pd
import time
import itertools
import multiprocessing
import functools
import numpy as np
import yaml
import uuid
import datetime
#TODO: Update to use the pick nav function 
from source.tools.benchutils import pick_nav
from source.navs import perfect_memory as pm
from source.utils import pre_process, load_route_naw, check_for_dir_and_create, calc_dists, squash_deg
#TODO Get rid of this import
from source.navs import seqnav as spm
from source.routedatabase import Route, load_all_bob_routes, load_routes, load_bob_routes, load_bob_routes_repeats
from source.imgproc import Pipeline
from source.navs import infomax


class Benchmark:
    def __init__(self, results_path, routes_path, grid_path=None,  filename='results.csv', 
                 route_path_suffix=None, grid_dist=None, route_repeats=None, bench_data=None):
        self.results_path = results_path
        self.routes_path = routes_path
        self.route_path_suffix = route_path_suffix
        self.route_repeats = route_repeats
        self.grid_path = grid_path
        check_for_dir_and_create(results_path)
        self.jobs = 0
        self.total_jobs = 1
        self.bench_logs = []
        self.route_ids = None
        self.routes_data = []
        #self.dist = 0.2  # Distance between grid images and route images
        self.dist = grid_dist
        self.bench_data = bench_data
        self.log = {'route_id': [], 'blur': [], 'edge': [], 'res': [], 'window': [],
                    'matcher': [], 'mean_error': [], 'errors': [], 'seconds': [],
                    'abs_index_diff': [], 'window_log': [], 'best_sims': [], 'dist_diff': [],
                    'tx': [], 'ty': [], 'th': []}

    def load_routes(self, route_ids):
        self.route_ids = route_ids
        for rid in self.route_ids:
            route_path = self.routes_path + '/route' + str(rid) + '/'
            route_data = load_route_naw(route_path, route_id=id, query=True,  max_dist=self.dist)
            self.routes_data.append(route_data)

    def get_grid_chunks(self, grid_gen, chunks=1):
        lst = list(grid_gen)
        return [lst[i::chunks] for i in range(chunks)]

    def remove_blur_edge(self, combo):
        # removes a combination if it has both blurring and edges
        return not (combo['edge_range'] and combo['blur'])

    def remove_non_blur_edge(self, combo):
        return not combo.get('edge_range') and not combo.get('blur') and not combo.get('gauss_loc_norm') and not combo.get('loc_norm')
        #return not combo['edge_range'] and not combo['blur']

    def remove_edge_loc_gloc_combos(self, combo):
        return not (combo.get('edge_range') and combo.get('loc_norm') 
                    and combo.get('gauss_loc_norm'))
    
    def remove_edge_gloc_combos(self, combo):
        return not (combo.get('edge_range') and combo.get('gauss_loc_norm'))  
    
    def remove_edge_loc_combos(self, combo):
        return not (combo.get('edge_range') and combo.get('loc_norm'))  

    def remove_gloc_loc_combos(self, combo):
        return not (combo.get('gauss_loc_norm') and combo.get('loc_norm')) 

    def get_grid_dict(self, params):
        grid = itertools.product(*[params[k] for k in params])

        grid = [*grid]
        grid_dict = []
        for combo in grid:
            combo_dict = {}
            for i, k in enumerate(params):
                combo_dict[k] = combo[i]
            grid_dict.append(combo_dict)

        if params.get('edge_range') or params.get('loc_norm') or params.get('gauss_loc_norm'):
            #grid_dict[:] = [x for x in grid_dict if self.remove_blur_edge(x)]
            #grid_dict[:] = [x for x in grid_dict if not self.remove_non_blur_edge(x)]
            #grid_dict[:] = [x for x in grid_dict if self.remove_edge_loc_gloc_combos(x)]
            #grid_dict[:] = [x for x in grid_dict if self.remove_edge_gloc_combos(x)]
            #grid_dict[:] = [x for x in grid_dict if self.remove_edge_loc_combos(x)]
            grid_dict[:] = [x for x in grid_dict if self.remove_gloc_loc_combos(x)]
        return grid_dict

    @staticmethod
    def _init_shared():
        manager = multiprocessing.Manager()
        # shared = manager.list([0, 0])
        shared = manager.dict({'jobs': 0, 'total_jobs': 0})
        return shared

    def bench_paral(self, params, route_ids=None, cores=None):
        # save the parmeters of the test in a json file
        check_for_dir_and_create(self.results_path)
        check_for_dir_and_create(os.path.join(self.results_path, 'metadata'))
        param_path = os.path.join(self.results_path, 'params.yml')
        temp_params = copy.deepcopy(params)
        temp_params['routes_path'] = self.routes_path
        temp_params['route_ids'] = route_ids
        with open(param_path, 'w') as fp:
            yaml.dump(temp_params, fp)

        existing_cores = multiprocessing.cpu_count()
        if cores and cores > existing_cores:
            cores = existing_cores - 1
        elif cores and cores <= existing_cores:
            cores = cores
        else:
            cores = existing_cores - 1
        print(existing_cores, ' CPU cores found. Using ', cores, ' cores')

        grid = self.get_grid_dict(params)
        shared = self._init_shared()
        self.total_jobs = len(grid)
        shared['total_jobs'] = self.total_jobs * len(route_ids)

        if self.total_jobs < cores:
            no_of_chunks = self.total_jobs
        else:
            no_of_chunks = cores
        # Generate list chunks of grid combinations
        chunks = self.get_grid_chunks(grid, no_of_chunks)

        print('{} combinations, testing on {} routes, running on {} cores'.format(self.total_jobs, len(route_ids), no_of_chunks))
        print('data path: ', self.routes_path)
        print('save path: ', self.results_path)
        #Make a dict of parameter arguments
        arg_params = {'route_ids': route_ids, 'grid_dist':self.dist,
                      'routes_path':self.routes_path, 'grid_path':self.grid_path,
                      'route_path_suffix':self.route_path_suffix,
                      'repeats':self.route_repeats, 'results_path':self.results_path}
        # Partial callable

        if self.bench_data == 'bob':
            worker = functools.partial(self.worker_bench_repeats, arg_params, shared)
        elif self.bench_data == 'aw2':
            worker = functools.partial(self.worker_bench, arg_params, shared)
        else:
            raise Exception("Provide database type")

        pool = multiprocessing.Pool(processes=no_of_chunks)

        logs = pool.map_async(worker, chunks)
        pool.close()
        pool.join()

        return logs
    
    def bench_singe_core_aw(self, params, route_ids=None):
        # save the parmeters of the test in a json file
        check_for_dir_and_create(self.results_path)
        check_for_dir_and_create(os.path.join(self.results_path, 'metadata'))
        param_path = os.path.join(self.results_path, 'params.yml')
        temp_params = copy.deepcopy(params)
        temp_params['routes_path'] = self.routes_path
        temp_params['route_ids'] = route_ids
        with open(param_path, 'w') as fp:
            yaml.dump(temp_params, fp)

        grid = self.get_grid_dict(params)
        self.total_jobs = len(grid)
        self.total_jobs = self.total_jobs * len(route_ids)
        print('{} combinations, testing on {} routes, running on 1 core'.format(self.total_jobs, len(route_ids)))
        print('data path: ', self.routes_path)
        print('save path: ', self.results_path)

        log = {'route_id': [],'nav-name':[], 'blur': [], 'edge': [], 'res': [],  'histeq':[], 'vcrop':[], 
               'window': [], 'matcher': [], 'deg_range':[], 'mean_error': [], 
               'seconds': [], 'errors': [], 'index_diff': [], 'window_log': [], 
               'matched_index': [], 'min_dist_index': [], 'dist_diff': [], 'tx': [], 'ty': [], 'th': [],
               'ah': [] , 'rmfs_file':[],'best_sims':[], 'loc_norm':[], 
               'gauss_loc_norm':[], 'wave':[]}
        
        routes = load_routes(self.routes_path, route_ids, max_dist=self.dist, grid_path=self.grid_path)
        #  Go though all combinations in the chunk
        for combo in grid:

            matcher = combo['matcher']
            window = combo['window']
            window_log = None
            for route in routes:  # for every route
                tic = time.perf_counter()
                # Preprocess images
                pipe = Pipeline(**combo)
                route_imgs = pipe.apply(route.get_imgs())
                test_imgs = pipe.apply(route.get_qimgs())
                # Run navigation algorithm
                if window:
                    nav = spm.SequentialPerfectMemory(route_imgs, **combo)
                    recovered_heading, window_log = nav.navigate(test_imgs)
                elif window == 0:
                    nav = pm.PerfectMemory(route_imgs, **combo)
                    recovered_heading = nav.navigate(test_imgs)
                else:
                    infomaxParams = infomax.Params()
                    nav = infomax.InfomaxNetwork(infomaxParams, route_imgs, **combo)
                    recovered_heading = nav.navigate(test_imgs)

                toc = time.perf_counter()
                # Get time complexity
                time_compl = toc - tic
                # Get the errors and the minimum distant index of the route memory
                qxy = route.get_qxycoords()
                traj = {'x': qxy['x'], 'y': qxy['y'], 'heading': recovered_heading}
                #!!!!!! Important step to get the heading in the global coord system
                traj['heading'] = squash_deg(route.get_qyaw() + recovered_heading)
                errors, min_dist_index = route.calc_aae(traj)
                # Difference between matched index and minimum distance index and distance between points
                matched_index = nav.get_index_log()
                if matched_index:
                    index_diffs = np.subtract(min_dist_index, nav.get_index_log())
                    dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index)
                    index_diffs = index_diffs.tolist()
                    dist_diff = dist_diff.tolist()
                else:
                    index_diffs = None
                    dist_diff = None

                mean_route_error = np.mean(errors)
                window_log = nav.get_window_log()
                rec_headings = nav.get_rec_headings()
                deg_range = nav.deg_range

                rmf_logs = np.array(nav.get_rsims_log(), dtype=object)
                rmf_logs_file = f"rmfs-{uuid.uuid4().hex}"
                rmfs_path = os.path.join(self.results_path, 'metadata', rmf_logs_file)
                np.save(rmfs_path, rmf_logs)

                log['nav-name'].append(nav.get_name())
                log['route_id'].append(route.get_route_id())
                log['blur'].append(combo.get('blur'))
                log['histeq'].append(combo.get('histeq'))
                log['edge'].append(combo.get('edge_range'))
                log['res'].append(combo.get('shape'))
                log['vcrop'].append(combo.get('vcrop'))
                log['window'].append(window)
                log['loc_norm'].append(combo.get('loc_norm'))
                log['gauss_loc_norm'].append(combo.get('gauss_loc_norm'))
                log['wave'].append(combo.get('wave'))
                log['matcher'].append(matcher)
                log['deg_range'].append(deg_range)
                log['mean_error'].append(mean_route_error)
                log['seconds'].append(time_compl)
                log['window_log'].append(window_log)
                log['rmfs_file'].append(rmf_logs_file)
                log['tx'].append(traj['x'].tolist())
                log['ty'].append(traj['y'].tolist())
                # This is the heading in the global coord system
                log['th'].append(traj['heading'].tolist())
                # This is the agent heading from the egocentric agent reference
                log['ah'].append(rec_headings)
                log['matched_index'].append(matched_index)
                log['min_dist_index'].append(min_dist_index)
                log['index_diff'].append(index_diffs)
                log['dist_diff'].append(dist_diff)
                log['errors'].append(errors)
                log['best_sims'].append(nav.get_best_sims())
                self.jobs += 1
                print(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
                print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))
        return log
    
    def  bench_singe_core(self, params, route_ids=None):
        # save the parmeters of the test in a json file
        check_for_dir_and_create(self.results_path)
        check_for_dir_and_create(os.path.join(self.results_path, 'metadata'))
        param_path = os.path.join(self.results_path, 'params.yml')
        temp_params = copy.deepcopy(params)
        temp_params['routes_path'] = self.routes_path
        temp_params['route_ids'] = route_ids
        with open(param_path, 'w') as fp:
            yaml.dump(temp_params, fp)

        grid = self.get_grid_dict(params)
        self.total_jobs = len(grid)
        self.total_jobs = self.total_jobs * len(route_ids)
        print('{} combinations, testing on {} routes, running on 1 core'.format(self.total_jobs, len(route_ids)))
        print('data path: ', self.routes_path)
        print('save path: ', self.results_path)

        log = {'route_id': [],'ref_route':[], 'rep_id': [], 'nav-name':[], 'sample_rate':[], 'blur': [], 
            'histeq':[], 'edge': [], 'res': [], 'vcrop':[], 'loc_norm':[], 'gauss_loc_norm':[],
            'window': [], 'matcher': [], 'deg_range':[], 'mean_error': [], 'seconds': [], 'errors': [], 
            'index_diff': [], 'window_log': [], 'matched_index': [], 'min_dist_index': [],
            'dist_diff': [], 'tx': [], 'ty': [], 'th': [],'ah': [] , 
            'rmfs_file':[], 'best_ridfs_file': [],
            'best_sims':[], 'wave':[]}

        routes = load_all_bob_routes(self.routes_path, route_ids, suffix=self.route_path_suffix, repeats=self.route_repeats)

        #  Go though all combinations in the grid
        for combo in grid:

            matcher = combo.get('matcher')
            window = combo.get('window')
            window_log = None

            for ri, route in enumerate(routes):  # for every route
                #print('ref route -> ', combo['ref_route'])
                ref_rep = route[combo['ref_route']]
                ref_rep.set_sample_step(combo['sample_step'])
                repeat_ids = [*range(1, self.route_repeats+1)]
                #print(repeat_ids)
                repeat_ids.remove(combo['ref_route'])
                #print(repeat_ids)
                for rep_id in repeat_ids: # for every repeat route
                    test_rep = route[rep_id]
                    test_rep.set_sample_step(combo['sample_step'])
                    # Preprocess images
                    pipe = Pipeline(**combo)
                    route_imgs = pipe.apply(ref_rep.get_imgs())
                    test_imgs = pipe.apply(test_rep.get_imgs())
                
                    # Preprocess images
                    pipe = Pipeline(**combo)
                    route_imgs = pipe.apply(ref_rep.get_imgs())
                    test_imgs = pipe.apply(test_rep.get_imgs())
                    
                    if window:
                        nav = spm.SequentialPerfectMemory(route_imgs, **combo)
                        recovered_heading, window_log = nav.navigate(test_imgs)
                    elif window == 0:
                        nav = pm.PerfectMemory(route_imgs, **combo)
                        recovered_heading = nav.navigate(test_imgs)
                    else:
                        infomaxParams = infomax.Params()
                        nav = infomax.InfomaxNetwork(route_imgs, infomaxParams, **combo)
                        recovered_heading = nav.navigate(test_imgs)

                    time_compl = nav.get_time_com()
                    # Get the errors and the minimum distant index of the route memory
                    qxy = test_rep.get_xycoords()
                    traj = {'x': qxy['x'], 'y': qxy['y'], 'heading': recovered_heading}

                    #################!!!!!! Important step to get the heading in the global coord system
                    traj['heading'] = squash_deg(test_rep.get_yaw() + recovered_heading)
                    errors, min_dist_index = ref_rep.calc_aae(traj)
                    # Difference between matched index and minimum distance index and distance between points
                    matched_index = nav.get_index_log()
                    if matched_index:
                        index_diffs = np.subtract(min_dist_index, nav.get_index_log())
                        dist_diff = calc_dists(ref_rep.get_xycoords(), min_dist_index, matched_index)
                        index_diffs = index_diffs.tolist()
                        dist_diff = dist_diff.tolist()
                    else:
                        index_diffs = None
                        dist_diff = None
                    mean_route_error = np.mean(errors)
                    window_log = nav.get_window_log()
                    rec_headings = nav.get_rec_headings()
                    deg_range = nav.deg_range

                    #TODO check if we have empty logs from PM and omit saving those arrays
                    # check if empty arrays cause saving problems.
                    #rmf_logs = np.array(nav.get_rsims_log(), dtype=object)
                    rmf_logs_file = f"rmfs-{self.jobs}_{uuid.uuid4().hex}"
                    #rmfs_path = os.path.join(results_path, 'metadata', rmf_logs_file)
                    #np.save(rmfs_path, rmf_logs)

                    best_ridfs = nav.get_best_ridfs()
                    best_ridfs = np.array(best_ridfs)
                    ridfs_file = f"ridfs-{self.jobs}_{uuid.uuid4().hex}"
                    ridfs_path = os.path.join(self.results_path, 'metadata', ridfs_file)
                    np.save(ridfs_path, best_ridfs)

                    log['nav-name'].append(nav.get_name())
                    log['route_id'].append(ri)
                    log['ref_route'].append(combo.get('ref_route'))
                    log['rep_id'].append(test_rep.get_route_id())
                    log['sample_rate'].append(combo.get('sample_step'))
                    log['blur'].append(combo.get('blur'))
                    log['histeq'].append(combo.get('histeq'))
                    log['edge'].append(combo.get('edge_range'))
                    log['res'].append(combo.get('shape'))
                    log['vcrop'].append(combo.get('vcrop'))
                    log['window'].append(window)
                    log['loc_norm'].append(combo.get('loc_norm'))
                    log['gauss_loc_norm'].append(combo.get('gauss_loc_norm'))
                    log['wave'].append(combo.get('wave'))
                    log['matcher'].append(matcher)
                    log['deg_range'].append(deg_range)
                    log['mean_error'].append(mean_route_error)
                    log['seconds'].append(time_compl)
                    log['window_log'].append(window_log)
                    log['tx'].append(traj['x'].tolist())
                    log['ty'].append(traj['y'].tolist())
                    # This is the heading in the global coord system
                    log['th'].append(traj['heading'].tolist())
                    # This is the agent heading from the egocentric agent reference
                    log['ah'].append(rec_headings)
                    log['rmfs_file'].append(rmf_logs_file)
                    log['best_ridfs_file'].append(ridfs_file)
                    log['matched_index'].append(matched_index)
                    log['min_dist_index'].append(min_dist_index)
                    log['index_diff'].append(index_diffs)
                    log['dist_diff'].append(dist_diff)
                    log['errors'].append(errors)
                    log['best_sims'].append(nav.get_best_sims())
            self.jobs += 1
            print(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
            print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))
        return log

    def _total_jobs(self, params):
        for k in params:
            self.total_jobs = self.total_jobs * len(params[k])
        print('Total number of jobs: {}'.format(self.total_jobs))

    def benchmark(self, params, route_ids, parallel=True, cores=None):
        print(f'testing params: {params}')
        assert isinstance(params, dict)
        assert isinstance(route_ids, list)

        if parallel:
            self.log = None
            self.log = self.bench_paral(params, route_ids, cores=cores)
            self.unpack_results()
        else:
            self.log = self.bench_singe_core(params, route_ids)

        bench_results = pd.DataFrame(self.log)
        write_path = os.path.join(self.results_path, 'results.csv')
        bench_results.to_csv(write_path, index=False)
        #print(bench_results)

    def unpack_results(self):
        results = self.log.get()
        print(len(results), 'Results produced')
        self.log = results[0]
        for dictionary in results[1:]:
            for k in dictionary:
                self.log[k].extend(dictionary[k])


    @staticmethod
    def worker_bench(arg_params, shared, chunk):
        # unpack shared bench parameters
        route_ids = arg_params.get('route_ids')
        dist =  arg_params.get('grid_dist')
        routes_path =  arg_params.get('routes_path')
        grid_path =  arg_params.get('grid_path')
        results_path = arg_params.get('results_path')
        chunk_id = multiprocessing.current_process()._identity

        log = {'route_id': [],'nav-name':[], 'blur': [], 'edge': [], 'res': [],  'histeq':[], 'vcrop':[], 
               'window': [], 'matcher': [], 'deg_range':[], 'mean_error': [], 
               'seconds': [], 'errors': [], 'index_diff': [], 'window_log': [], 
               'matched_index': [], 'min_dist_index': [], 'dist_diff': [], 'tx': [], 'ty': [], 'th': [],
               'ah': [] , 'rmfs_file':[],'best_sims':[], 'loc_norm':[], 
               'gauss_loc_norm':[], 'wave':[]}
        
        # Load all routes
        routes = load_routes(routes_path, route_ids, max_dist=dist, grid_path=grid_path)

        #  Go though all combinations in the chunk
        for combo in chunk:

            matcher = combo['matcher']
            window = combo['window']
            window_log = None
            for route in routes:  # for every route

                tic = time.perf_counter()
                # Preprocess images
                pipe = Pipeline(**combo)
                route_imgs = pipe.apply(route.get_imgs())
                test_imgs = pipe.apply(route.get_qimgs())
                # Run navigation algorithm
                if window:
                    nav = spm.SequentialPerfectMemory(route_imgs, **combo)
                    recovered_heading, window_log = nav.navigate(test_imgs)
                elif window == 0:
                    nav = pm.PerfectMemory(route_imgs, **combo)
                    recovered_heading = nav.navigate(test_imgs)
                else:
                    infomaxParams = infomax.Params()
                    nav = infomax.InfomaxNetwork(infomaxParams, route_imgs, **combo)
                    recovered_heading = nav.navigate(test_imgs)

                toc = time.perf_counter()
                # Get time complexity
                time_compl = toc - tic
                # Get the errors and the minimum distant index of the route memory
                qxy = route.get_qxycoords()
                traj = {'x': qxy['x'], 'y': qxy['y'], 'heading': recovered_heading}
                #!!!!!! Important step to get the heading in the global coord system
                traj['heading'] = squash_deg(route.get_qyaw() + recovered_heading)
                errors, min_dist_index = route.calc_aae(traj)
                # Difference between matched index and minimum distance index and distance between points
                matched_index = nav.get_index_log()
                if matched_index:
                    index_diffs = np.subtract(min_dist_index, nav.get_index_log())
                    dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index)
                    index_diffs = index_diffs.tolist()
                    dist_diff = dist_diff.tolist()
                else:
                    index_diffs = None
                    dist_diff = None
                
                mean_route_error = np.mean(errors)
                window_log = nav.get_window_log()
                rec_headings = nav.get_rec_headings()
                deg_range = nav.deg_range

                rmf_logs = np.array(nav.get_rsims_log(), dtype=object)
                rmf_logs_file = f"rmfs-{chunk_id}{shared['jobs']}_{uuid.uuid4().hex}"
                rmfs_path = os.path.join(results_path, 'metadata', rmf_logs_file)
                np.save(rmfs_path, rmf_logs)

                log['nav-name'].append(nav.get_name())
                log['route_id'].append(route.get_route_id())
                log['blur'].append(combo.get('blur'))
                log['histeq'].append(combo.get('histeq'))
                log['edge'].append(combo.get('edge_range'))
                log['res'].append(combo.get('shape'))
                log['vcrop'].append(combo.get('vcrop'))
                log['window'].append(window)
                log['loc_norm'].append(combo.get('loc_norm'))
                log['gauss_loc_norm'].append(combo.get('gauss_loc_norm'))
                log['wave'].append(combo.get('wave'))
                log['matcher'].append(matcher)
                log['deg_range'].append(deg_range)
                log['mean_error'].append(mean_route_error)
                log['seconds'].append(time_compl)
                log['window_log'].append(window_log)
                log['rmfs_file'].append(rmf_logs_file)
                log['tx'].append(traj['x'].tolist())
                log['ty'].append(traj['y'].tolist())
                # This is the heading in the global coord system
                log['th'].append(traj['heading'].tolist())
                # This is the agent heading from the egocentric agent reference
                log['ah'].append(rec_headings)
                log['matched_index'].append(matched_index)
                log['min_dist_index'].append(min_dist_index)
                log['index_diff'].append(index_diffs)
                log['dist_diff'].append(dist_diff)
                log['errors'].append(errors)
                log['best_sims'].append(nav.get_best_sims())
                # Increment the complete jobs shared variable
                shared['jobs'] = shared['jobs'] + 1
                print(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"),
                      multiprocessing.current_process(), 
                      ' jobs completed: {}/{}'.format(shared['jobs'], shared['total_jobs']))
        return log
    
    @staticmethod
    def worker_bench_repeats(arg_params, shared, chunk):
        # unpack shared bench parameters
        route_ids = arg_params.get('route_ids')
        dist =  arg_params.get('grid_dist')
        routes_path =  arg_params.get('routes_path')
        grid_path =  arg_params.get('grid_path')
        results_path = arg_params.get('results_path')
        route_path_suffix = arg_params.get('route_path_suffix')
        repeats = arg_params.get('repeats')
        chunk_id = multiprocessing.current_process()._identity

        log = {'route_id': [],'ref_route':[], 'rep_id': [], 'nav-name':[], 'sample_rate':[], 'blur': [], 
               'histeq':[], 'edge': [], 'res': [], 'vcrop':[],'window': [], 'matcher': [],
               'deg_range':[], 'mean_error': [], 'seconds': [], 'errors': [], 
               'index_diff': [], 'window_log': [], 'matched_index': [], 'min_dist_index': [],
               'dist_diff': [], 'tx': [], 'ty': [], 'th': [],'ah': [] , 'rmfs_file':[], 'best_sims':[], 
               'loc_norm':[], 'gauss_loc_norm':[], 'wave':[]}

        
        # Load all routes
        routes = load_all_bob_routes(routes_path, route_ids, suffix=route_path_suffix, repeats=repeats)
        # routes = load_routes(routes_path, route_ids, max_dist=dist, grid_path=grid_path)
        #print('routes ->', routes)
        #print('routes dict -> ', routes[0])
        #  Go though all combinations in the chunk
        for combo in chunk:

            matcher = combo.get('matcher')
            window = combo.get('window')
            window_log = None
            for ri, route in enumerate(routes):  # for every route
                #print('ref route -> ', combo['ref_route'])
                ref_rep = route[combo['ref_route']]
                ref_rep.set_sample_step(combo['sample_step'])
                repeat_ids = [*range(1, repeats+1)]
                #print(repeat_ids)
                repeat_ids.remove(combo['ref_route'])
                #print(repeat_ids)

                for rep_id in repeat_ids: # for every repeat route
                    test_rep = route[rep_id]
                    test_rep.set_sample_step(combo['sample_step'])
                    # print(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"),
                    #       ' testting: ', combo)
                    tic = time.perf_counter()
                    # Preprocess images
                    pipe = Pipeline(**combo)
                    route_imgs = pipe.apply(ref_rep.get_imgs())
                    test_imgs = pipe.apply(test_rep.get_imgs())
                    # Run navigation algorithm
                    if window:
                        nav = spm.SequentialPerfectMemory(route_imgs, **combo)
                        recovered_heading, window_log = nav.navigate(test_imgs)
                    elif window == 0:
                        nav = pm.PerfectMemory(route_imgs, **combo)
                        recovered_heading = nav.navigate(test_imgs)
                    else:
                        infomaxParams = infomax.Params()
                        nav = infomax.InfomaxNetwork(infomaxParams, route_imgs, **combo)
                        recovered_heading = nav.navigate(test_imgs)
                    # here i need a navigate method for infomax.
                    toc = time.perf_counter()
                    # Get time complexity
                    time_compl = toc - tic
                    # Get the errors and the minimum distant index of the route memory
                    qxy = test_rep.get_xycoords()
                    traj = {'x': qxy['x'], 'y': qxy['y'], 'heading': recovered_heading}

                    #################!!!!!! Important step to get the heading in the global coord system
                    traj['heading'] = squash_deg(test_rep.get_yaw() + recovered_heading)
                    errors, min_dist_index = ref_rep.calc_aae(traj)
                    # Difference between matched index and minimum distance index and distance between points
                    matched_index = nav.get_index_log()
                    if matched_index:
                        index_diffs = np.subtract(min_dist_index, nav.get_index_log())
                        dist_diff = calc_dists(ref_rep.get_xycoords(), min_dist_index, matched_index)
                        index_diffs = index_diffs.tolist()
                        dist_diff = dist_diff.tolist()
                    else:
                        index_diffs = None
                        dist_diff = None
                    mean_route_error = np.mean(errors)
                    window_log = nav.get_window_log()
                    rec_headings = nav.get_rec_headings()
                    deg_range = nav.deg_range
                    
                    #TODO check if we have empty logs from PM and omit saving those arrays
                    # check if empty arrays cause saving problems.
                    #rmf_logs = np.array(nav.get_rsims_log(), dtype=object)
                    rmf_logs_file = f"rmfs-{chunk_id}{shared['jobs']}_{uuid.uuid4().hex}"
                    #rmfs_path = os.path.join(results_path, 'metadata', rmf_logs_file)
                    #np.save(rmfs_path, rmf_logs)

                    log['nav-name'].append(nav.get_name())
                    log['route_id'].append(ri)
                    log['ref_route'].append(combo.get('ref_route'))
                    log['rep_id'].append(test_rep.get_route_id())
                    log['sample_rate'].append(combo.get('sample_step'))
                    log['blur'].append(combo.get('blur'))
                    log['histeq'].append(combo.get('histeq'))
                    log['edge'].append(combo.get('edge_range'))
                    log['res'].append(combo.get('shape'))
                    log['vcrop'].append(combo.get('vcrop'))
                    log['window'].append(window)
                    log['loc_norm'].append(combo.get('loc_norm'))
                    log['gauss_loc_norm'].append(combo.get('gauss_loc_norm'))
                    log['wave'].append(combo.get('wave'))
                    log['matcher'].append(matcher)
                    log['deg_range'].append(deg_range)
                    log['mean_error'].append(mean_route_error)
                    log['seconds'].append(time_compl)
                    log['window_log'].append(window_log)
                    log['tx'].append(traj['x'].tolist())
                    log['ty'].append(traj['y'].tolist())
                    # This is the heading in the global coord system
                    log['th'].append(traj['heading'].tolist())
                    # This is the agent heading from the egocentric agent reference
                    log['ah'].append(rec_headings)
                    log['rmfs_file'].append(rmf_logs_file)
                    log['matched_index'].append(matched_index)
                    log['min_dist_index'].append(min_dist_index)
                    log['index_diff'].append(index_diffs)
                    log['dist_diff'].append(dist_diff)
                    log['errors'].append(errors)
                    log['best_sims'].append(nav.get_best_sims())
                # Increment the complete jobs shared variable
                shared['jobs'] = shared['jobs'] + 1
                print(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
                print(multiprocessing.current_process(), ' jobs completed: {}/{}'.format(shared['jobs'], shared['total_jobs']))
        return log
