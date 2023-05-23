from source.utils import pre_process, load_route_naw, seq_angular_error, check_for_dir_and_create, calc_dists
from source import seqnav as spm, perfect_memory as pm
import os
import pandas as pd
import time
import itertools
import multiprocessing
import functools
import numpy as np
import yaml
from source.routedatabase import Route, load_routes
from source.imgproc import Pipeline
from source import infomax


class Benchmark:
    def __init__(self, results_path, routes_path, grid_path=None,  filename='results.csv'):
        self.results_path = results_path + filename
        self.routes_path = routes_path
        self.grid_path = grid_path
        check_for_dir_and_create(results_path)
        self.jobs = 0
        self.total_jobs = 1
        self.bench_logs = []
        self.route_ids = None
        self.routes_data = []
        self.dist = 0.2  # Distance between grid images and route images
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
        return not (combo['edge_range'] and combo['blur'])

    def remove_non_blur_edge(self, combo):
        return not combo.get('edge_range') and not combo.get('blur') and not combo.get('gauss_loc_norm') and not combo.get('loc_norm')
        #return not combo['edge_range'] and not combo['blur']

    def get_grid_dict(self, params):
        grid = itertools.product(*[params[k] for k in params])

        grid = [*grid]
        grid_dict = []
        for combo in grid:
            combo_dict = {}
            for i, k in enumerate(params):
                combo_dict[k] = combo[i]
            grid_dict.append(combo_dict)

        grid_dict[:] = [x for x in grid_dict if self.remove_blur_edge(x)]
        grid_dict[:] = [x for x in grid_dict if not self.remove_non_blur_edge(x)]
        return grid_dict

    @staticmethod
    def _init_shared():
        manager = multiprocessing.Manager()
        # shared = manager.list([0, 0])
        shared = manager.dict({'jobs': 0, 'total_jobs': 0})
        return shared

    def bench_paral(self, results_path,  params, route_ids=None, cores=None):
        # save the parmeters of the test in a json file
        check_for_dir_and_create(results_path)
        param_path = os.path.join(results_path, 'params.yml')
        with open(param_path, 'w') as fp:
            yaml.dump(params, fp)

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
        # Partial callable
        worker = functools.partial(self.worker_bench, route_ids,
                                   self.dist, self.routes_path, self.grid_path, shared)

        pool = multiprocessing.Pool()

        logs = pool.map_async(worker, chunks)
        pool.close()
        pool.join()

        return logs

    def bench_singe_core(self, params, route_ids=None):
        self._total_jobs(params)

        # Get list of parameter keys
        keys = [*params]

        grid = itertools.product(*[params[k] for k in params])

        #  Go though all combinations in the grid
        for combo in grid:

            # create combo dictionary
            combo_dict = {}
            for i, k in enumerate(keys):
                combo_dict[k] = combo[i]

            matcher = combo_dict['matcher']
            window = combo_dict['window']
            window_log = None
            path = '../new-antworld/'
            for route_id in route_ids:  # for every route
                # TODO: In the future the code below (inside the loop) should all be moved inside the navigator class
                route_path = '../new-antworld/route' + str(route_id) + '/'
                # route = load_route_naw(route_path, route_id=route_id, imgs=True, query=True, max_dist=0.2)
                # _, test_x, test_y, test_imgs, route_x, route_y, \
                #     route_heading, route_imgs = load_route(route, self.dist)
                route = Route(route_path, route_id, grid_path=self.grid_path)
                
                # TODO: here the query images need to be the othe repeat of the route!
                tic = time.perf_counter()
                # Preprocess images
                test_imgs = pre_process(route.get_qimgs(), combo_dict)
                route_imgs = pre_process(route.get_imgs(), combo_dict)
                # Run navigation algorithm
                if window:
                    nav = spm.SequentialPerfectMemory(route_imgs, matcher, window=window)
                    recovered_heading, window_log = nav.navigate(test_imgs)
                else:
                    nav = pm.PerfectMemory(route_imgs, matcher)
                    recovered_heading = nav.navigate(test_imgs)

                toc = time.perf_counter()
                # Get time complexity
                time_compl = toc-tic
                # Get the errors and the minimum distant index of the route memory
                # errors, min_dist_index = degree_error(test_x, test_y, route_x, route_y, route_heading, recovered_heading)
                traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
                errors, min_dist_index = route.calc_errors(traj)
                # Difference between matched index and minimum distance index and distance between points
                matched_index = nav.get_index_log()
                abs_index_diffs = np.absolute(np.subtract(nav.get_index_log(), min_dist_index))
                dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index)
                mean_route_error = np.mean(errors)
                self.log['route_id'].extend([route_id])
                self.log['blur'].extend([combo_dict.get('blur')])
                self.log['edge'].extend([combo_dict.get('edge_range')])
                self.log['res'].append(combo_dict.get('shape'))
                self.log['window'].extend([window])
                self.log['matcher'].extend([matcher])
                self.log['mean_error'].append(mean_route_error)
                self.log['seconds'].append(time_compl)
                self.log['window_log'].append(window_log)
                self.log['best_sims'].append(nav.get_best_sims())
                self.log['tx'].append(traj['x'].tolist())
                self.log['ty'].append(traj['y'].tolist())
                self.log['th'].append(traj['heading'])
                self.log['abs_index_diff'].append(abs_index_diffs.tolist())
                self.log['dist_diff'].append(dist_diff.tolist())
                self.log['errors'].append(errors)
            self.jobs += 1
            print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))
        return self.log

    def _total_jobs(self, params):
        for k in params:
            self.total_jobs = self.total_jobs * len(params[k])
        print('Total number of jobs: {}'.format(self.total_jobs))

    def benchmark(self, params, route_ids, parallel=True, cores=None):

        assert isinstance(params, dict)
        assert isinstance(route_ids, list)

        if parallel:
            self.log = None
            self.log = self.bench_paral(params, route_ids, cores=cores)
            self.unpack_results()
        else:
            self.log = self.bench_singe_core(params, route_ids)

        bench_results = pd.DataFrame(self.log)
        bench_results.to_csv(self.results_path, index=False)
        print(bench_results)

    def unpack_results(self):
        results = self.log.get()
        print(len(results), 'Results produced')
        self.log = results[0]
        for dictionary in results[1:]:
            for k in dictionary:
                self.log[k].extend(dictionary[k])


    @staticmethod
    def worker_bench(route_ids, dist, routes_path, grid_path, shared, chunk):

        log = {'route_id': [], 'blur': [], 'edge': [], 'res': [], 'window': [], 'matcher': [],
             'deg_range':[], 'mean_error': [], 'seconds': [], 'errors': [], 
             'abs_index_diff': [], 'window_log': [], 'matched_index': [], 'dist_diff': [], 
             'tx': [], 'ty': [], 'th': [],'ah': [] ,'best_sims':[], 
             'loc_norm':[], 'gauss_loc_norm':[], 'wave':[], 'nav-name':[]}
        
        # Load all routes
        routes = load_routes(routes_path, route_ids, max_dist=dist, grid_path=grid_path)
        #TODO: here i need to make the query images and data for each route.
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
                    nav = spm.SequentialPerfectMemory(route_imgs, matcher, window=window, **combo)
                    recovered_heading, window_log = nav.navigate(test_imgs)
                elif window == 0:
                    nav = pm.PerfectMemory(route_imgs, matcher, **combo)
                    recovered_heading = nav.navigate(test_imgs)
                # else:
                #     infomaxParams = infomax.Params()
                #     nav = infomax.InfomaxNetwork(infomaxParams, route_imgs, deg_range=(-180, 180), **combo)
                # here i need a navigate method for infomax.
                toc = time.perf_counter()
                # Get time complexity
                time_compl = toc - tic
                # Get the errors and the minimum distant index of the route memory
                traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
                errors, min_dist_index = route.calc_errors(traj)
                # Difference between matched index and minimum distance index and distance between points
                matched_index = nav.get_index_log()
                abs_index_diffs = np.absolute(np.subtract(nav.get_index_log(), min_dist_index))
                dist_diff = calc_dists(route.get_xycoords(), min_dist_index, matched_index)
                mean_route_error = np.mean(errors)
                window_log = nav.get_window_log()
                rec_headings = nav.get_rec_headings()
                deg_range = nav.deg_range

                log['nav-name'].append(nav.get_name())
                log['route_id'].append(route.get_route_id())
                log['blur'].append(combo.get('blur'))
                log['edge'].append(combo.get('edge_range'))
                log['res'].append(combo.get('shape'))
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
                log['th'].append(traj['heading'])
                log['ah'].append(recovered_heading)
                log['matched_index'].append(matched_index)
                log['abs_index_diff'].append(abs_index_diffs.tolist())
                log['dist_diff'].append(dist_diff.tolist())
                log['errors'].append(errors)
                log['best_sims'].append(nav.get_best_sims())
                # Increment the complete jobs shared variable
                shared['jobs'] = shared['jobs'] + 1
                print(multiprocessing.current_process(), ' jobs completed: {}/{}'.format(shared['jobs'], shared['total_jobs']))
        return log
