from source.utils import pre_process, load_route, degree_error
from source import seqnav as spm, perfect_memory as pm
import pandas as pd
import timeit
import itertools
import multiprocessing
import functools

import cv2 as cv
import numpy as np

class Benchmark:
    def __init__(self, results_path):
        self.results_path = results_path
        self.jobs = 0
        self.total_jobs = 1
        self.bench_logs = []
        self.route_ids = None
        self.routes_data = []
        self.dist = 100  # Distance between grid images and route images
        self.log = {'tested routes': [], 'blur': [], 'edge':[], 'window': [],
                    'matcher': [], 'mean error': [], 'errors': [], 'seconds': [],
                    'abs index diff': []}

    def load_routes(self, route_ids):
        self.route_ids = route_ids
        for id in self.route_ids:
            route_data = load_route(id, self.dist)
            self.routes_data.append(route_data)

    def get_grid_chunks(self, grid_gen, chunks=1):
        lst = list(grid_gen)
        return [lst[i::chunks] for i in range(chunks)]

    @staticmethod
    def _init_shared():
        manager = multiprocessing.Manager()
        shared = manager.list([0, 0])
        return shared

    def bench_paral(self, params, route_ids=None):
        print(multiprocessing.cpu_count(), ' CPU cores found')
        self._total_jobs(params)

        # Get list of parameter keys
        keys = [*params]

        shared = self._init_shared()
        shared[1] = self.total_jobs

        # Generate grid iterable
        grid = itertools.product(*[params[k] for k in params])
        # Generate list chunks of grid combinations
        chunks =  self.get_grid_chunks(grid, multiprocessing.cpu_count() - 1)
        # Partial callable
        worker = functools.partial(self.worker_bench, keys, route_ids, self.dist, shared)

        pool = multiprocessing.Pool()

        logs = pool.map_async(worker, chunks)
        pool.close()
        pool.join()

        return logs

    def bench_singe_core(self, params, route_ids=None):

        # Get list of parameter keys
        keys = [*params]

        grid = itertools.product(*[params[k] for k in params])

        no_of_routes = len(route_ids)
        #  Go though all combinations in the grid
        for combo in grid:

            # create combo dictionary
            combo_dict = {}
            for i, k in enumerate(keys):
                combo_dict[k] = combo[i]

            route_errors = []
            time_compl = []
            abs_index_diffs = []

            matcher = combo_dict['matcher']
            window = combo_dict['window']
            for route in route_ids:  # for every route
                _, test_x, test_y, test_imgs, route_x, route_y, \
                    route_heading, route_imgs = load_route(route, self.dist)
                tic = timeit.default_timer()
                # Preprocess images
                test_imgs = pre_process(test_imgs, combo_dict)
                route_imgs = pre_process(route_imgs, combo_dict)
                # Run navigation algorithm
                nav = spm.SequentialPerfectMemory(route_imgs, matcher)
                recovered_heading, logs, window_log = nav.navigate(test_imgs, window)
                toc = timeit.default_timer()
                # Get time complexity
                time_compl.append(toc-tic)
                # Get the errors and the minimum distant index of the route memory
                errors, min_dist_index = degree_error(test_x, test_y, route_x, route_y, route_heading, recovered_heading)
                route_errors.extend(errors)
                # Difference between matched index and minimum distance index
                abs_index_diffs.extend([abs(i - j) for i, j in zip(nav.get_index_log(), min_dist_index)])
            self.jobs += 1
            print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))


            mean_route_error = sum(route_errors) / len(route_errors)
            self.log['tested routes'].extend([no_of_routes])
            self.log['blur'].extend([combo_dict['blur']])
            self.log['edge'].extend([combo_dict['edge_range']])
            self.log['window'].extend([window])
            self.log['matcher'].extend([matcher])
            self.log['mean error'].extend([mean_route_error])
            self.log['errors'].append(route_errors)
            self.log['seconds'].append(time_compl)
            self.log['abs index diff'].append(abs_index_diffs)
        return self.log

    def _total_jobs(self, params):
        for k in params:
            self.total_jobs = self.total_jobs * len(params[k])
        print('Total number of jobs: {}'.format(self.total_jobs))

    def benchmark(self, params, route_ids, parallel=False):

        assert isinstance(params, dict)
        assert isinstance(route_ids, list)

        if parallel:
            self.bench_paral(params, route_ids)
        else:
            self.log = self.bench_singe_core(params, route_ids)

        bench_results = pd.DataFrame(self.log)

        bench_results.to_csv(self.results_path)
        print(bench_results)

    @staticmethod
    def worker_bench(keys, route_ids, dist, shared, chunk):

        log = {'tested routes': [], 'blur': [], 'edge':[], 'window': [],
               'matcher': [], 'mean error': [], 'errors': [], 'seconds': [],
               'abs index diff': []}

        no_of_routes = len(route_ids)
        #  Go though all combinations in the chunk
        for combo in chunk:
            route_errors = []
            time_compl = []
            abs_index_diffs = []

            # create combo dictionary
            combo_dict = {}
            for i, k in enumerate(keys):
                combo_dict[k] = combo[i]

            matcher = combo_dict['matcher']
            window = combo_dict['window']
            for route in route_ids:  # for every route
                _, test_x, test_y, test_imgs, route_x, route_y, \
                route_heading, route_imgs = load_route(route, dist)
                tic = timeit.default_timer()
                # Preprocess images
                test_imgs = pre_process(test_imgs, combo_dict)
                route_imgs = pre_process(route_imgs, combo_dict)
                # Run navigation algorithm
                nav = spm.SequentialPerfectMemory(route_imgs, matcher)
                recovered_heading, logs, window_log = nav.navigate(test_imgs, window)
                toc = timeit.default_timer()
                # Get time complexity
                time_compl.append(toc - tic)
                # Get the errors and the minimum distant index of the route memory
                errors, min_dist_index = degree_error(test_x, test_y, route_x, route_y, route_heading,
                                                      recovered_heading)
                route_errors.extend(errors)
                # Difference between matched index and minimum distance index
                abs_index_diffs.extend([abs(i - j) for i, j in zip(nav.get_index_log(), min_dist_index)])
            # Increment the complete jobs shared variable
            shared[0] = shared[0] + 1
            print(multiprocessing.current_process(),' jobs completed: {}/{}'.format(shared[0], shared[1]))

            mean_route_error = sum(route_errors) / len(route_errors)
            log['tested routes'].extend([no_of_routes])
            log['blur'].extend([combo_dict['blur']])
            log['edge'].extend([combo_dict['edge_range']])
            log['window'].extend([window])
            log['matcher'].extend([matcher])
            log['mean error'].extend([mean_route_error])
            log['errors'].append(route_errors)
            log['seconds'].append(time_compl)
            log['abs index diff'].append(abs_index_diffs)
        return log
