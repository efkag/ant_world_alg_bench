from source.utils import pre_process, load_route, degree_error, load_route_naw, angular_error, check_for_dir_and_create, calc_dists
from source import seqnav as spm, perfect_memory as pm
import pandas as pd
import time
import itertools
import multiprocessing
import functools
import numpy as np


class Benchmark:
    def __init__(self, results_path, filename='results.csv'):
        self.results_path = results_path + filename
        check_for_dir_and_create(results_path)
        self.jobs = 0
        self.total_jobs = 1
        self.bench_logs = []
        self.route_ids = None
        self.routes_data = []
        self.dist = 0.2  # Distance between grid images and route images
        self.log = {'route_id': [], 'blur': [], 'edge': [], 'res': [], 'window': [],
                    'matcher': [], 'mean_error': [], 'errors': [], 'seconds': [],
                    'abs_index_diff': [], 'window_log': [], 'dist_diff': [],
                    'tx': [], 'ty': [], 'th': []}

    def load_routes(self, route_ids):
        path = '../new-antworld/'
        self.route_ids = route_ids
        for rid in self.route_ids:
            route_path = path + '/route' + str(rid) + '/'
            route_data = load_route_naw(route_path, route_id=id, query=True,  max_dist=self.dist)
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
        if self.total_jobs < multiprocessing.cpu_count():
            no_of_chunks = self.total_jobs
        else:
            no_of_chunks = multiprocessing.cpu_count() - 1
        # Generate list chunks of grid combinations
        chunks = self.get_grid_chunks(grid, no_of_chunks)
        # Partial callable
        worker = functools.partial(self.worker_bench, keys, route_ids, self.dist, shared)

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
                route = load_route_naw(route_path, route_id=route_id, imgs=True, query=True, max_dist=0.2)
                # _, test_x, test_y, test_imgs, route_x, route_y, \
                #     route_heading, route_imgs = load_route(route, self.dist)

                tic = time.perf_counter()
                # Preprocess images
                test_imgs = route['qimgs']
                test_imgs = pre_process(test_imgs, combo_dict)
                route_imgs = route['imgs']
                route_imgs = pre_process(route_imgs, combo_dict)
                # Run navigation algorithm
                if window:
                    nav = spm.SequentialPerfectMemory(route_imgs, matcher)
                    recovered_heading, logs, window_log = nav.navigate(test_imgs, window)
                else:
                    nav = pm.PerfectMemory(route_imgs, matcher)
                    recovered_heading, logs = nav.navigate(test_imgs)

                toc = time.perf_counter()
                # Get time complexity
                time_compl = toc-tic
                # Get the errors and the minimum distant index of the route memory
                # errors, min_dist_index = degree_error(test_x, test_y, route_x, route_y, route_heading, recovered_heading)
                traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
                errors, min_dist_index = angular_error(route, traj)
                # Difference between matched index and minimum distance index and distance between points
                matched_index = nav.get_index_log()
                abs_index_diffs = np.absolute(np.subtract(nav.get_index_log(), min_dist_index))
                dist_diff = calc_dists(route, min_dist_index, matched_index)
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

    def benchmark(self, params, route_ids, parallel=False):

        assert isinstance(params, dict)
        assert isinstance(route_ids, list)

        if parallel:
            self.log = None
            self.log = self.bench_paral(params, route_ids)
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
    def worker_bench(keys, route_ids, dist, shared, chunk):

        log = {'route_id': [], 'blur': [], 'edge': [], 'res': [], 'window': [],
               'matcher': [], 'mean_error': [], 'seconds': [], 'errors': [],
               'abs_index_diff': [], 'window_log': [], 'dist_diff': [],
               'tx': [], 'ty': [], 'th': []}
        #  Go though all combinations in the chunk
        for combo in chunk:
            # create combo dictionary
            combo_dict = {}
            for i, k in enumerate(keys):
                combo_dict[k] = combo[i]

            matcher = combo_dict['matcher']
            window = combo_dict['window']
            window_log = None
            for route_id in route_ids:  # for every route
                route_path = '../new-antworld/route' + str(route_id) + '/'
                route = load_route_naw(route_path, route_id=route_id, imgs=True, query=True, max_dist=0.2)
                # _, test_x, test_y, test_imgs, route_x, route_y, \
                # route_heading, route_imgs = load_route(route, dist)
                tic = time.perf_counter()
                # Preprocess images
                test_imgs = route['qimgs']
                test_imgs = pre_process(test_imgs, combo_dict)
                route_imgs = route['imgs']
                route_imgs = pre_process(route_imgs, combo_dict)
                # Run navigation algorithm
                if window:
                    nav = spm.SequentialPerfectMemory(route_imgs, matcher)
                    recovered_heading, logs, window_log = nav.navigate(test_imgs, window)
                else:
                    nav = pm.PerfectMemory(route_imgs, matcher)
                    recovered_heading, logs = nav.navigate(test_imgs)
                toc = time.perf_counter()
                # Get time complexity
                time_compl = toc - tic
                # Get the errors and the minimum distant index of the route memory
                traj = {'x': route['qx'], 'y': route['qy'], 'heading': recovered_heading}
                errors, min_dist_index = angular_error(route, traj)
                # Difference between matched index and minimum distance index
                matched_index = nav.get_index_log()
                abs_index_diffs = np.absolute(np.subtract(matched_index, min_dist_index))
                dist_diff = calc_dists(route, min_dist_index, matched_index)
                mean_route_error = np.mean(errors)
                log['route_id'].extend([route_id])
                log['blur'].extend([combo_dict.get('blur')])
                log['edge'].extend([combo_dict.get('edge_range')])
                log['res'].append(combo_dict.get('shape'))
                log['window'].extend([window])
                log['matcher'].extend([matcher])
                log['mean_error'].append(mean_route_error)
                log['seconds'].append(time_compl)
                log['window_log'].append(window_log)
                log['tx'].append(traj['x'].tolist())
                log['ty'].append(traj['y'].tolist())
                log['th'].append(traj['heading'])
                log['abs_index_diff'].append(abs_index_diffs.tolist())
                log['dist_diff'].append(dist_diff.tolist())
                log['errors'].append(errors)
            # Increment the complete jobs shared variable
            shared[0] = shared[0] + 1
            print(multiprocessing.current_process(), ' jobs completed: {}/{}'.format(shared[0], shared[1]))
        return log
