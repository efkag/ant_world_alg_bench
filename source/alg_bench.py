from source.utils import pre_process, load_route, degree_error
from source import sequential_perfect_memory as spm, perfect_memory as pm
import pandas as pd
import timeit
import itertools
import multiprocessing
import functools

class Benchmark:
    def __init__(self, results_path):
        self.results_path = results_path
        self.jobs = 0
        self.total_jobs = 1
        self.bench_logs = []
        self.route_ids = None
        self.routes_data = []
        self.dist = 100  # Distance between grid images and route images
        self.log = {'tested routes': [], 'pre-proc': [], 'window': [],
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

    def bench_paral(self, params, route_ids=None):
        print(multiprocessing.cpu_count(), ' CPU cores found')
        self._total_jobs(params)
        # get the indices of each parameter
        params_ind = {}
        for i, k in enumerate(params):
            params_ind[k] = i
        global total_jobs, jobs
        jobs = 0
        total_jobs = self.total_jobs
        # Generate grid iterable
        grid = itertools.product(*[params[k] for k in params])
        # Generate chunks
        chunks =  self.get_grid_chunks(grid, multiprocessing.cpu_count())

        worker = functools.partial(self.worker_bench, params_ind, route_ids, self.dist)

        pool = multiprocessing.Pool()

        logs = pool.map(worker, chunks)
        pool.close()
        pool.join()
        print(logs)
        return logs

    def bench_seq_pm(self, params, route_ids=None):

        grid = itertools.product(*[params[k] for k in params])

        # get the indices of each parameter
        params_ind = {}
        for i, k in enumerate(params):
            params_ind[k] = i

        no_of_routes = len(route_ids)
        #  Go though all combinations in the grid
        for combo in grid:
            route_errors = []
            time_compl = []
            abs_index_diffs = []

            matcher = combo[params_ind['matcher']]
            window = combo[params_ind['window']]
            for route in route_ids:  # for every route
                _, test_x, test_y, test_imgs, route_x, route_y, \
                    route_heading, route_imgs = load_route(route, self.dist)
                tic = timeit.default_timer()
                # Preprocess images
                test_imgs = pre_process(test_imgs, combo, params_ind)
                route_imgs = pre_process(route_imgs, combo, params_ind)
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
            self.log['blur'].extend([combo[params_ind['blur']]])
            self.log['edge'].extend([combo[params_ind['edge_range']]])
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

    def benchmark(self, alg, route_ids, params):
        if alg == 'spm':
            # Benchmark for sequential-pm
            # Get the total number of jobs form the params dictionary
            self._total_jobs(params)
            self.bench_seq_pm(params, route_ids)
        # elif alg == 'pm':
        #     self.total_jobs = len(pre_processing) * len(route_ids) * len(matchers)
        #     print('Total number of jobs: {}'.format(self.total_jobs))
        #     self.bench_pm(route_ids, pre_processing, matchers)

        bench_results = pd.DataFrame(self.log)

        bench_results.to_csv(self.results_path)
        print(bench_results)

    @staticmethod
    def worker_bench(params_ind, route_ids, dist, chunk):
        log = {'tested routes': [], 'pre-proc': [], 'window': [],
               'matcher': [], 'mean error': [], 'errors': [], 'seconds': [],
               'abs index diff': []}

        no_of_routes = len(route_ids)
        #  Go though all combinations in the chunk
        for combo in chunk:
            route_errors = []
            time_compl = []
            abs_index_diffs = []

            matcher = combo[params_ind['matcher']]
            window = combo[params_ind['window']]
            for route in route_ids:  # for every route
                _, test_x, test_y, test_imgs, route_x, route_y, \
                route_heading, route_imgs = load_route(route, dist)
                tic = timeit.default_timer()
                # Preprocess images
                test_imgs = pre_process(test_imgs, combo, params_ind)
                route_imgs = pre_process(route_imgs, combo, params_ind)
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
                # TODO: need to figure out how to pass messages  between processes to update the jobs parameter
                # jobs.value += 1
                # print('Jobs completed: {}/{}'.format(jobs, total_jobs))

            mean_route_error = sum(route_errors) / len(route_errors)
            log['tested routes'].extend([no_of_routes])
            log['blur'].extend([combo[params_ind['blur']]])
            log['edge'].extend([combo[params_ind['edge_range']]])
            log['window'].extend([window])
            log['matcher'].extend([matcher])
            log['mean error'].extend([mean_route_error])
            log['errors'].append(route_errors)
            log['seconds'].append(time_compl)
            log['abs index diff'].append(abs_index_diffs)
        return log