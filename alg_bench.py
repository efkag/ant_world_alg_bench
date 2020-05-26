from utils import pre_process, mean_degree_error, load_route, sample_from_wg, plot_map, degree_error
import sequential_perfect_memory as spm
import perfect_memory as pm
import pandas as pd
import timeit

DIST = 100

class Benchmark():
    def __init__(self, results_path):
        self.results_path = results_path
        self.jobs = None
        self.total_jobs = None
        self.bench_logs = []
        self.route_ids = None
        self.routes_data = []
        self.dist = 100  # Distance between grid images and route images
        self.img_shape = (180, 50)
        self.log = {'tested routes': [], 'pre-proc': [], 'seq': [], 'window': [],
                    'matcher': [], 'mean error': [], 'errors': [], 'seconds': [],
                    'abs index diff': []}

    def load_routes(self, route_ids):
        self.route_ids = route_ids
        for id in self.route_ids:
            route_data = load_route(id, self.dist)
            self.routes_data.append(route_data)

    def bench_seq_pm(self, route_ids=None, pre_procs=None, window_range=None, matchers=None):
        for window in window_range:
            for matching in matchers:
                for pre_proc in pre_procs:
                    route_errors = []
                    time_compl = []
                    abs_index_diffs = []
                    for route in route_ids:  # for every route
                        _, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_route(route, self.dist)
                        tic = timeit.default_timer()
                        # Preprocess images
                        pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
                        pre_route_images = pre_process(route_images, pre_proc)
                        # Run navigation algorithm
                        nav = spm.SequentialPerfectMemory(pre_route_images, matching)
                        recovered_heading, logs, window_log = nav.navigate(pre_world_grid_imgs, window)
                        toc = timeit.default_timer()
                        time_compl.append(toc-tic)
                        # Get the errors and the minimum distant index of the route memory
                        errors, min_dist_index = degree_error(x_inlimit, y_inlimit, x_route, y_route, route_heading, recovered_heading)
                        route_errors.extend(errors)
                        # Difference between matched index and minimum distance index
                        abs_index_diffs.extend([abs(i - j) for i, j in zip(nav.get_index_log(), min_dist_index)])
                        self.jobs += 1
                        print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))

                    # Flatten errors
                    # route_errors = [item for sublist in route_errors for item in sublist]
                    # Mean route error
                    mean_route_error = sum(route_errors) / len(route_errors)
                    self.log['tested routes'].extend([len(route_ids)])
                    self.log['pre-proc'].extend([str(pre_proc)])
                    self.log['seq'].extend([True])
                    self.log['window'].extend([window])
                    self.log['matcher'].extend([matching])
                    self.log['mean error'].extend([mean_route_error])
                    self.log['errors'].append(route_errors)
                    self.log['seconds'].append(time_compl)
                    self.log['abs index diff'].append(abs_index_diffs)
        return self.log

    def bench_pm(self, route_ids=None, pre_procs=None, matchers=None):
        for matching in matchers:
            for pre_proc in pre_procs:
                route_errors = []
                time_compl = []
                for route in route_ids:  # for every route
                    _, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                        route_heading, route_images = load_route(route, self.dist)
                    tic = timeit.default_timer()
                    # Preprocess images
                    pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
                    pre_route_images = pre_process(route_images, pre_proc)
                    # Run navigation algorithm
                    nav = pm.PerfectMemory(pre_route_images, matching)
                    recovered_heading, logs = nav.navigate(pre_world_grid_imgs)
                    toc = timeit.default_timer()
                    time_compl.extend([toc - tic])
                    route_errors.extend(degree_error(x_inlimit, y_inlimit, x_route, y_route,
                                                          route_heading, recovered_heading))
                    # #plot the full route with headings_heading
                    # U, V = pol_2cart_headings(recovered_heading)
                    # plot_map([x_route, y_route], [x_inrange, y_inrange], size=(15, 15), grid_vectors=[U, V],
                    #          scale=80, route_zoom=True)
                    self.jobs += 1
                    print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))
                # Flatten errors
                # route_errors = [item for sublist in route_errors for item in sublist]
                # Mean route error
                mean_route_error = sum(route_errors) / len(route_errors)
                self.log['tested routes'].extend([len(route_ids)])
                self.log['pre-proc'].extend([str(pre_proc)])
                self.log['seq'].extend([False])
                self.log['window'].extend([None])
                self.log['matcher'].extend([matching])
                self.log['mean error'].extend([mean_route_error])
                self.log['errors'].append(route_errors)
                self.log['seconds'].append(time_compl)
        return self.log

    def benchmark_init(self, alg, route_ids, pre_processing, window_range=None, matchers=None):
        self.jobs = 0
        if alg == 'spm':
            # Benchmark for sequential-pm
            self.total_jobs = len(pre_processing) * len(window_range) * len(route_ids) * len(matchers)
            print('Total number of jobs: {}'.format(self.total_jobs))
            self.bench_seq_pm(route_ids, pre_processing, window_range, matchers)
        elif alg == 'pm':
            self.total_jobs = len(pre_processing) * len(route_ids) * len(matchers)
            print('Total number of jobs: {}'.format(self.total_jobs))
            self.bench_pm(route_ids, pre_processing, matchers)

        bench_results = pd.DataFrame(self.log)

        bench_results.to_csv(self.results_path)
        print(bench_results)
