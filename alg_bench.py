from utils import pre_process, mean_degree_error, load_route, sample_from_wg, plot_map, degree_error
import sequential_perfect_memory as spm
import perfect_memory as pm
import pandas as pd
import numpy as np

DIST = 100

class Benchmark():
    def __init__(self, results_path):
        self.results_path = results_path
        self.jobs = None
        self.total_jobs = None
        self.bench_logs = []
        self.dist = 100  # Distance between grid images and route images
        self.img_shape = (180, 50)
        self.log = {'tested routes': [], 'pre-proc': [], 'seq': [], 'window': [],
                    'matcher': [], 'mean error': [], 'errors': []}

    def bench_seq_pm(self, route_ids=None, pre_procs=None, window_range=None, matchers=None):
        for window in window_range:
            for matching in matchers:
                for pre_proc in pre_procs:
                    route_errors = []
                    for route in route_ids:  # for every route
                        _, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_route(route, route_direction='right2left')

                        x_inlimit, y_inlimit, world_grid_imgs = sample_from_wg(x_inlimit, y_inlimit,
                                                                                x_route, y_route,
                                                                                world_grid_imgs, DIST)
                        # Preprocess images
                        pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
                        pre_route_images = pre_process(route_images, pre_proc)
                        # Run navigation algorithm
                        nav = spm.SequentialPerfectMemory(pre_route_images, matching)
                        recovered_heading, logs, window_log = nav.navigate(pre_world_grid_imgs, window)

                        route_errors.append(degree_error(x_inlimit, y_inlimit, x_route, y_route,
                                                              route_heading, recovered_heading))

                        self.jobs += 1
                        print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))

                    # Flatten errors
                    route_errors = [item for sublist in route_errors for item in sublist]
                    # Mean route error
                    mean_route_error = sum(route_errors) / len(route_errors)
                    self.log['tested routes'].extend([len(route_ids)])
                    self.log['pre-proc'].extend([pre_proc.keys()])
                    self.log['seq'].extend([True])
                    self.log['window'].extend([window])
                    self.log['matcher'].extend([matching])
                    self.log['mean error'].extend([mean_route_error])
                    self.log['errors'].append(route_errors)

        return self.log

    def bench_pm(self, route_ids=None, pre_procs=None, matchers=None):
        for matching in matchers:
            for pre_proc in pre_procs:
                route_errors = []
                for route in route_ids:  # for every route
                    _, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                        route_heading, route_images = load_route(str(route))

                    x_inrange, y_inrange, w_g_imgs_inrange = sample_from_wg(x_inlimit, y_inlimit,
                                                                            x_route, y_route,
                                                                            world_grid_imgs, DIST)
                    # Preprocess images
                    pre_world_grid_imgs = pre_process(w_g_imgs_inrange, pre_proc)
                    pre_route_images = pre_process(route_images, pre_proc)
                    # Run navigation algorithm
                    nav = pm.PerfectMemory(pre_route_images, matching)
                    recovered_heading, logs = nav.navigate(pre_world_grid_imgs)

                    route_errors.append(mean_degree_error(x_inrange, y_inrange, x_route, y_route,
                                                          route_heading, recovered_heading))

                    # #plot the full route with headings_heading
                    # U, V = pol_2cart_headings(recovered_heading)

                    # plot_map([x_route, y_route], [x_inrange, y_inrange], size=(15, 15), grid_vectors=[U, V],
                    #          scale=80, route_zoom=True)
                    self.jobs += 1
                    print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))
                # Mean route error
                mean_route_error = sum(route_errors) / len(route_errors)

                # Num of routes, pre_proc, Seq, memory window, Matcher, Mean route error
                self.bench_logs.append([len(route_ids), pre_proc.keys(), False, None, matching, mean_route_error])
        return self.bench_logs

    def benchmark_init(self, route_ids, pre_processing, window_range=None, matchers=None):
        nav_alg_num = 1
        ## TODO need to alter code to pass any number of navigation algorithms to test
        # self.total_jobs = len(pre_processing) * len(route_ids) * len(matchers) * nav_alg_num
        self.total_jobs = len(pre_processing) * len(window_range) * len(route_ids) * len(matchers) * nav_alg_num
        print('Total number of jobs: {}'.format(self.total_jobs))
        self.jobs = 0

        # Benchmark for sequential-pm
        self.bench_seq_pm(route_ids, pre_processing, window_range, matchers)

        # self.bench_pm(route_ids, pre_processing, matchers)

        bench_results = pd.DataFrame(self.log)

        bench_results.to_csv(self.results_path)
        print(bench_results)
