from utils import pre_process, mean_degree_error, load_route, sample_from_wg, plot_map
from sequential_perfect_memory import seq_perf_mem, seq_perf_mem_cor
import pandas as pd
import numpy as np


class benchmark():
    def __init__(self):
        self.jobs = None
        self.total_jobs = None
        self.bench_logs = []
        self.dist = 100  # Distance between grid images and route images
        self.img_shape = (180, 50)

    def bench_seq_pm(self, route_ids=None, window_range=None, corr_coef=True, edges=False):
        logs = []
        for window in window_range:
            route_errors = []
            for route in route_ids:  # for every route
                _, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, route_heading, route_images = load_route(
                    str(route))

                X_inrange, Y_inrange, w_g_imgs_inrange = sample_from_wg(X_inlimit, Y_inlimit,
                                                                        X_route, Y_route,
                                                                        world_grid_imgs, self.dist)

                pre_world_grid_imgs = pre_process(w_g_imgs_inrange, self.img_shape, edges)
                pre_route_images = pre_process(route_images, self.img_shape, edges)

                if corr_coef:
                    Recovered_Heading, logs, window_log = seq_perf_mem_cor(pre_world_grid_imgs, pre_route_images,
                                                                           window=window, mem_pointer=0)
                else:
                    Recovered_Heading, logs, window_log = seq_perf_mem(pre_world_grid_imgs, pre_route_images,
                                                                       window=window, mem_pointer=0)
                mean_error = mean_degree_error(X_inrange, Y_inrange, X_route, Y_route, route_heading, Recovered_Heading)
                route_errors.append(mean_error)

                # #plot the full route with headings_heading
                # U, V = pol_2cart_headings(Recovered_Heading)
                # plot_map([X_route, Y_route], [X_inrange, Y_inrange], size=(15, 15), grid_vectors=[U, V],
                #          scale=80, route_zoom=True)

                self.jobs += 1
                print('Jobs completed: {}/{}'.format(self.jobs, self.total_jobs))
            # Mean route error
            mean_route_error = sum(route_errors) / len(route_errors)

            #                                       Seq, Edges, window, CorCoef, RMSE, Mean route error
            self.bench_logs.append([len(route_ids), True, edges,  window, corr_coef, not corr_coef, mean_route_error])
        return logs


    def benchmark_init(self, route_ids=None, window_range=None):
        nav_alg_num = 2
        self.total_jobs = len(window_range) * len(route_ids) * nav_alg_num * 2
        print('Total number of jobs: {}'.format(self.total_jobs))
        self.jobs = 0

        # Benchmark for sequential pm with RMSE
        self.bench_seq_pm(route_ids, window_range, corr_coef=False)

        # Benchmark for sequential pm with RMSE, with edges
        self.bench_seq_pm(route_ids, window_range, corr_coef=False, edges=True)

        # Benchmark for sequential pm with corr_coef
        self.bench_seq_pm(route_ids, window_range, corr_coef=True)

        # Benchmark for sequential pm with corr_coef
        self.bench_seq_pm(route_ids, window_range, corr_coef=True, edges=True)

        bench_results = pd.DataFrame(self.bench_logs,
                                     columns=['Tested routes', 'Seq', 'Edges',  'Window', 'CorCoef', 'RMSE', 'Mean Error'])
        print(bench_results)
