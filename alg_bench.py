from utils import pre_process, mean_degree_error, load_route, sample_from_wg,  plot_map
from sequential_perfect_memory import seq_perf_mem, seq_perf_mem_cor
import pandas as pd

DIST = 100



def benchmark(route_ids=None, window_range=None):
    total_jobs = len(window_range) * len(route_ids) * 2
    print('Total number of jobs: {}'.format(total_jobs))

    jobs = 0
    bench_logs = []
    for window in window_range:
        route_errors = []
        for route in route_ids:  # for every route
            _, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, route_heading, route_images = load_route(str(route))

            X_inrange, Y_inrange, w_g_imgs_inrange = sample_from_wg(X_inlimit, Y_inlimit,
                                                                    X_route, Y_route,
                                                                    world_grid_imgs, DIST)

            pre_world_grid_imgs = pre_process(w_g_imgs_inrange, (180, 50))
            pre_route_images = pre_process(route_images, (180, 50))

            Recovered_Heading, logs, window_log = seq_perf_mem(pre_world_grid_imgs, pre_route_images, window=window,
                                                               mem_pointer=0)
            mean_error = mean_degree_error(X_inrange, Y_inrange, X_route, Y_route, route_heading, Recovered_Heading)
            route_errors.append(mean_error)

            # #plot the full route with headings_heading
            # U, V = pol_2cart_headings(Recovered_Heading)

            # plot_map([X_route, Y_route], [X_inrange, Y_inrange], size=(15, 15), grid_vectors=[U, V],
            #          scale=80, route_zoom=True)
            jobs += 1
            print('Jobs completed: {}/{}'.format(jobs, total_jobs))
        # Mean route error
        mean_route_error = sum(route_errors) / len(route_errors)

        row = []
        row.append(len(route_ids))      # Sequential
        row.append(True)                # Seq
        row.append(window)              # memory window
        row.append(False)               # CorCoef
        row.append(True)               # RMSE
        row.append(mean_route_error)    # Mean route error

        bench_logs.append(row)

    #TODO: need to put this code in a function.
    for window in window_range:
        route_errors = []
        for route in route_ids:  # for every route
            _, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, route_heading, route_images = load_route(str(route))

            X_inrange, Y_inrange, w_g_imgs_inrange = sample_from_wg(X_inlimit, Y_inlimit,
                                                                    X_route, Y_route,
                                                                    world_grid_imgs, DIST)

            pre_world_grid_imgs = pre_process(w_g_imgs_inrange, (180, 50))
            pre_route_images = pre_process(route_images, (180, 50))

            Recovered_Heading, logs, window_log = seq_perf_mem_cor(pre_world_grid_imgs, pre_route_images, window=window,
                                                               mem_pointer=0)
            mean_error = mean_degree_error(X_inrange, Y_inrange, X_route, Y_route, route_heading, Recovered_Heading)
            route_errors.append(mean_error)

            # #plot the full route with headings_heading
            # U, V = pol_2cart_headings(Recovered_Heading)

            # plot_map([X_route, Y_route], [X_inrange, Y_inrange], size=(15, 15), grid_vectors=[U, V],
            #          scale=80, route_zoom=True)
            jobs += 1
            print('Jobs completed: {}/{}'.format(jobs, total_jobs))
        # Mean route error
        mean_route_error = sum(route_errors) / len(route_errors)

        row = []
        row.append(len(route_ids))      # Sequential
        row.append(True)                # Seq
        row.append(window)              # memory window
        row.append(True)               # CorCoef
        row.append(False)               # RMSE
        row.append(mean_route_error)    # Mean route error

        bench_logs.append(row)

    bench_results = pd.DataFrame(bench_logs, columns=['Tested routes', 'Seq', 'Window', 'CorCoef', 'RMSE', 'Mean Error'])
    print(bench_results)


benchmark([1, 2], list(range(5, 16)))


def bench_seq_pm(route_ids=None, window_range=None):
    logs = []
    for window in window_range:
        route_errors = []
        for route in route_ids:  # for every route
            _, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, route_heading, route_images = load_route(str(route))

            X_inrange, Y_inrange, w_g_imgs_inrange = sample_from_wg(X_inlimit, Y_inlimit,
                                                                    X_route, Y_route,
                                                                    world_grid_imgs, DIST)

            pre_world_grid_imgs = pre_process(w_g_imgs_inrange, (180, 50))
            pre_route_images = pre_process(route_images, (180, 50))

            Recovered_Heading, logs, window_log = seq_perf_mem(pre_world_grid_imgs, pre_route_images, window=window,
                                                               mem_pointer=0)
            mean_error = mean_degree_error(X_inrange, Y_inrange, X_route, Y_route, route_heading, Recovered_Heading)
            route_errors.append(mean_error)

            # #plot the full route with headings_heading
            # U, V = pol_2cart_headings(Recovered_Heading)

            # plot_map([X_route, Y_route], [X_inrange, Y_inrange], size=(15, 15), grid_vectors=[U, V],
            #          scale=80, route_zoom=True)
            jobs += 1
            print('Jobs completed: {}/{}'.format(jobs, total_jobs))
        # Mean route error
        mean_route_error = sum(route_errors) / len(route_errors)

        # Seq, memory window, CorCoef, RMSE, Mean route error
        logs.append([len(route_ids), True, window, False, True, mean_route_error])

    return logs

