from utils import pre_process, mean_degree_error, load_route, plot_map
from sequential_perfect_memory import seq_perf_mem

world, X, Y, world_grid_imgs, X_route, Y_route, route_heading, route_images = utl.load_route("2")

plot_map(world, [X_route, Y_route], [X, Y])


def benchmark(route_ids=None, window_range=None):
    bench_logs = []
    for window in window_range:
        route_errors = []
        for route in route_ids:  # for every route
            _, X_inlimit, Y_inlimit, world_grid_imgs, X_route, Y_route, route_heading, route_images = load_route(str(route))

            X_inrange, Y_inrange, w_g_imgs_inrange = sample_from_wg(100)

            pre_world_grid_imgs = pre_process(w_g_imgs_inrange, (180, 50))
            pre_route_images = pre_process(route_images, (180, 50))

            Recovered_Heading, logs, window_log = seq_perf_mem(pre_world_grid_imgs, pre_route_images, window=window,
                                                               mem_pointer=0)
            mean_error = mean_degree_error(X_inrange, Y_inrange, X_route, Y_route, route_heading, Recovered_Heading)
            print("mean error: " + str(mean_error))
            route_errors.append(mean_error)

            # #plot the full route with headings_heading
            # U, V = pol_2cart_headings(Recovered_Heading)

            # plot_map([X_route, Y_route], [X_inrange, Y_inrange], size=(15, 15), grid_vectors=[U, V],
            #          scale=80, route_zoom=True)
        # Mean route error
        mean_route_error = sum(route_errors) / len(route_errors)

        row = []
        row.append(len(route_ids))  # Sequential
        row.append(True)            # corrCoef
        row.append(window)          # memory window
        row.append(False)           # Mean route error
        row.append(True)            # RMSE
        row.append(mean_route_error)# Mean route error

        bench_logs.append(row)
