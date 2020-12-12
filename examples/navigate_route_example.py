from source.utils import pre_process, mean_degree_error, load_route, check_for_dir_and_create, plot_map
from source import sequential_perfect_memory as spm

directory = 'LoopRoutes/'
matcher = 'rmse'
pre_proc = {'blur': True, 'shape': (180, 50)}
window = 14
figures_path = 'Figures/'
dist = 100
check_for_dir_and_create(figures_path)

w, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_route(route_id=1, grid_pos_limit=dist)



plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15),
         route_headings=route_heading, scale=40)

pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
pre_route_images = pre_process(route_images, pre_proc)

nav = spm.SequentialPerfectMemory(pre_route_images, matcher)
recovered_heading, logs, window_log = nav.navigate(pre_world_grid_imgs, window)
# nav = pm.PerfectMemory(pre_route_images, matcher)
# recovered_heading, logs = nav.navigate(pre_world_grid_imgs)

print(mean_degree_error(x_inlimit, y_inlimit, x_route, y_route, route_heading, recovered_heading))

plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15),
         route_headings=route_heading, grid_headings=recovered_heading, scale=40)


# id = 0
# for window in window_log:
#     plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15), marker_size=100, scale=40,
#              vectors=[U, V], grid_vectors=[grid_U, grid_V], show=False, save=True, save_id=id,
#              window=window, path='Figures/')
#     id += 1
#
# print('break')
