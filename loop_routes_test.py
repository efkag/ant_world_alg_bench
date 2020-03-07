from utils import pre_process, mean_degree_error, load_grid_route, check_for_dir_and_create, plot_map, pol_2cart_headings
import sequential_perfect_memory as spm

directory = 'LoopRoutes/'
matcher = 'corr'
pre_proc = dict({'blur': True, 'shape': (180, 50), 'edge_range': (180, 200)})
window = 5
figures_path = 'Figures/'
check_for_dir_and_create(figures_path)

w, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_grid_route(directory, route_id=2, grid_pos_limit=120)


U, V = pol_2cart_headings(route_heading)
plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15),
         vectors=[U, V], scale=70)

pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
pre_route_images = pre_process(route_images, pre_proc)

nav = spm.SequentialPerfectMemory(pre_route_images, matcher)
recovered_heading, logs, window_log = nav.navigate(pre_world_grid_imgs, window)


grid_U, grid_V = pol_2cart_headings(route_heading)
plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15),
         vectors=[U, V], grid_vectors=[grid_U, grid_V], scale=70)


id = 0
for window in window_log:
    plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15), marker_size=100, scale=80,
             vectors=[U, V], grid_vectors=[grid_U, grid_V], show=False, save=True, save_id=id,
             window=window, path='Figures/')
    id += 1

print('break')

