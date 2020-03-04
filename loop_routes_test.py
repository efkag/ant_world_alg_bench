from utils import pre_process, mean_degree_error, load_grid_route, sample_from_wg, plot_map, pol_2cart_headings
import sequential_perfect_memory as spm

directory = 'LoopRoutes/'
matcher = 'idf'
pre_proc = dict({'blur': True, 'shape': (180, 50)})
window = 5
w, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_grid_route(route_dir=directory)


#plot the full route
U, V = pol_2cart_headings(route_heading)
plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15), vectors=[U, V], scale=80)

pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
pre_route_images = pre_process(route_images, pre_proc)

nav = spm.SequentialPerfectMemory(pre_route_images, matcher)
recovered_heading, logs, window_log = nav.navigate(pre_world_grid_imgs, window)

grid_U, grid_V = pol_2cart_headings(route_heading)
plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15),
         vectors=[U, V], grid_vectors=[grid_U, grid_V], scale=80)

print('break')

