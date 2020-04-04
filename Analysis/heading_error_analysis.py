from utils import pre_process, mean_degree_error, degree_error_logs, display_image, load_route, \
    check_for_dir_and_create, plot_map, save_image
import sequential_perfect_memory as spm
import perfect_memory as pm
import numpy as np


matcher = 'idf'
pre_proc = {'blur': True, 'shape': (180, 50)}
window = 14
figures_path = 'Figures/'
dist = 100
route = 1
check_for_dir_and_create(figures_path)

w, x_inlimit, y_inlimit, grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_route(route_id=route, grid_pos_limit=dist)

# plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15),
#          route_headings=route_heading, scale=40)

pre_grid_imgs = pre_process(grid_imgs, pre_proc)
pre_route_images = pre_process(route_images, pre_proc)

nav = spm.SequentialPerfectMemory(pre_route_images, matcher)
recovered_heading, logs, window_log = nav.navigate(pre_grid_imgs, window)
# nav = pm.PerfectMemory(pre_route_images, matcher)
# recovered_heading, logs = nav.navigate(pre_grid_imgs)

print(mean_degree_error(x_inlimit, y_inlimit, x_route, y_route, route_heading, recovered_heading))

# plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15),
#          route_headings=route_heading, grid_headings=recovered_heading, scale=40)
error_threshold = 45
error_logs = degree_error_logs(x_inlimit, y_inlimit, x_route, y_route,
                               route_heading, recovered_heading, error_threshold)

xy_route_error = [error_logs['x_route'], error_logs['y_route']]
xy_grid_error = [error_logs['x_grid'], error_logs['y_grid']]
heading_route_error = error_logs['route_heading']
heading_grid_error = error_logs['grid_heading']
# plot_map(w, xy_route_error, xy_grid_error, size=(15, 15),
#          route_headings=heading_route_error, grid_headings=heading_grid_error, scale=40)


# Write error threshold images to a new directory
destination = 'Images/'
check_for_dir_and_create(destination)
for i, idx in enumerate(error_logs['grid_idx']):
    error_dir_path = destination + 'error_' + str(i) + '_' + str(int(error_logs['errors'][i])) + 'deg/'
    check_for_dir_and_create(error_dir_path)
    save_image(error_dir_path + 'grid_' + str(idx) + '.png', grid_imgs[idx])
    save_image(error_dir_path + 'route_' + str(error_logs['route_idx'][i]) + '.png', route_images[error_logs['route_idx'][i]])

