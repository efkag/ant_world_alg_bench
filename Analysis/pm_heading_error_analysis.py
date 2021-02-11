from source.utils import pre_process, mean_degree_error, degree_error_logs, check_for_dir_and_create, plot_map, save_image, load_loop_route
import matplotlib.pyplot as plt
from source import perfect_memory as pm
import numpy as np
import seaborn as sns
sns.set(font_scale=2.5)

directory = '../LoopRoutes/'
matcher = 'rmse'
pre_proc = {'blur': True, 'shape': (90, 25)}
dist = 100
route = 1
error_threshold = 40
#figures_path = 'norm_routes/pm/route_' + str(route) + '/'
figures_path = 'loop_routes/pm/route_' + str(route) + '_alt__error/'
check_for_dir_and_create(figures_path)

# w, x_inlimit, y_inlimit, grid_imgs, x_route, y_route, \
#                             route_heading, route_images = load_route(route_id=route, grid_pos_limit=dist)
w, x_inlimit, y_inlimit, grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_loop_route(directory, route_id=route, grid_pos_limit=dist)
img_path = figures_path + 'Route.png'
plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(6, 6), path=img_path,
         route_headings=route_heading, scale=40, route_zoom=False, save=True)

pre_grid_imgs = pre_process(grid_imgs, pre_proc)
pre_route_images = pre_process(route_images, pre_proc)

nav = pm.PerfectMemory(pre_route_images, matcher)
recovered_heading, logs = nav.navigate(pre_grid_imgs)

print(mean_degree_error(x_inlimit, y_inlimit, x_route, y_route, route_heading, recovered_heading))
zoom = [np.mean(x_route), np.mean(y_route)]
img_path = figures_path + 'errorRoute.png'
plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(6, 6), path=img_path, # zoom=zoom,
         route_headings=route_heading, grid_headings=recovered_heading, scale=40, save=True)

error_logs = degree_error_logs(x_inlimit, y_inlimit, x_route, y_route,
                               route_heading, recovered_heading, error_threshold)
# World map with correct and error grid possitions
zoom = [np.mean(x_route), np.mean(y_route)]
img_path = figures_path + 'zoomCorrectsErrors.pdf'
plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(4.8, 4.8), zoom=zoom, path=img_path, error_indexes=error_logs['grid_idx'],
         route_headings=route_heading, grid_headings=recovered_heading, scale=25, save=True)


temp = np.array(error_logs['errors'])
xy_route_error = [error_logs['x_route'], error_logs['y_route']]
xy_grid_error = [error_logs['x_grid'], error_logs['y_grid']]
heading_route_error = error_logs['route_heading']
heading_grid_error = error_logs['grid_heading']
zoom = [np.mean(x_route), np.mean(y_route)]
img_path = figures_path + 'zoomError>' + str(error_threshold) +'.png'
plot_map(w, xy_route_error, xy_grid_error, size=(6, 6), path=img_path, zoom=zoom,
         route_headings=heading_route_error, grid_headings=heading_grid_error, scale=40, save=True)


# Write error threshold images to a new directory
destination = figures_path
check_for_dir_and_create(destination)
matched_index_logs = nav.get_index_log()
for i, v in enumerate(zip(error_logs['route_idx'], error_logs['grid_idx'])):
    # Index of the route position closest to the grid test position
    route_idx = v[0]
    # Index of the gird position tha generated error
    grid_idx = v[1]
    error_dir_path = destination + 'error_' + str(i) + '_' + str(int(error_logs['errors'][i])) + 'deg/'
    check_for_dir_and_create(error_dir_path)
    # Save the test image and the closest route imagea dn the matched route image.
    save_image(error_dir_path + 'grid_' + str(grid_idx) + '.png', grid_imgs[grid_idx])
    save_image(error_dir_path + 'route_' + str(route_idx) + '.png', route_images[route_idx])
    save_image(error_dir_path + 'matched_' + str(matched_index_logs[grid_idx]) + '.png', route_images[matched_index_logs[grid_idx]])
    fig_title = 'route heading = ' + str(route_heading[route_idx]) \
                + ', recovered heading = ' + str(recovered_heading[grid_idx])
    plot_map(w, [[x_route[route_idx]], [y_route[route_idx]]], [[x_inlimit[grid_idx]], [y_inlimit[grid_idx]]],
             route_headings=[route_heading[route_idx]], grid_headings=[recovered_heading[grid_idx]],
             path=error_dir_path, save=True, show=False, title=fig_title)
    # Save a heat-map of the window similarity matrix
    fig = plt.figure(figsize=(20, 40))
    title = 'Matched index: ' + str(matched_index_logs[grid_idx])
    plt.title(title)
    ax = sns.heatmap(logs[grid_idx])
    plt.xlabel('Degrees')
    plt.ylabel('Route memory index')
    ax.figure.savefig(error_dir_path + 'heat.png')
    plt.close()