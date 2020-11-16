from source.utils import pre_process, mean_degree_error, degree_error_logs, load_route, \
    check_for_dir_and_create, plot_map, save_image
from source import sequential_perfect_memory as spm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(font_scale=2.5)

pre_proc = {'shape': (360, 75), 'edge_range': (180, 200)}
window = 17
dist = 100
route = 1
error_threshold = 40
# figures_path = 'norm_routes/spm/route_' + str(route) + '/'
logs_path = 'Figures/spm/route_' + str(route) + '_alt__error/'
check_for_dir_and_create(logs_path)

w, x_inlimit, y_inlimit, grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_route(route_id=route, grid_pos_limit=dist)

pre_grid_imgs = pre_process(grid_imgs, pre_proc)
pre_route_images = pre_process(route_images, pre_proc)

'''
Check SPM with IDF and with correlation
'''
matcher = 'corr'
figures_path = logs_path + matcher + '/'
check_for_dir_and_create(figures_path)

nav = spm.SequentialPerfectMemory(pre_route_images, matcher)
recovered_heading, logs, window_log = nav.navigate(pre_grid_imgs, window)

print(mean_degree_error(x_inlimit, y_inlimit, x_route, y_route, route_heading, recovered_heading))
zoom = [np.mean(x_route), np.mean(y_route)]
img_path = figures_path + 'errorRoute.png'
plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15), path=img_path,
         route_headings=route_heading, grid_headings=recovered_heading, scale=40, save=True)

error_logs = degree_error_logs(x_inlimit, y_inlimit, x_route, y_route,
                               route_heading, recovered_heading, error_threshold)
df_error_logs = pd.DataFrame(error_logs)
df_error_logs.to_csv(figures_path + 'error_logs.csv')
zoom = [np.mean(x_route), np.mean(y_route)]
img_path = figures_path + 'zoomCorrectsErrors.png'
plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15), zoom=zoom, path=img_path,
         error_indexes=error_logs['grid_idx'], route_headings=route_heading, grid_headings=recovered_heading, scale=30, save=True)

for i, v in enumerate(zip(error_logs['route_idx'], error_logs['grid_idx'])):
    # Index of the route position closest to the grid test position
    route_idx = v[0]
    # Index of the gird position tha generated error
    grid_idx = v[1]
    error_dir_path = figures_path + 'error_' + str(i) + '_' + str(int(error_logs['errors'][i])) + 'deg/'
    check_for_dir_and_create(error_dir_path)
    # Save the test image and the closest route image.
    save_image(error_dir_path + 'grid_' + str(grid_idx) + '.png', grid_imgs[grid_idx])
    save_image(error_dir_path + 'route_' + str(route_idx) + '.png', route_images[route_idx])
    fig_title = 'route heading = ' + str(route_heading[route_idx]) \
                + ', recovered heading = ' + str(recovered_heading[grid_idx])
    plot_map(w, [[x_route[route_idx]], [y_route[route_idx]]], [[x_inlimit[grid_idx]], [y_inlimit[grid_idx]]],
             route_headings=[route_heading[route_idx]], grid_headings=[recovered_heading[grid_idx]],
             path=error_dir_path, save=True, show=False, title=fig_title)
    # Save every image in the window
    window = range(window_log[grid_idx][0], window_log[grid_idx][1])
    # Save world map with window route section.
    for idx in window:
        save_image(error_dir_path + 'window_' + str(idx) + '.png', route_images[idx])
    plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15), marker_size=100, scale=40,
             route_headings=route_heading, grid_headings=recovered_heading, show=False, save=True, save_id=grid_idx,
             window=window_log[grid_idx], path=error_dir_path)
    # Save a heat-map of the window similarity matrix
    fig = plt.figure(figsize=(40, 15))
    ax = sns.heatmap(logs[grid_idx], yticklabels=window)
    plt.xlabel('Degrees')
    plt.ylabel('Route memory index')
    ax.figure.savefig(error_dir_path + 'heat.png')
    plt.close()
