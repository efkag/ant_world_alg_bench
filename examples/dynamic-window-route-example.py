from source.utils import pre_process, mean_degree_error, load_route, check_for_dir_and_create, plot_map
from source import seqnav as spm
from matplotlib import pyplot as plt

matcher = 'rmse'
pre_proc = {'blur': True, 'shape': (180, 50)}
window = 15
figures_path = 'Figures/'
dist = 100
check_for_dir_and_create(figures_path)

w, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                            route_heading, route_images = load_route(route_id=10, grid_pos_limit=dist)
print(len(x_inlimit))

plot_map(w, [x_route, y_route], [x_inlimit, y_inlimit], size=(15, 15),
         route_headings=route_heading, scale=40)

pre_world_grid_imgs = pre_process(world_grid_imgs, pre_proc)
pre_route_images = pre_process(route_images, pre_proc)

nav = spm.Seq2SeqPerfectMemory(pre_route_images, matcher)
recovered_heading, logs, window_log = nav.navigate(pre_world_grid_imgs, window)
window_sims = nav.get_window_sims()
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

confidence = nav.get_confidence()

plt.plot(range(len(confidence)), confidence)
plt.show()

window_size = [w[1] - w[0] for w in window_log]
print("average window size: " + str(sum(window_size)/len(window_size)))

plt.bar(range(len(window_size)), window_size)
plt.show()

for i in range(len(window_sims)):
    plt.plot(range(window_log[i][0], window_log[i][1]), window_sims[i])
plt.show()

cma = nav.get_CMA()
plt.plot(range(len(cma)), cma)
plt.show()
