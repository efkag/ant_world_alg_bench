from utils import load_grid, plot_map, gen_route_line, pol_2cart_headings, route_imgs_from_indexes, check_for_dir_and_create
import pandas as pd
import numpy as np
# right = 105
# left = -105
# up = 1
# down = -1
# up_r = right + up
# up_l = left + up
# down_r = right + down
# down_l = left + down

x, y, w = load_grid()

# ---------------- Amend code here below to change route
step = 105
start = 800
stop = 3000

headings = []
indexes = list(range(start, stop, step))
headings.extend([0] * len(indexes))

indexes = gen_route_line(indexes, 'up_r', 10)
headings.extend([45] * 10)

indexes = gen_route_line(indexes, 'right', 10)
headings.extend([0] * 10)

# indexes = gen_route_line(indexes, 'up', 4)
# headings.extend([90] * 4)
#
# indexes = gen_route_line(indexes, 'up_l', 4)
# headings.extend([135] * 4)
#
# indexes = gen_route_line(indexes, 'left', 4)
# headings.extend([180] * 4)
#
# indexes = gen_route_line(indexes, 'down_l', 4)
# headings.extend([225] * 4)
#
# indexes = gen_route_line(indexes, 'down', 4)
# headings.extend([270] * 4)
#
# indexes = gen_route_line(indexes, 'down_r', 8)
# headings.extend([315] * 8)

# ------------------Amend code here above to change route

# Remove the first heading and duplicate the last one
headings = headings[1:]
headings.append(headings[-1])

route_x = x[indexes]
route_y = y[indexes]
route_data = {'X': route_x, 'Y': route_y, 'Z':[10]*len(route_x), 'Heading': headings,
              'Filename':[str(i)+'.png' for i in range(0, len(route_x))]}
route = pd.DataFrame(route_data)
# Rearange column order
route = route[['X', 'Y', 'Z', 'Heading', 'Filename']]

directory = 'LoopRoutes/route_1/'
check_for_dir_and_create(directory)
route_imgs = route_imgs_from_indexes(indexes, headings, directory)
route.to_csv(directory + 'AntLoop.csv')


u, v = pol_2cart_headings(headings)
plot_map(w, route_cords=[route_x, route_y], vectors=[u, v])
