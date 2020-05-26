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
route_id = 3

# ---------------- Amend code here below to change route
step = 105
start = 1635
stop = 3500

headings = []
indexes = list(range(start, stop, step))
headings.extend([0] * len(indexes))

indexes, headings = gen_route_line(indexes, headings, 'down_r', 6)

indexes, headings = gen_route_line(indexes, headings, 'right', 8)

indexes, headings = gen_route_line(indexes, headings, 'up_r', 10)

indexes, headings = gen_route_line(indexes, headings, 'up', 10)

indexes, headings = gen_route_line(indexes, headings, 'up_l', 8)

indexes, headings = gen_route_line(indexes, headings, 'left', 6)

indexes, headings = gen_route_line(indexes, headings, 'down_l', 6)

indexes, headings = gen_route_line(indexes, headings, 'down', 4)

indexes, headings = gen_route_line(indexes, headings, 'down_r', 5)

indexes, headings = gen_route_line(indexes, headings, 'down', 4)

indexes, headings = gen_route_line(indexes, headings, 'down_r', 15)

indexes, headings = gen_route_line(indexes, headings, 'right', 5)


# ------------------Amend code here above to change route

# Remove the first heading and duplicate the last one
headings = headings[1:]
headings.append(headings[-1])

route_x = x[indexes]
route_y = y[indexes]
route_data = {'X': route_x, 'Y': route_y, 'Z':[10]*len(route_x), 'Heading': headings,
              'Filename': [str(i)+'.png' for i in range(0, len(route_x))]}
route = pd.DataFrame(route_data)
# Rearange column order
route = route[['X', 'Y', 'Z', 'Heading', 'Filename']]

route_dir = 'route_' + str(route_id)
directory = 'LoopRoutes/' + route_dir + '/'
check_for_dir_and_create(directory)
route_imgs = route_imgs_from_indexes(indexes, headings, directory)
route.to_csv(directory + route_dir + '.csv', index=False)


plot_map(w, route_cords=[route_x, route_y], route_headings=headings)
