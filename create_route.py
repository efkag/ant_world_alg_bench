from utils import load_grid, plot_map, gen_route_line
# right = 105
# left = -105
# up = 1
# down = -1
# up_r = right + up
# up_l = left + up
# down_r = right + down
# down_l = left + down

x, y, w = load_grid()
#
# lower = 1100
# upper = 1206
#
# plot_map(w, grid_cords=[x[lower:upper], y[lower:upper]])


step = 105
start = 800
stop = 3200

indexes = list(range(start, stop, step))

# for i in range(4):
#     indexes.append(indexes[-1] + 106)

indexes = gen_route_line(indexes, 'up_r', 4)

indexes = gen_route_line(indexes, 'up', 4)

indexes = gen_route_line(indexes, 'up_l', 4)

indexes = gen_route_line(indexes, 'left', 4)

indexes = gen_route_line(indexes, 'down_l', 4)

indexes = gen_route_line(indexes, 'down', 4)

indexes = gen_route_line(indexes, 'down_r', 8)



route_x = x[indexes]
route_y = y[indexes]




plot_map(w, grid_cords=[route_x, route_y])
