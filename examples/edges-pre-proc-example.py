from source.utils import load_route, pre_process, r_cor_coef, display_image, rmf, cc
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use( 'tkagg' )

_, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                        route_heading, route_images = load_route(1)
img_indx = 0

original_img = route_images[img_indx]


fig = plt.figure()

logs = r_cor_coef(original_img, original_img, 360, 1)
plt.plot(range(len(logs)), logs, label='original')

low_bounds = list(range(150, 200, 20))
for bound in low_bounds:
    pre_proc = {'edge_range': (bound, bound+20)}
    keys = {'edge_range':0}
    edges_img = pre_process([route_images[img_indx]], pre_proc, keys)[0]
    logs = rmf(edges_img, edges_img, d_range=(-180, 180), d_step=1)

    plt.plot(range(len(logs)), logs, label=str(pre_proc['edge_range']))

plt.legend()
fig.savefig('test.png')
plt.show()
