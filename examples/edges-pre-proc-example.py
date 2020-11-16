from source.utils import load_route, pre_process, r_cor_coef, ridf, display_image
import matplotlib.pyplot as plt

_, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
                        route_heading, route_images = load_route(1)
img_indx = 0

original_img = route_images[img_indx]


fig = plt.figure()

logs = r_cor_coef(original_img, original_img, 360, 1)
plt.plot(range(len(logs)), logs, label='original')

low_bounds = list(range(100, 200, 20))
for bound in low_bounds:
    pre_proc = {'edge_range': (bound, bound+20)}
    edges_img = pre_process([route_images[img_indx]], pre_proc)[0]
    logs = r_cor_coef(edges_img, edges_img, 360, 1)

    plt.plot(range(len(logs)), logs, label=str(pre_proc['edge_range']))

plt.legend()
fig.savefig('test.png')
plt.show()
