from source2 import load_route_naw, pre_process, display_image, rmf, cor_dist
from matplotlib import pyplot as plt

# matplotlib.use( 'tkagg' )

# _, x_inlimit, y_inlimit, world_grid_imgs, x_route, y_route, \
#                         route_heading, route_images = load_route(1)
# original_img = route_images[img_indx]

route_id = 3
path = '../new-antworld/exp1/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True)
img_indx = 98

original_img = route['imgs'][img_indx]
display_image(original_img)
pre_proc = {'edge_range': (220, 240)}
edges_img = pre_process(original_img, pre_proc)
display_image(edges_img)


fig = plt.figure()

logs = rmf(original_img, original_img, matcher=cor_dist, d_range=(-180, 180), d_step=1)
plt.plot(range(len(logs)), logs, label='original')

low_bounds = list(range(150, 240, 20))
for bound in low_bounds:
    pre_proc = {'edge_range': (bound, bound+20)}
    edges_img = pre_process(original_img, pre_proc)
    logs = rmf(edges_img, edges_img, d_range=(-180, 180), d_step=1)

    plt.plot(range(len(logs)), logs, label=str(pre_proc['edge_range']))

plt.legend()
fig.savefig('test.png')
plt.show()
