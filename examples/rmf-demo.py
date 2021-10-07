import numpy as np
from source.utils import mae, rmf, cor_dist, save_image, rotate
from source.routedatabase import Route
import matplotlib.pyplot as plt

route_path = '../new-antworld/exp1/route1/'
route = Route(route_path, 1)


img = route.get_imgs()[0]


deg_range = (-180, 180)
degrees = np.arange(*deg_range)
sims = []
for i, r in enumerate(degrees):

    fig = plt.figure(figsize=(10, 7))
    plt.suptitle('deg={}'.format(r))
    rows = 3
    cols = 1
    ax = fig.add_subplot(rows, cols, 1)
    ax.set_title('train')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 2)
    ax.set_title('test')
    plt.imshow(rotate(r, img), cmap='gray')
    plt.axis('off')

    ax = fig.add_subplot(rows, cols, 3)
    ax.set_title('RMF')
    sims.append(mae(img, rotate(r, img)))
    plt.plot(degrees[:i+1], sims)
    plt.xlim(-180, 180)

    fig.savefig('{}.png'.format(r+180))
    plt.close(fig)


