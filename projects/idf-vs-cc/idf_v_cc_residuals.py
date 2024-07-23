import sys
import os

# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import cv2 as cv
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from source.imageproc.imgproc import Pipeline
from source.utils import rmf, cor_dist, mae, rotate, check_for_dir_and_create
from source.routedatabase import Route
from source.display import heat_with_marginals


fig_save_path = os.path.join(fwd, 'figures')
check_for_dir_and_create(fig_save_path)

route_id = 1
path = 'new-antworld/exp1/route' + str(route_id) + '/'
route = Route(path, route_id=route_id)
route_imgs = route.get_imgs()


img = route_imgs[92]
cv.imwrite(os.path.join(fig_save_path, "orignial.png"), img)

params = {'blur': True,
        'shape': (180, 80), 
        #'edge_range': (180, 200)
        }
pipe = Pipeline(**params)
img = pipe.apply(img)
cv.imwrite(os.path.join(fig_save_path, "proc-orignial.png"), img)

def mae_residual_image(ref_img, current_img, d):
    current_img = rotate(d, current_img)
    return np.absolute(ref_img - current_img)

    # ax.figure.savefig('idf_v_correlation_figures/residual image ' + str(d) + '.png')
    # save_image('idf_v_correlation_figures/residual image ' + str(d) + '.png', res_img)


def cc_residual_image(a, b, d):
    current_img = rotate(d, b)
    amu = np.mean(a)
    bmu = np.mean(b)
    a = a - amu
    b = b - bmu
    ab = a * b
    avar = np.mean(np.square(a))
    bvar = np.mean(np.square(b))
    return ab / np.sqrt(avar * bvar)



logs = rmf(img, img, matcher=mae, d_range=(-180, 180), d_step=1)
min_h = np.argmin(logs)
print(min_h)
fig = plt.figure()
plt.plot(range(len(logs)), logs)
fig.suptitle('RIDF')
plt.xlabel('Degrees')
plt.ylabel('IDF')
fig.savefig(os.path.join(fig_save_path, 'ridf.png'))
plt.close()
#plt.show()
# Save residual images

rimg = mae_residual_image(img, img, min_h)
fig = heat_with_marginals(rimg)
fig.savefig(os.path.join(fig_save_path, 'idf_res_img.png'))
#plt.show()


logs = rmf(img, img, matcher=cor_dist, d_range=(-180, 180), d_step=1)
min_h = np.argmin(logs)
print(min_h)
fig = plt.figure()
plt.plot(range(len(logs)), logs)
fig.suptitle('CC')
plt.xlabel('Degrees')
plt.ylabel('CC')
fig.savefig(os.path.join(fig_save_path, 'cc.png'))
#plt.show()
# Save residual images
rimg = cc_residual_image(img, img, min_h)
fig = heat_with_marginals(rimg)
fig.savefig(os.path.join(fig_save_path, 'cc_res_img.png'))
