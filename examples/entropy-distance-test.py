import sys
import os
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
from source.utils import mae, cor_dist, entropy_im, mutual_inf, entropy_dist, rmf, cor_dist, scale2_0_1, save_image, rotate, check_for_dir_and_create
from source.routedatabase import Route
from source.imgproc import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
sns.set_context("paper", font_scale=1)

route_path = 'new-antworld/exp1/route1/'
# for local machine
# route_path = 'test-routes/route1'
route = Route(route_path, 1)

save_path = os.path.join(fwd, 'rmf-curves')
check_for_dir_and_create(save_path)

img_idx = 91
img = route.get_imgs()[91]

params = {'blur': True,
        'shape': (180, 40), 
        }
pipe = Pipeline(**params)
img = pipe.apply(img)

deg_range = (-180, 180)

entropy_dist_curve = scale2_0_1(rmf(img, img, matcher=entropy_dist, d_range=deg_range))
corr_dist_curve = scale2_0_1(rmf(img, img, matcher=cor_dist, d_range=deg_range))
mae_dist_curve = scale2_0_1(rmf(img, img, matcher=mae, d_range=deg_range))
mi_curve = scale2_0_1(rmf(img, img, matcher=mutual_inf, d_range=deg_range))
degrees = np.arange(*deg_range)
# im_entr = [entropy_im(rotate(r, img)) for r in degrees]


figsize = (6, 4)
fig = plt.figure(figsize=figsize)
plt.scatter(degrees, entropy_dist_curve, s=8)
plt.plot(degrees, entropy_dist_curve, label='entropy dist')

plt.scatter(degrees, corr_dist_curve, s=8)
plt.plot(degrees, corr_dist_curve, label='correlation dist')

plt.scatter(degrees, mae_dist_curve, s=8)
plt.plot(degrees, mae_dist_curve, label='mae')

# plt.scatter(degrees, mi_curve, s=8)
# plt.plot(degrees, mi_curve, label='mi')

plt.xlabel('Degrees')
plt.ylabel('Matcher (scaled to [0, 1])')
plt.legend()
plt.tight_layout()
# fig_save_path = os.path.join(save_path, 'rmf-mae.png')
# fig.savefig(fig_save_path)
plt.show()