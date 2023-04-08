import sys
import os
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
from source.utils import mae, cor_dist, entrop_dist, rmf, cor_dist, scale2_0_1, save_image, rotate, check_for_dir_and_create
from source.routedatabase import Route
from source.imgproc import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
sns.set_context("paper", font_scale=1)

route_path = 'new-antworld/exp1/route1/'
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
entropy_dist_curve = scale2_0_1(rmf(img, img, matcher=entrop_dist, d_range=deg_range))
corr_dist_curve = scale2_0_1(rmf(img, img, matcher=cor_dist, d_range=deg_range))
mae_dist_curve = scale2_0_1(rmf(img, img, matcher=mae, d_range=deg_range))
degrees = np.arange(*deg_range)

figsize = (6, 4)
fig = plt.figure(figsize=figsize)
plt.scatter(degrees, entropy_dist_curve, s=8)
plt.plot(degrees, entropy_dist_curve, label='entropy')

plt.scatter(degrees, corr_dist_curve, s=8)
plt.plot(degrees, corr_dist_curve, label='correlation')

plt.scatter(degrees, mae_dist_curve, s=8)
plt.plot(degrees, mae_dist_curve, label='mae')

plt.xlabel('Degrees')
plt.ylabel('Matcher (scaled to [0, 1])')
plt.legend()
plt.tight_layout()
# fig_save_path = os.path.join(save_path, 'rmf-mae.png')
# fig.savefig(fig_save_path)
plt.show()