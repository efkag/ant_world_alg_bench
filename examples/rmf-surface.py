import sys
import os
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
from source.utils import mae, rmf, cor_dist,mutual_inf, entropy_dist, check_for_dir_and_create
from source.routedatabase import Route
from source.imgproc import Pipeline
from source.display import plot_3d
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
sns.set_context("paper", font_scale=1)

route_path = 'new-antworld/exp1/route1/'
route = Route(route_path, 1)

save_path = os.path.join(fwd, 'rmf-curves')
check_for_dir_and_create(save_path)

q_img_idx = 15
search_margin = 10
q_img = route.get_imgs()[q_img_idx]
ref_imgs = route.get_imgs()[q_img_idx - search_margin:q_img_idx + search_margin]


params = {'blur': True,
        'shape': (180, 40), 
        }
pipe = Pipeline(**params)
q_img = pipe.apply(q_img)
ref_imgs = pipe.apply(ref_imgs)

# #save processed image
# img_path = os.path.join(save_path, 'proc-img.png')
# cv.imwrite(img_path, img)

deg_range = (-180, 180)
#mae_dist_surf = rmf(q_img, ref_imgs, matcher=mae, d_range=deg_range)
entropy_dist_surf = (rmf(q_img, ref_imgs, matcher=entropy_dist, d_range=deg_range))
#corr_dist_surf = (rmf(q_img, ref_imgs, matcher=cor_dist, d_range=deg_range))
# degrees array
degrees = np.arange(*deg_range)
save_path = os.path.join(fwd, 'rmf-surfaces', 'entropy_dist_surf.png')
plot_3d(entropy_dist_surf, save=True, path=save_path)