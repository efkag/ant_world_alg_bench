import sys
import os
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
from source.utils import mae, rmf, cor_dist, cos_dist, dot_dist, mutual_inf, entropy_dist, check_for_dir_and_create
from source.routedatabase import Route
from source.imageproc.imgproc import Pipeline
from source.display import plot_3d
import seaborn as sns
import matplotlib.pyplot as plt
from time import perf_counter
sns.set_context("paper", font_scale=1)

route_path = 'new-antworld/exp1/route1/'
route_path = 'test-routes/route1'
route = Route(route_path, 1)

save_path = os.path.join(fwd, 'rmf-surfaces')
check_for_dir_and_create(save_path)

q_img_idx = 50
search_margin = 30
q_img = route.get_imgs()[q_img_idx]
ref_imgs = route.get_imgs()[q_img_idx - search_margin:q_img_idx + search_margin]

matcher = cos_dist
params = {'blur': True,
        'shape': (180, 40),
        'normstd': True, 
        }
pipe = Pipeline(**params)
q_img = pipe.apply(q_img)
ref_imgs = pipe.apply(ref_imgs)

# #save processed image
# img_path = os.path.join(save_path, 'proc-img.png')
# cv.imwrite(img_path, img)

deg_range = (-180, 180)
tic = perf_counter()
#mae_dist_surf = rmf(q_img, ref_imgs, matcher=mae, d_range=deg_range)
#entropy_dist_surf = (rmf(q_img, ref_imgs, matcher=entropy_dist, d_range=deg_range))
corr_dist_surf = (rmf(q_img, ref_imgs, matcher=matcher, d_range=deg_range))

print('time elapsed in sec. ', perf_counter()-tic)
# degrees array
degrees = np.arange(*deg_range)
fig_path = os.path.join(save_path, 'cor_dist_surf.png')
plot_3d(corr_dist_surf, save=True, path=fig_path)