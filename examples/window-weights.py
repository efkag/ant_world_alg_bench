import sys
import os
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
from source.utils import mae, rmf, cor_dist, cos_dist, dot_dist, mutual_inf, entropy_dist, check_for_dir_and_create
from source.routedatabase import Route
from source.imgproc import Pipeline
from source.display import plot_3d
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import perf_counter
sns.set_context("paper", font_scale=1)

route_path = 'datasets/new-antworld/exp1/route1/'

route = Route(route_path, 1)

save_path = os.path.join(fwd, 'rmf-surfaces')
check_for_dir_and_create(save_path)

q_img_idx = 50
search_margin = 30
q_img = route.get_imgs()[q_img_idx]
ref_imgs = route.get_imgs()[q_img_idx - search_margin:q_img_idx + search_margin]

matcher = mae
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
mae_dist_surf = rmf(q_img, ref_imgs, matcher=mae, d_range=deg_range)
#corr_dist_surf = (rmf(q_img, ref_imgs, matcher=matcher, d_range=deg_range))

ridf_mins = np.min(mae_dist_surf, axis=1)

##artificialy distort
ridf_mins = ridf_mins + 1.5 
ridf_mins[10:15] = ridf_mins[10:15] - 11.

w = 20
mu = 0
sig = 1
rv = norm(loc=mu, scale=sig)
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), len(ridf_mins))
pdf = rv.pdf(x)
weights = 1 - pdf

ridf_mins_w = ridf_mins * weights
idx = np.argmin(ridf_mins_w)

plt.plot(ridf_mins, label='raw')
plt.plot(ridf_mins_w, label='weighted')
plt.scatter([idx], [ridf_mins_w[idx]])
plt.plot(weights)
plt.legend()
plt.show()




ws = [60]
for w in ws:
    x = np.linspace(norm.ppf(0.01),
                    norm.ppf(0.99), w)
    pdf = rv.pdf(x)
    pdf = 1 - pdf
    plt.plot(range(w), pdf)
    plt.scatter(range(w), pdf)


#plt.show()