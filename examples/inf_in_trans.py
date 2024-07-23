import sys
import os
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
from scipy.stats import entropy
from source.utils import mae, rmf, cor_dist, cos_dist, dot_dist, mutual_inf, entropy_dist, check_for_dir_and_create
from source.routedatabase import Route, BoBRoute
from source.imgproc import Pipeline
from source.display import plot_3d
import seaborn as sns
import matplotlib.pyplot as plt
from time import perf_counter
sns.set_context("paper", font_scale=1)

# route_path = 'datasets/new-antworld/exp1/route1/'
# #route_path = 'test-routes/route1'
# route = Route(route_path, 1)

route_path = '/home/efkag/navlib/data/campus/route1/r1'
route = BoBRoute(path=route_path, read_imgs=True)

save_path = os.path.join(fwd, 'rmf-surfaces')
check_for_dir_and_create(save_path)

bins = 25
q_img_idx = 300
search_margin = 50
q_img = route.get_imgs()[q_img_idx]
ref_imgs = route.get_imgs()[q_img_idx - search_margin:q_img_idx + search_margin]

params = {'blur': True,
        'shape': (180, 40), 
        'gauss_loc_norm': {'sig1':2, 'sig2':20},
        }
pipe = Pipeline(**params)
q_img = pipe.apply(q_img)
ref_imgs = pipe.apply(ref_imgs)
ref_imgs = np.asarray(ref_imgs)

bins = np.histogram_bin_edges(ref_imgs.flatten(), bins=bins)

q_img_hist = np.histogram(q_img.flatten(), bins=bins, density=True)[0]

plt.hist(q_img.flatten(), bins=bins, density=False)
plt.show()

ref_imgs_hist = [np.histogram(im.flatten(),bins=bins, density=True)[0] for im in ref_imgs]

klds = [entropy(q_img_hist, im_hist) for im_hist in ref_imgs_hist]
print(klds)
print(np.argmin(klds))

plt.plot(klds)
plt.show()