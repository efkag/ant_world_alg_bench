import sys
import os
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
from scipy.stats import entropy
from source.utils import mae, rmf, cor_dist, cos_dist, dot_dist, mutual_inf, entropy_dist, check_for_dir_and_create
from source.routedatabase import Route
from source.imgproc import Pipeline
from source.display import plot_3d
import seaborn as sns
import matplotlib.pyplot as plt
from time import perf_counter
sns.set_context("paper", font_scale=1)

route_path = 'datasets/new-antworld/exp1/route1/'
#route_path = 'test-routes/route1'
route = Route(route_path, 1)

save_path = os.path.join(fwd, 'rmf-surfaces')
check_for_dir_and_create(save_path)

q_img_idx = 50
search_margin = 30
q_img = route.get_imgs()[q_img_idx]
ref_imgs = route.get_imgs()[q_img_idx - search_margin:q_img_idx + search_margin]

params = {'blur': True,
        'shape': (180, 40), 
        }
pipe = Pipeline(**params)
q_img = pipe.apply(q_img)
ref_imgs = pipe.apply(ref_imgs)


q_img_hist = np.histogram(q_img.flatten(), bins=256, density=True)[0]

ref_imgs_hist = [np.histogram(im.flatten(),bins=256, density=True)[0] for im in ref_imgs]

klds = [entropy(q_img_hist, im_hist) for im_hist in ref_imgs_hist]
print(klds)
print(np.argmin(klds))

plt.plot(klds)
plt.show()