import sys
import os
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
from source.utils import mae, rmf, cor_dist, save_image, rotate, check_for_dir_and_create
from source.routedatabase import Route
from source.imgproc import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
sns.set_context("paper", font_scale=1)

route_path = 'datasets/new-antworld/exp1/route1/'
route = Route(route_path, 1)

save_path = os.path.join(fwd, 'rmf-curves')
check_for_dir_and_create(save_path)

img = route.get_imgs()[0]


params = {'blur': True,
        'shape': (360, 80), 
        }
pipe = Pipeline(**params)
img = pipe.apply(img)

#save processed image
img_path = os.path.join(save_path, 'proc-img.png')
cv.imwrite(img_path, img)

deg_range = (-180, 180)
rmf_curve = rmf(img, img, matcher=mae, d_range=deg_range)
degrees = np.arange(*deg_range)

figsize = (6, 4)
fig = plt.figure(figsize=figsize)
plt.scatter(degrees, rmf_curve, s=8)
plt.plot(degrees, rmf_curve)
plt.xlabel('Degrees')
plt.ylabel('IDF AbsDiff')
plt.tight_layout()
fig_save_path = os.path.join(save_path, 'rmf-mae.png')
fig.savefig(fig_save_path)


