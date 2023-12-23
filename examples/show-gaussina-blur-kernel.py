import sys
import os
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
from source.utils import check_for_dir_and_create
from source.routedatabase import Route
import seaborn as sns
import matplotlib.pyplot as plt
import cv2 as cv
sns.set_context("paper", font_scale=1)

route_path = 'datasets/new-antworld/exp1/route1/'
route = Route(route_path, 1)

save_path = os.path.join(fwd, 'rmf-curves')
check_for_dir_and_create(save_path)

q_img_idx = 15
search_margin = 10
q_img = route.get_imgs()[q_img_idx]
ref_imgs = route.get_imgs()[q_img_idx - search_margin:q_img_idx + search_margin]

blank = np.zeros_like(q_img)
blank = np.zeros((61, 61))
shape = blank.shape
x = shape[0] // 2
y = shape[1] // 2

blank[x, y] = 1

# plt.imshow(blank, cmap='hot')
# plt.show()
k = cv.GaussianBlur(blank, (0,0), 20)
print(k)
plt.imshow(k, cmap='hot')
plt.show()

k = cv.getGaussianKernel(11, 2)
print(k)
# plt.imshow(k, cmap='hot')
# plt.show()