import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())
import time


import numpy as np
import torch
from source.routedatabase import Route
from source import infomax
from source.imgproc import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale=1)


route_path = 'new-antworld/exp1/route1/'
route = Route(route_path, 1)

imgs = route.get_imgs()
deg_range = (-180, 180)
params = {'blur': True,
        'shape': (180, 80), 
        }
pipe = Pipeline(**params)
imgs = pipe.apply(route.get_imgs())

# keep one numpy image for testing
img_array = imgs[0]

infomaxParams = infomax.Params()
tic = time.perf_counter()
infomaxnet = infomax.InfomaxNetwork(infomaxParams, imgs, deg_range=deg_range)
dt = time.perf_counter() - tic
print(f'it took {dt} seconds to train infomax')
# only to be used if further training is required
#infomaxnet.TrainNet(imgs)
tic = time.perf_counter()
rsim = infomaxnet.get_rsim(img_array)
dt = time.perf_counter() - tic
print(f'it took {dt} seconds to eval. one image RIDF with infomax')

plt.plot(range(*deg_range), rsim)
plt.show()


#### see how the image looks
query_img = torch.unsqueeze(torch.from_numpy(img_array).float(), 0)
query_img = infomaxnet.Standardize(query_img)
out = infomaxnet.Forward(query_img)
out = torch.reshape(out, params['shape'])

out = out.squeeze().detach().numpy()

plt.imshow(out)
plt.show()