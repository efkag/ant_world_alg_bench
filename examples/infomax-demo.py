import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
import torch
from source.routedatabase import Route
from source import infomax
from source.imgproc import Pipeline
import matplotlib.pyplot as plt


route_path = 'new-antworld/exp1/route1/'
route = Route(route_path, 1)

imgs = route.get_imgs()
params = {'blur': True,
        'shape': (180, 80), 
        }
pipe = Pipeline(**params)
imgs = pipe.apply(route.get_imgs())

# keep one numpy image for testing
img_array = imgs[0]

infomaxParams = infomax.Params()

infomaxnet = infomax.InfomaxNetwork(infomaxParams, imgs)

# only to be used if further training is required
#infomaxnet.TrainNet(imgs)

rsim = infomaxnet.get_heading(img_array)

plt.plot(range(len(rsim)), rsim)
plt.show()