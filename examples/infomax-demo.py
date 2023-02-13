import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import numpy as np
import torch
from source.routedatabase import Route
from source import infomax
from source import imgproc


route_path = 'new-antworld/exp1/route1/'
route = Route(route_path, 1)

imgs = route.get_imgs()
imgs = [torch.from_numpy(item).float() for item in imgs]

# flatten the img to get the shape of it
input_size = imgs[0].flatten().size(0)
infomaxParams = infomax.Params()

infomaxnet = infomax.InfomaxNetwork(input_size, infomaxParams=infomaxParams)

infomaxnet.TrainNet(imgs)