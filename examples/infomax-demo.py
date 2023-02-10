import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

import torch
from source.routedatabase import Route
from source import infomax


route_path = 'new-antworld/exp1/route1/'
route = Route(route_path, 1)

imgs = route.get_imgs()
imgs = [torch.from_numpy(item).float() for item in imgs]

network_size = 100
infomaxParams = infomax.Params()

infomaxnet = infomax.InfomaxNetwork(network_size, infomaxParams=infomaxParams)

infomaxnet.TrainNet(imgs)