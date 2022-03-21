import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

from source.utils import pre_process
from source.imgproc import lin, glin, make_pipeline, Pipeline
from source.routedatabase import Route
import matplotlib.pyplot as plt


route_path = 'new-antworld/exp1/route1/'
route = Route(route_path, 1)

params = {'blur': True,
        'shape': (180, 50), 
        #'edge_range': (180, 200),
        'gauss_loc_norm': {'sig1':2, 'sig2':20}
        }

imgs = route.get_imgs()
imgs = pre_process(imgs, params)

plt.imshow(imgs[10], cmap='gray')
plt.show()

pipe = Pipeline(**params)
imgs = pipe.apply(route.get_imgs())

plt.imshow(imgs[10], cmap='gray')
plt.show()
