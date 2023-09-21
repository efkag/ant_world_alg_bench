import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

from source.utils import pre_process
from source.imgproc import lin, glin, make_pipeline, Pipeline
from source.routedatabase import Route
from source.routedatabase import BoBRoute
import matplotlib.pyplot as plt


route_path = 'new-antworld/exp1/route1/'
route = Route(route_path, 1)

params = {'blur': True,
        'shape': (180, 35),
        'histeq': True,
        #'edge_range': (180, 230),
        'gauss_loc_norm': {'sig1':2, 'sig2':20},
        # 'loc_norm': {'kernel_shape':(3, 3)},
        #'vcrop':1.
        }



imgs = route.get_imgs()
plt.imshow(imgs[10], cmap='gray')
plt.show()


# imgs = pre_process(imgs, params)
# plt.imshow(imgs[10], cmap='gray')
# plt.show()

pipe = Pipeline(**params)
imgs = pipe.apply(route.get_imgs())
plt.imshow(imgs[10], cmap='gray')
plt.show()

pipe = Pipeline()
imgs = pipe.apply(route.get_imgs())
plt.imshow(imgs[10], cmap='gray')
plt.show()



#route_path = 'test-routes/FTLroutes/N-1-01'
route_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes/route1/N-1'
#route_path = '/home/efkag/sussex-ftl-dataset/new-routes/ftl-1'

route = BoBRoute(path=route_path, read_imgs=True)

imgs = route.get_imgs()
plt.imshow(imgs[10], cmap='gray')
plt.show()

pipe = Pipeline(**params)
imgs = pipe.apply(route.get_imgs())
plt.imshow(imgs[10], cmap='gray')
plt.show()