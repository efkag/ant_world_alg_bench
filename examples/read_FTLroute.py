import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

from source.routedatabase import BoBRoute
from source.unwraper import Unwraper
from source.imageproc.imgproc import Pipeline
from matplotlib import pyplot as plt
from source.display import plot_ftl_route

route_path = 'test-routes/FTLroutes/N-1-01'
#route_path = '/home/efkag/ant_world_alg_bench/ftl/repeating-routes/route1/N-1'
#route_path = '/home/efkag/sussex-ftl-dataset/new-routes/ftl-1'

route = BoBRoute(path=route_path, read_imgs=True, unwraper=Unwraper)
im_shape = route.img_shape
route_dict = route.get_route_dict()

combo = {'shape':(180, 80),'vcrop':0.6}
pipe = Pipeline(**combo)

imgs = route_dict['imgs']
imgs = pipe.apply(imgs)

im =  imgs[0]
plt.imshow(im, cmap='gray')
plt.show()

# for i in range(0, 200, 5):
#     plt.imshow(imgs[i], cmap='gray')
#     plt.show() 

plot_ftl_route(route_dict)