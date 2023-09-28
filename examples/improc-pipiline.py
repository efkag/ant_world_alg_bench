import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())

from source.utils import pre_process, save_image, check_for_dir_and_create
from source.imgproc import lin, glin, make_pipeline, Pipeline
from source.routedatabase import Route
from source.routedatabase import BoBRoute
import matplotlib.pyplot as plt


#route_path = '/home/efkag/ant_world_alg_bench/ftl/repeating-routes/route1/N-1'
#route_path = '/home/efkag/sussex-ftl-dataset/new-routes/ftl-1'
route_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes/route1/N-1'
#route = BoBRoute(path=route_path, read_imgs=True)


route_path = 'new-antworld/exp1/route1/'
route = Route(route_path, 1)
fig_save_path = os.path.join(fwd, 'figures')
check_for_dir_and_create(fig_save_path)
params = {#'blur': True,
        'shape': (360, 80),
        #'histeq': True, 
        #'edge_range': (180, 200),
        'gauss_loc_norm': {'sig1':2, 'sig2':20},
        #'vcrop':1.
        }
im_file = f'{str(params)}'
im_file = im_file.replace('{', '')
im_file = im_file.replace('}', '')
im_file = im_file + '.png'


imgs = route.get_imgs()
plt.imshow(imgs[10], cmap='gray')
plt.show()


imgs = pre_process(imgs, params)
plt.imshow(imgs[10], cmap='gray')
plt.show()

pipe = Pipeline(**params)
imgs = pipe.apply(route.get_imgs())

im_path = os.path.join(fig_save_path, im_file) 
save_image(im_path, imgs[10])
plt.imshow(imgs[10], cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig(im_path, bbox_inches='tight')
plt.show()

pipe = Pipeline()
imgs = pipe.apply(route.get_imgs())
plt.imshow(imgs[10], cmap='gray')
plt.show()