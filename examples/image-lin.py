import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())


from source.imgproc import lin, glin
from source.routedatabase import Route
import matplotlib.pyplot as plt


route_path = 'new-antworld/exp1/route1/'
route = Route(route_path, 1)

img = route.get_imgs()[0]
img = lin(img, (10, 10))

plt.imshow(img, cmap='gray')
plt.show()


img = route.get_imgs()[0]
img = glin(img, sig1=2, sig2=15)

plt.imshow(img, cmap='gray')
plt.show()




