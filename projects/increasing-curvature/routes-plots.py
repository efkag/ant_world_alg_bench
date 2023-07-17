import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())


from source.routedatabase import Route, load_routes
from source.display import plot_route
from source.utils import meancurv2d

routes_path = '/its/home/sk526/ant_world_alg_bench/new-antworld/curve-bins'
route_ids = [5, 6, 7, 8, 9]
#route_ids = [*range(5)]
routes = load_routes(routes_path, route_ids)


for r in routes:
    xy = r.get_xycoords()
    k = meancurv2d(xy['x'], xy['y'])
    plot_route(r.get_route_dict(), size=(3, 3), title=f'k={k}')