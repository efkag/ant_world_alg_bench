import sys
import os
# path = os.path.join(os.path.dirname(__file__), os.pardir)
fwd = os.path.dirname(__file__)
sys.path.append(os.getcwd())
import yaml
import numpy as np
import pandas as pd

from source.routedatabase import Route, load_routes
from source.display import plot_route
from source.utils import meancurv2d, curv2d

routes_path = 'datasets/new-antworld/curve-bins'
route_ids = [*range(20)]
#route_ids = [*range(5)]
routes = load_routes(routes_path, route_ids)

all_k_data = {'route_id':[], 'k':[], 'mean_k':[], 'std_k':[]}

for r in routes:
    metadata = {}
    xy = r.get_xycoords()
    k_vec = curv2d(xy['x'], xy['y'])
    mean_k = np.mean(k_vec).item()
    std_k = np.std(k_vec).item()
    metadata['k'] = np.round(k_vec, 5).tolist()
    metadata['mean_k'] = mean_k
    metadata['std_k'] = std_k
    data_path = os.path.join(routes_path, f'route{r.get_route_id()}', 'metadata.yml')
    with open(data_path, 'w') as fp:
        yaml.dump(metadata, fp)
    
    all_k_data['route_id'].append(r.get_route_id())
    all_k_data['k'].append(np.round(k_vec, 5).tolist())
    all_k_data['mean_k'].append(mean_k)
    all_k_data['std_k'].append(std_k)

    #plot_route(r.get_route_dict(), size=(10, 10), title=f'k={round(mean_k, 5)}')
df = pd.DataFrame(all_k_data)
save_path = os.path.join(fwd, 'curvatures.csv')
df.to_csv(save_path, index=False)
