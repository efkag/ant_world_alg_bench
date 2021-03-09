from source.utils import pre_process, calc_dists, load_route_naw, angular_error, check_for_dir_and_create
from source import seqnav as spm, perfect_memory as pm
import pandas as pd
import time
import itertools
import multiprocessing
import functools
import numpy as np
from source import antworld2 as aw


def get_grid_dict(params):
    grid = itertools.product(*[params[k] for k in params])
    grid_dict = []
    for combo in grid:
        combo_dict = {}
        for i, k in enumerate(params):
            combo_dict[k] = combo[i]
        grid_dict.append(combo_dict)
    return grid_dict


# TODO: Params I will need: t? r?, random intial possition (sigma),
def bench(params, route_ids):
    log = {'route_id': [], 'blur': [], 'edge': [], 'res': [], 'window': [],
           'matcher': [], 'mean_error': [], 'seconds': [], 'errors': [],
           'dist_diff': [], 'abs_index_diff': [], 'window_log': [],
           'tx': [], 'ty': [], 'th': []}

    grid = get_grid_dict(params)
    #  Go though all combinations in the chunk
    for combo in grid:

        matcher = combo['matcher']
        window = combo['window']
        window_log = None
        for route_id in route_ids:  # for every route
            route_path = '../new-antworld/route' + str(route_id) + '/'
            route = load_route_naw(route_path, route_id=route_id, imgs=True)

            # Preprocess images
            route_imgs = route['imgs']
            route_imgs = pre_process(route_imgs, combo)
            # Run navigation algorithm
            if window:
                nav = spm.SequentialPerfectMemory(route_imgs, matcher, deg_range=(-180, 180), window=window)
            else:
                nav = pm.PerfectMemory(route_imgs, matcher, deg_range=(-180, 180))

            tic = time.perf_counter()
            traj, nav = aw.test_nav(route, nav, t=20, r=0.1, preproc=combo)
            toc = time.perf_counter()

            time_compl = toc - tic
            # Get the errors and the minimum distant index of the route memory
            errors, min_dist_index = angular_error(route, traj)
            # Difference between matched index and minimum distance index
            matched_index = nav.get_index_log()
            abs_index_diffs = np.absolute(np.subtract(matched_index, min_dist_index))
            dist_diff = calc_dists(route, min_dist_index, matched_index)
            mean_route_error = np.mean(errors)
            log['route_id'].extend([route_id])
            log['blur'].extend([combo.get('blur')])
            log['edge'].extend([combo.get('edge_range')])
            log['res'].append(combo.get('shape'))
            log['window'].extend([window])
            log['matcher'].extend([matcher])
            log['mean_error'].append(mean_route_error)
            log['seconds'].append(time_compl)
            log['window_log'].append(window_log)
            log['tx'].append(traj['x'].tolist())
            log['ty'].append(traj['y'].tolist())
            log['th'].append(traj['heading'])
            log['abs_index_diff'].append(abs_index_diffs.tolist())
            log['dist_diff'].append(dist_diff.tolist())
            log['errors'].append(errors)
        # Increment the complete jobs shared variable
    return log


def benchmark(results_path, params, route_ids):

    assert isinstance(params, dict)
    assert isinstance(route_ids, list)

    log = bench(params, route_ids)

    bench_results = pd.DataFrame(log)
    bench_results.to_csv(results_path, index=False)


# results_path = '../Results/newant/test.csv'
# parameters = {'blur': [True], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
#               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}
#
# benchmark(results_path, parameters, [1, 2])
