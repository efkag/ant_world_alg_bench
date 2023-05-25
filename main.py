from source import cbench
from datetime import date
today = date.today()
string_date = today.strftime("%Y-%m-%d")
from source import navbench


def static_bench():
    results_path = f'/its/home/sk526/ant_world_alg_bench/Results/ftl/{string_date}'
    routes_path = '/its/home/sk526/sussex-ftl-dataset/repeating-routes'
    # grid_path = '/home/efkag/PycharmProjects/ant_world_alg_bench/new-antworld/grid70'
    # parameters = {'blur': [True], 'segment_l': [3], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}
    
    parameters = {'blur': [True, False], 
                  'shape': [(180, 80)],
                  'vcrop':[.6],
                  'edge_range': [(180, 200), False],
                  'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                  'deg_range':(-180, 180),
                  'window': [0, 15, 20, 25, -15], 
                  'matcher': ['mae', 'corr']}
    
    routes = [1, 2, 3]
    bench = navbench.Benchmark(results_path, routes_path, 
                               grid_path=None, 
                               filename='results.csv',
                               route_path_suffix='N-',
                               route_repeats=5)
    bench.benchmark(parameters, routes, parallel=True, cores=1)


def live_bench():
    #'segment_length':[3],
    results_path = f'/its/home/sk526/ant_world_alg_bench/Results/newant/{string_date}_test_pm part'
    routes_path = '/its/home/sk526/ant_world_alg_bench/new-antworld/curve-bins'
    parameters = {'repos_thresh':[.3], 
                  'r': [0.05], 
                  't': [300], 
                  'blur': [True],
                  'shape': [(180, 80)],
                #  'wave' : [True, False], 
                #  'edge_range': [(180, 200), False],
                #  'loc_norm': [{'kernel_shape':(5, 5)}, False],
                  'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                  'window': [0],
                  'matcher': ['corr'],
                  }

    routes = [*range(20)]
    num_of_repeats = 3
    parameters['repeat'] = [*range(num_of_repeats)]
    cbench.benchmark(results_path, routes_path, parameters, routes, 
                    parallel=True, num_of_repeats=num_of_repeats)


def main():
    static_bench()
    #live_bench()

if __name__ == "__main__":
    main()
