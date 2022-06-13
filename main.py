# from numpy import core
from source import cbench
# import navbench

def main():
    # results_path = '../Results/newant/'
    # routes_path = '../new-antworld/exp1/'
    # grid_path = '/home/efkag/PycharmProjects/ant_world_alg_bench/new-antworld/grid70'
    # # parameters = {'blur': [True], 'segment_l': [3], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    # #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}
    #
    # parameters = {'blur': [True, False], 'shape': [(180, 50)], 'edge_range': [(180, 200), False],
    #               'window': [15, -20], 'matcher': ['mae']}
    #
    # routes = [1, 2]
    # bench = navbench.Benchmark(results_path, routes_path, grid_path, filename='test.csv')
    # bench.benchmark(parameters, routes, parallel=False)
    
    #'segment_length':[3],
    results_path = '/its/home/sk526/ant_world_alg_bench/Results/newant/2022-06-13'
    routes_path = '/its/home/sk526/ant_world_alg_bench/new-antworld/exp1'
    parameters = {'repos_thresh':[.3], 
                  'r': [0.05], 
                  't': [150], 
                  #'blur': [True, False],
                  'shape': [(180, 80)],
                #   'wave' : [True, False], 
                  'edge_range': [(180, 200), False],
                #   'loc_norm': [{'kernel_shape':(5, 5)}, False],
                #   'gauss_loc_norm': [{'sig1':2, 'sig2':20}, False],
                #   'window': [15, 25, 30, -15, 0],
                  'window': [15, 20, 25, -15, 0],
                  'matcher': ['mae', 'corr']
                  }

    routes = [1]
    cbench.benchmark(results_path, routes_path, parameters, routes, parallel=True, cores=1)
if __name__ == "__main__":
    main()
