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
    results_path = '/its/home/sk526/ant_world_alg_bench/Results/newant/2023-03-29_test'
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
                  'window': [15, 20, 25, -15],
                  'matcher': ['corr'],
                  'w_thresh':[0.05]
                  }

    routes = [0, 1, 5, 7]
    num_of_repeats = 3
    cbench.benchmark(results_path, routes_path, parameters, routes, 
                    parallel=True, num_of_repeats=num_of_repeats)
if __name__ == "__main__":
    main()
