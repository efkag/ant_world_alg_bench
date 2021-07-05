from source import cbench
from source import navbench

def main():
    # results_path = '../Results/newant/'
    # routes_path = '../new-antworld/exp1/'
    # grid_path = '/home/efkag/PycharmProjects/ant_world_alg_bench/new-antworld/grid70'
    # # parameters = {'blur': [True], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    # #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}
    #
    # parameters = {'blur': [True, False], 'shape': [(180, 50)], 'edge_range': [(180, 200), False],
    #               'window': [15, -20], 'matcher': ['mae']}
    #
    # routes = [1, 2]
    # bench = navbench.Benchmark(results_path, routes_path, grid_path, filename='test.csv')
    # bench.benchmark(parameters, routes, parallel=False)

    results_path = '../Results/newant/test2.csv'
    routes_path = '../new-antworld/exp1'
    parameters = {'r': [0.05], 't': [50], 'segment_l': [3], 'blur': [True],
                  'shape': [(180, 50)], 'edge_range': [False],
                  'window': [20], 'matcher': ['corr']}

    routes = [3]
    cbench.benchmark(results_path, routes_path, parameters, routes, parallel=False)
if __name__ == "__main__":
    main()
