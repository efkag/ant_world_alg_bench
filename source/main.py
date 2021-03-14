from source import navbench, cbench


def main():
    results_path = '../Results/newant/'
    routes_path = '../new-antworld/exp1/'
    grid_path = '../new-antworld/grid70/'
    # parameters = {'blur': [True], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}

    parameters = {'blur': [True], 'shape': [(180, 50)],
                  'window': [15, 20], 'matcher': ['mae']}

    routes = [1, 2]
    bench = navbench.Benchmark(results_path, routes_path, grid_path, filename='test.csv')
    bench.benchmark(parameters, routes, parallel=True)

    # results_path = '../Results/newant/test.csv'
    # routes_path = '../new-antworld/exp1'
    # parameters = {'r': [0.05], 't': [10], 'segment_l': [3], 'blur': [True],
    #               'shape': [(180, 50), (90, 25)],  # 'edge_range': [(180, 200)],
    #               'window': list(range(10, 12)), 'matcher': ['mae', 'corr']}
    #
    # routes = [1, 2]
    # cbench.benchmark(results_path, routes_path, parameters, routes, parallel=True)


if __name__ == "__main__":
    main()
