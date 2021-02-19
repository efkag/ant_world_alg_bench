from source import alg_bench
from multiprocessing import Process


def main():
    results_path = '../Results/newant/'

    # parameters = {'blur': [True], 'shape': [(180, 50), (90, 25)], 'edge_range': [(180, 200)],
    #               'window': list(range(10, 12)), 'matcher': ['corr', 'rmse']}

    parameters = {'blur': [True], 'shape': [(180, 50)],
                  'window': [5, 10, 15], 'matcher': ['mae', 'corr', 'rmse']}

    routes = [1, 2, 3]
    bench = alg_bench.Benchmark(results_path, filename='wresults.csv')
    bench.benchmark(parameters, routes)

    # bench.bench_paral(parameters, routes)


if __name__ == "__main__":
    main()
